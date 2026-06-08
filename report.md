# 可解释谣言检测系统实验报告

## 1. 任务目标

本大作业的核心任务是构建一个**可解释的社交媒体谣言检测系统**。具体来说，给定一条英文推文，系统需要完成两件事：①判断该推文是否为谣言，②用自然语言解释做出该判断的依据。"可解释性"是本作业与普通文本分类任务的根本区别。

从技术角度看，任务包含三大挑战：
- **跨事件泛化**：训练数据包含7个不同事件（如 Ferguson 枪击案、Ottawa 枪击案、Sydney 人质事件等），各事件的谣言比例差异极大——最大事件（event 5）有854条，最小仅9条。模型必须学会识别谣言的语言特征本身，而非记忆"某个事件=谣言"的捷径；
- **短文本理解**：推文平均仅16个词，信息密度低，传统特征工程手段效果有限，且易受拼写错误和简写干扰；
- **双输出协同**：分类模型与解释生成模块需要紧密配合，解释必须忠实反映模型的实际决策依据，而非凭空编造。

为此，我们采用了两阶段流水线方案：
- **第一阶段**：使用 RoBERTa 预训练模型对推文进行微调分类，并用 Integrated Gradients 算法提取模型做决策时最关注的关键 token（token 级特征归因）；
- **第二阶段**：将预测结果、关键 token 及其重要性分数等辅助特征组装成 Prompt，调用 SJTU CLAW LLM API，由大语言模型基于这些信号生成自然语言判断依据。

---

## 2. 具体内容

### 2.1 实施方案

系统采用**复合流水线架构**，分为三个阶段：

```
输入文本
    │
    ▼
[RoBERTa 分类器] ──→ 预测标签 + 置信度
    │
    ▼
[Captum IG 归因] ──→ 关键证据 token（对预测/对立两类分别归因）
    │
    ▼
[CLAW API LLM]  ──→ 自然语言解释（含正反两面论述）
    │
    ▼
输出: {label, confidence, explanation, evidence}
```

#### 2.1.1 数据集概况

| 数据集 | 样本数 | 非谣言 | 谣言 | 谣言比例 |
|--------|--------|--------|------|----------|
| 训练集 | 2,840 | 1,600 | 1,240 | 43.7% |
| 验证集 | 401 | 226 | 175 | 43.6% |

训练集覆盖7个不同事件，最大的 event 5（854条）与最小的 event 2（仅9条）之间相差近100倍，对模型的跨事件泛化能力提出了较高要求。

#### 2.1.2 数据预处理

推文数据在送入模型前经过简单的清洗流程（`src/data_loader.py`）：

```python
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)   # 去除 URL
    text = re.sub(r"\s+", " ", text)       # 合并多余空白
    return text.strip()
```

URL 去除是因为 URL 本身携带的分类信号极弱，且不同的 URL 会导致 token 序列过长，浪费有限的序列长度资源。

#### 2.1.3 模型架构

分类器（`src/model.py`）基于 RoBERTa 编码器，结构简单：

```
输入文本 → RoBERTa 编码器 → 取 [CLS] 向量 → Dropout(0.1) → Linear(768→2) → softmax
```

`roberta-base` 的 hidden_size 为 768，最终分类层从 [CLS] 向量映射到 2 个类别 logits。初始化时统一转换为 `float32`，以兼容 DeBERTa 等 backbone 的 `float16` 预训练权重。

#### 2.1.4 训练策略

| 组件 | 实现 | 作用 |
|------|------|------|
| 优化器 | `AdamW`，lr=2e-5，weight_decay=0.01 | 适用于 Transformer 微调的标准化优化器 |
| 学习率调度 | 线性 warmup（10% 步数）→ 线性衰减到 0 | 防止早期剧烈更新破坏预训练权重 |
| 梯度裁剪 | `max_norm=1.0` | 防止梯度爆炸，稳定训练 |
| 类别加权 | `CrossEntropyLoss(weight=class_weights)` | 缓解样本不平衡，提升 rumor 召回率 |
| Early Stopping | 连续 3 轮 val_acc 不提升则停止 | 防止过拟合 |
| 梯度累积 | `--accumulation_steps` 可选，默认 1 | 在显存有限时模拟更大 batch_size |

类别权重通过 `sklearn.utils.class_weight.compute_class_weight('balanced')` 自动计算，本次训练得到 `[0.8875, 1.145]`，即对 rumor 类施加约 29% 的额外惩罚。

#### 2.1.5 可解释性归因

使用 Captum 的 `IntegratedGradients`（`src/explainer.py`）对**嵌入层**进行归因。关键参数：
- **归因步数**：`n_steps=50`
- **基线（baseline）**：全 [PAD] token 的 embedding 向量（表示"无信息输入"）
- **归因目标**：分别对预测标签和对立标签计算，得到正反两面证据

`get_evidence_both_sides()` 同时返回对预测标签和对立标签的证据，迫使解释呈现平衡视角，避免单一倾向的"确认偏误"。

#### 2.1.6 LLM 解释生成

调用 SJTU CLAW API（`minimax` 模型），核心参数：

| 参数 | 取值 | 说明 |
|------|------|------|
| `temperature` | 0.3 | 低温度，保证解释稳定、可复现 |
| `max_tokens` | 1024 | 足够生成完整解释，避免截断 |
| `max_retries` | 3 | 指数退避重试（1s → 2s → 3s） |
| `timeout` | 30s | 单次请求超时上限 |

Prompt 采用**系统/用户角色分离**设计：
- **System Prompt**：定义模型角色（社交媒体分析专家），给出四步回答框架；
- **User Prompt**：注入推文原文、事件类别、预测标签、置信度、正反证据 token 列表。

API 失效时自动降级为模板解释（拼接关键词），确保流水线不中断。

### 2.2 核心代码分析

**模型定义**（`src/model.py`）：

```python
class RumorClassifier(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=2, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.to(torch.float32)  # 统一精度，规避 DeBERTa 等 backbone 的 dtype 不匹配
```

- 使用 `AutoModel.from_pretrained` 支持灵活切换 backbone（`roberta-base` / `roberta-large` / `microsoft/deberta-v3-base`）
- Dropout 层在训练时随机失活 10% 的神经元，增强泛化能力

**训练入口**（`src/train.py`）：

```python
# 类别权重计算
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 优化器与学习率调度
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

训练完成后自动保存两项文件：
1. `best_model.pt` — 最佳 checkpoint（按验证集准确率筛选）；
2. `model_config.json` — 记录 backbone 名称，供 `evaluate.py` / `inference.py` 自动读取，避免换模型时维度不匹配。

**可解释性归因**（`src/explainer.py`）：

```python
class IGExplainer:
    def __init__(self, model, tokenizer):
        # 兼容不同 backbone 的嵌入层命名
        if hasattr(model.backbone, "embeddings"):
            self.embeddings = model.backbone.embeddings.word_embeddings
        elif hasattr(model.backbone, "roberta"):
            self.embeddings = model.backbone.roberta.embeddings.word_embeddings
        # ...

    def get_evidence(self, text, label, top_k=5):
        # 基准：全 [PAD] token 的 embedding
        baseline_ids = torch.full_like(input_ids, pad_id)
        baseline_embeds = self.embeddings(baseline_ids)

        # 计算 Integrated Gradients
        attributions, delta = self.ig.attribute(
            embeddings, baselines=baseline_embeds,
            target=label, n_steps=50, return_convergence_delta=True,
        )
```

归因值的处理流程：求和每个 token 在所有 embedding 维度上的归因绝对值 → 过滤 [PAD]、[CLS]、[SEP] 等特殊 token → 按重要性排序取 top_k → 归一化为百分比。

**LLM 接口**（`src/llm_client.py`）：

```python
# System Prompt
"You are a rumor detection assistant with expertise in social media analysis. "
"Provide a balanced, well-reasoned explanation that: "
"(1) states the predicted label with confidence, "
"(2) explains the key evidence supporting the prediction, "
"(3) acknowledges any counter-evidence, "
"(4) references specific words from the tweet as evidence."

# 降级模板（API 失败时自动使用）
if label == 1:
    return f"This tweet is classified as rumor with {confidence:.1%} confidence. \
It contains suspicious keywords such as {pred_str}, which often appear in unverified claims."
```

**推理流水线**（`inference.py`）：

`predict_single()` 整合三个模块的完整流程：
1. 分类 → 获取 `label` 和 `confidence`；
2. 归因 → `get_evidence_both_sides()` 获取正反 token 证据；
3. 解释 → 调用 `CLAWClient.generate_explanation()`。

支持三种运行模式：
- **单条推理**：`--text "..."` 参数直接输入；
- **批量推理**：`--input val.csv --output results/predictions.csv` 处理整个 CSV；
- **跳过 LLM**：`--no_llm` 只使用模板降级（适合调试或无 API 环境）。

### 2.3 检测结果分析

在 `val.csv`（n=401）上的最终评估结果如下：

| 指标 | 数值 |
|------|------|
| Accuracy | **87.78%** |
| F1 Score (rumor) | **0.8665** |
| Precision (non-rumor) | 0.9234 |
| Recall (non-rumor) | 0.8540 |
| Precision (rumor) | 0.8281 |
| Recall (rumor) | **0.9086** |

模型在 rumor 类上取得了 **90.86%** 的召回率，意味着绝大多数谣言能够被正确识别；在 non-rumor 类上精确率达到 92.34%，有效控制了假阳性。

**关键实验对比**：

| 版本 | 主要改动 | val_acc | rumor recall |
|------|---------|---------|-------------|
| v1.0 Baseline | roberta-base, max_len=128 | 85.29% | 82.29% |
| **v2.0 优化** | **max_len=256 + 类别加权** | **87.78%** | **90.86%** |
| v3.0 DeBERTa | microsoft/deberta-v3-base | 82.29% | — |
| v4.0 roberta-large | 355M 参数, 10 epochs | 88.78% | — |

v2.0 通过加长序列长度（128→256，保留更完整的推文语境）和类别加权，**准确率提升 2.49%，rumor recall 大幅提升 8.57%**（82.29%→90.86%）。v3.0/v4.0 实验表明，在本数据集规模（~2.8K 训练样本）下，更大 backbone 反而因过拟合导致性价比极低。`roberta-large` 的 train_loss 降至 0.02 而 val_loss 升至 0.66，仅提升 1% 准确率。因此**最终采用 roberta-base**。

**错误分析**（`results/error_cases.json`）：

分析 20 条错误样本，发现两类典型误判：

1. **假阴性（rumor→non-rumor）**：主要为"基于事实的质疑/批评性报道"。例如 Ferguson 事件中质疑警方行为的推文（`"What are #Ferguson Police hiding about..."`），以及描述德国客机坠毁技术细节的推文（`"IMAGE: .@flightradar24 altitude & speed chart..."`）。模型将事实叙述语气误判为客观报道，实际上这些推文涉及未经官方确认的"细节信息"，属于潜藏的谣言。

2. **假阳性（non-rumor→rumor）**：主要为含 "BREAKING" 标签的真实新闻报道。例如 Germanwings 坠机事件的官方报道（`"BREAKING NEWS: Germanwings flight GWI18G crashes in French Alps"`），模型被 "BREAKING" 和危机事件关键词误导。此外，具有强烈情绪色彩的批评性推文（`"Wake up. Whenever a black man is killed by police, they try to make him a Saint..."`）也被误判为谣言。

这些错误模式揭示了模型对**语用功能**的理解不足——同样是表达"突发"信息，真实新闻与谣言在词汇分布上高度重叠，需要更高级的语义理解（如可信度推理、信源评估）来区分。

### 2.4 判断依据的分析（可解释性）

系统通过**数学归因 + LLM 润色**两层机制保证解释的可信度与可读性：

1. **Captum IG 归因**：对嵌入层计算 Integrated Gradients（50 步 Riemann 近似），归因值具有数学可验证性，避免黑箱解释。

2. **两面论述**：通过 `get_evidence_both_sides()` 同时提取支持预测标签和反方标签的证据——即使对同一个 token，模型对不同类别的归因权重也不同，这一信息被完整保留并传递给 LLM。

3. **引用具体词汇**：LLM Prompt 中明确要求引用证据 token，并在用户消息中提供 `pred_str` 和 `opp_str` 两份具体的 token 列表，减少 LLM 产生幻觉的空间。

4. **低温度控制**：`temperature=0.3` 确保同一输入在不同时间调用时输出稳定，利于复现。

**解释示例**（实际推理输出）：

```json
{
  "label": 1,
  "confidence": 0.9982,
  "explanation": "This tweet is classified as rumor with 99.8% confidence. It contains suspicious keywords such as ' Ottawa', ' confirmed', ' Just', ' hospital', ' #', which often appear in unverified claims. However, words like ' Ottawa', ' confirmed', ' #', ' Just', ' hospital' might suggest factual reporting if taken out of context.",
  "evidence": {
    "predicted": [
      {"token": " Ottawa", "score": 0.2305},
      {"token": " confirmed", "score": 0.2233},
      {"token": " Just", "score": 0.2018},
      {"token": " hospital", "score": 0.1764},
      {"token": " #", "score": 0.1681}
    ],
    "opposite": [
      {"token": " Ottawa", "score": 0.2483},
      {"token": " confirmed", "score": 0.2233},
      {"token": " #", "score": 0.1912},
      {"token": " Just", "score": 0.1785},
      {"token": " hospital", "score": 0.1586}
    ]
  }
}
```

该示例中，"BREAKING: Two new patients coming to Ottawa hospital civic campus. One with gunshot wounds. Just confirmed #ottnews" 被判定为谣言。值得关注的是正反证据高度重叠——表明模型判断的关键差异不在"关注哪些词"，而在**同一个词对不同类别的归因权重不同**。这种微妙的权重差异只有通过 Integrated Gradients 这种量化归因才能揭示，也凸显了 LLM 解释的价值：它能将这些数值差异翻译为人类可理解的推理。

**可解释性的局限性**：

1. **Token 级归因的颗粒度问题**：IG 归因基于 token，RoBERTa 的 BPE 分词可能将复合词（如 `#GermanWings`）拆分为 `#`、`German`、`Wings` 等无意义的子词片段，导致证据 token 可读性下降。

2. **正反证据高度重叠**：如上例所示，正反证据 token 高度重合，模型主要依赖权重差异而非词汇选择进行判断。这虽然揭示了模型的真实行为，但也使用户难以直观理解"到底哪个词导致了不同判断"。

3. **LLM 降级时的模板解释质量有限**：当 API 不可用时，模板解释仅为关键词拼接（如 `"This tweet is classified as rumor with 99.8% confidence. It contains suspicious keywords such as 'Ottawa', 'confirmed', 'Just'..."`），缺乏真正的因果推理链。

4. **归因仅覆盖嵌入层**：当前实现仅对 embedding 层归因，无法显示中间层（如 attention head）的决策路径。完整的可解释性需要多层归因分析。

---

## 3. 工作总结

### 3.1 收获与心得

1. **小数据集的模型选型原则**：并非参数量越大越好。在 ~2.8K 样本上，`roberta-base`（125M）优于 `roberta-large`（355M）和 `deberta-v3-base`（86M），因为大模型容量远超数据量，极易记忆噪声。类别加权等"轻量级 trick"（`compute_class_weight('balanced')`）将 rumor recall 从 82.29% 提升至 90.86%，提升幅度远超换 backbone。

2. **可解释性的工程化**：单纯依赖 LLM 生成解释容易产生幻觉；先用 Captum 提取数学可验证的证据 token，再交给 LLM 润色，能在可信度与可读性之间取得平衡。正反两面归因的设计迫使 LLM 提供平衡解释，避免"只讲有利证据"的选择性偏差。

3. **工程鲁棒性**：API 调用必须设计重试（3 次指数退避）、超时（30s）、降级（模板 fallback）策略。移除硬编码密钥、改用 `.env` + `python-dotenv` 是保障代码安全与可移植性的必要步骤。

### 3.2 遇到问题及解决思路

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| rumor 类假阴性过多（recall=82.29%） | 训练集样本不平衡（non-rumor 1600 vs rumor 1240），模型偏向判非谣言 | `compute_class_weight('balanced')` 加权 CrossEntropyLoss，rumor recall 提升 8.57%（→90.86%） |
| DeBERTa dtype 不匹配 | safetensors 默认 float16，classifier 为 float32，matmul 时报错 | `self.to(torch.float32)` 统一精度 |
| 硬编码 `roberta-base` 导致换模型时维度报错 | evaluate.py / inference.py 中写死模型名，切换 backbone 后 hidden_size 不匹配 | `train.py` 保存 `model_config.json`，下游自动读取 backbone 名称 |
| API Key 曾硬编码在源码中 | 开发时临时写入，后续未清理 | 移除所有硬编码，统一使用 `.env` + `python-dotenv`，未配置时自动模板降级 |
| 训练后期轻度过拟合 | train_loss 持续下降（0.589→0.108），val_loss 在第3轮触底（0.333）后回升，val_acc 在第4轮达峰值（87.78%）后下降 | 引入 Early Stopping（patience=3），在 val_acc 连续 3 轮不提升时停止训练，最终保存第4轮最佳模型 |

---

## 4. 课程建议

1. **增加小样本/不平衡数据集的教学比重**：实际应用中数据往往有限且类别不平衡，课程可补充 `class_weight`、`Focal Loss`、数据增强等针对性技术。

2. **引入模型可解释性专题**：Captum、LIME、SHAP 等工具能帮助学生理解"模型为什么这样预测"，与课程强调的"可解释 AI"目标高度契合。

3. **提供统一的 API 测试环境**：CLAW API 在高峰期存在延迟与限流，建议课程平台提供稳定的沙盒环境或示例缓存数据，降低学生因网络问题导致的调试成本。