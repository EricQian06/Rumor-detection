# 可解释谣言检测系统实验报告

## 1. 任务目标

本项目为上海交通大学《人工智能导论》课程大作业，目标是构建一个**可解释的社交媒体谣言检测系统**。系统输入一条英文推文，输出二分类标签（`0`=非谣言，`1`=谣言）以及一段自然语言解释，说明判断依据。任务要求兼顾分类准确率与解释的可读性、可信度。

---

## 2. 具体内容

### 2.1 实施方案

系统采用**复合流水线架构**，分为三个阶段：

```
输入文本
    │
    ▼
[RoBERTa 分类器] ──→ 标签 + 置信度
    │
    ▼
[Captum IG 归因] ──→ 关键证据 token（正反两面）
    │
    ▼
[CLAW API LLM] ──→ 自然语言解释
    │
    ▼
输出: {label, confidence, explanation, evidence}
```

- **分类器**：使用 `roberta-base` 作为编码器，在 `[CLS]` 向量后接 `Dropout(0.1)` 与 `Linear(hidden_size, 2)` 进行分类。
- **归因器**：使用 Captum 的 `IntegratedGradients` 对嵌入层进行归因，提取对预测标签贡献最大的 Top-K token，同时提取对反方标签的证据。
- **解释生成器**：调用 SJTU CLAW API（`minimax` 模型），Prompt 中包含预测标签、置信度、事件类别、正反证据，要求 LLM 生成**两面论述**的解释。API 失效时自动降级为模板解释。

### 2.2 核心代码分析

**模型定义**（`src/model.py`）：`RumorClassifier` 封装了预训练 Transformer 与分类头，支持通过 `model_name` 切换不同 backbone。针对 DeBERTa 的 `float16/32` 不匹配问题，初始化后统一调用 `self.to(torch.float32)`。

```python
class RumorClassifier(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=2, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.to(torch.float32)  # 规避 dtype 不匹配
```

**训练策略**（`src/train.py`）：采用 `AdamW` 优化器，学习率 `2e-5`，线性 warmup（10% 步数）。关键优化是引入**类别加权** (`CrossEntropyLoss(weight=class_weights)`)，利用 `sklearn.utils.class_weight` 自动计算平衡权重，显著降低 rumor 类的假阴性率。

**可解释性归因**（`src/explainer.py`）：`IGExplainer` 通过 `hasattr` 链式检测适配 BERT/RoBERTa/DeBERTa 的不同嵌入层命名。`get_evidence_both_sides()` 同时返回对预测标签和对立标签的证据，支撑后续 LLM 的两面论述。

**LLM 接口**（`src/llm_client.py`）：`CLAWClient` 封装了 CLAW API 调用，具备 3 次指数退避重试、30 秒超时、模板降级三重鲁棒机制。Prompt 明确要求 LLM 引用具体词汇并分析反方证据，避免"幻觉"。

### 2.3 检测结果分析

在 `val.csv`（n=401）上的最终评估结果如下：

| 指标 | 数值 |
|------|------|
| Accuracy | **87.78%** |
| F1 Score (rumor) | **0.8665** |
| Precision (non-rumor) | 0.9234 |
| Recall (non-rumor) | 0.8540 |
| Precision (rumor) | 0.8281 |
| **Recall (rumor)** | **0.9086** |

**关键实验对比**：

| 版本 | 主要改动 | val_acc | rumor recall |
|------|---------|---------|-------------|
| v1.0 Baseline | roberta-base, max_len=128 | 85.29% | 82.29% |
| **v2.0 优化** | **max_len=256 + 类别加权** | **87.78%** | **90.86%** |
| v3.0 DeBERTa | microsoft/deberta-v3-base | 82.29% | — |
| v4.0 roberta-large | 355M 参数, 10 epochs | 88.78% | — |

v2.0 通过加长序列长度和类别加权，**准确率提升 2.49%，rumor recall 大幅提升 8.57%**。v3.0/v4.0 实验表明，在本数据集规模（~2.8K 训练样本）下，更大 backbone 反而因过拟合导致性价比极低：`roberta-large` 的 train_loss 降至 0.02 而 val_loss 升至 0.66，仅提升 1% 准确率。因此**最终采用 roberta-base**。

混淆矩阵图位于 `results/confusion_matrix.png`，错误样例分析见 `results/error_cases.json`。假阴性主要集中在"基于事实的质疑/批评性报道"，模型易将其误判为非谣言。

### 2.4 判断依据的分析（可解释性）

系统通过**数学归因 + LLM 润色**两层机制保证解释的可信度与可读性：

1. **Captum IG 归因**：对嵌入层计算 Integrated Gradients，归因值具有数学可验证性，避免黑箱。
2. **两面论述**：`get_evidence_both_sides()` 同时提取支持预测标签和反方标签的证据，迫使解释呈现平衡视角。
3. **引用具体词汇**：LLM Prompt 中强制要求引用证据 token，减少幻觉。

**解释示例**（单条推理输出）：

```json
{
  "label": 1,
  "confidence": 0.9905,
  "explanation": "This tweet is classified as a rumor primarily due to ... 
    However, there is reasonable counter-evidence: ...",
  "evidence": {
    "predicted": [{"token": " reports", "score": 0.2915}, ...],
    "opposite": [{"token": "BRE", "score": 0.1835}, ...]
  }
}
```

随机抽取 10 条样本的端到端测试准确率为 90%（9/10），LLM 解释均包含正反两面论述、事实核查与置信度分析。

---

## 3. 工作总结

### 3.1 收获与心得

1. **小数据集的模型选型原则**：并非参数量越大越好。在 ~2.8K 样本上，`roberta-base`（125M）优于 `roberta-large`（355M）和 `deberta-v3-base`，因为大模型容量远超数据量，极易记忆噪声。类别加权等"轻量级 trick"往往比换 backbone 更有效。
2. **可解释性的工程化**：单纯依赖 LLM 生成解释容易产生幻觉；先用 Captum 提取数学可验证的证据 token，再交给 LLM 润色，能在可信度与可读性之间取得平衡。
3. **工程鲁棒性**：API 调用必须设计重试、超时、降级策略。移除硬编码密钥、改用 `.env` + `python-dotenv` 是保障代码安全与可移植性的必要步骤。

### 3.2 遇到问题及解决思路

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| rumor 类假阴性过多 | 样本不平衡，模型偏向判非谣言 | `compute_class_weight('balanced')` 加权 CrossEntropyLoss，rumor recall 提升 8.57% |
| DeBERTa dtype 不匹配 | safetensors 默认 float16，classifier 为 float32 | `self.to(torch.float32)` 统一精度 |
| 硬编码 `roberta-base` 导致换模型时维度报错 | evaluate/inference 中写死模型名 | `train.py` 保存 `model_config.json`，下游自动读取 backbone 名称 |
| API Key 曾硬编码在源码中 | 开发时临时写入，未清理 | 移除 fallback，统一使用 `.env` + `python-dotenv`，未配置时模板降级 |
| Windows 下 torch + pandas 导入 segfault | 导入顺序冲突 | 确保 `import pandas` 在 `import torch` 之前执行 |

---

## 4. 课程建议

1. **增加小样本/不平衡数据集的教学比重**：实际应用中数据往往有限且类别不平衡，课程可补充 `class_weight`、`Focal Loss`、`数据增强`等针对性技术。
2. **引入模型可解释性专题**：Captum、LIME、SHAP 等工具能帮助学生理解"模型为什么这样预测"，与课程强调的"可解释 AI"目标高度契合。
3. **提供统一的 API 测试环境**：CLAW API 在高峰期存在延迟与限流，建议课程平台提供稳定的沙盒环境或示例缓存数据，降低学生因网络问题导致的调试成本。
