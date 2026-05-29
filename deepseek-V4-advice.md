# 人工智能导论大作业——可解释谣言检测：完整思路与分工建议

> 基于对 `人工智能导论大作业2026.pdf`、`rumer2026/train.csv`、`rumer2026/val.csv` 的详细分析。
> 生成日期：2026-05-29

---

## 一、数据集分析摘要

| 维度 | 训练集 (train.csv) | 验证集 (val.csv) |
|------|-------------------|------------------|
| 样本数 | 2,840 | 401 |
| 列 | id, text, label, event | 同 |
| 类别分布 | 0: 1,600 (56.3%), 1: 1,240 (43.7%) | 0: 226 (56.4%), 1: 175 (43.6%) |
| 事件数 | 7个 (event 0–6) | 7个 (event 0–6) |
| 文本长度 | 3–31词，平均16.3词 | 3–28词，平均16.3词 |

### 七个事件的详细分布（训练集）

| 事件 | 推测主题 | 样本数 | 谣言占比 | 备注 |
|------|----------|--------|----------|------|
| 0 | Gurlitt 纳粹艺术品争议 | 66 | 20% | 小样本，非谣言为主 |
| 1 | Ferguson / Mike Brown 枪击案 | 799 | 25% | 大样本，非谣言为主 |
| 2 | Essien 埃博拉谣言 | 9 | 100% | ⚠️ 极小样本，完全不平衡 |
| 3 | Prince 多伦多秘密演唱会 | 162 | 99% | ⚠️ 高度偏向谣言 |
| 4 | Germanwings 空难 | 327 | 51% | 相对均衡 |
| 5 | 悉尼人质危机 | 854 | 43% | 训练集最大事件 |
| 6 | 渥太华枪击案 | 623 | 53% | 相对均衡 |

### 关键挑战

1. **跨事件泛化**：训练集与验证集均包含 event 0–6，但不同事件的谣言比例差异极大，模型容易学到事件特定的虚假相关性。
2. **严重不平衡**：Event 2（9条，100%谣言）和 Event 3（162条，99%谣言）几乎全为单类，会导致模型过拟合到"某些event = 必然谣言"的捷径。
3. **文本极短**：平均仅16.3词，传统NLP特征（TF-IDF等）信息密度有限，需要依赖预训练语言模型的上下文理解能力。
4. **双输出要求**：需要同时输出分类标签 + 自然语言解释依据，这是普通文本分类任务不具备的挑战。

---

## 二、整体思路与流程（详细步骤）

### 阶段1：数据理解与探索性分析（EDA）

**1.1 数据统计分析**
- 文本长度分布直方图
- 各事件的高频词云（按谣言/非谣言分组）
- 标签分布可视化（全局 + 按事件堆叠柱状图）
- 各事件的谣言-非谣言语言差异（情感极性、标点使用、大写比例、hashtag数量、mention数量、URL出现率等）

**1.2 文本预处理管线**
- 去除 URL（`http://t.co/...`）和 HTML 实体（`&amp;`）
- **保留** hashtag 文本（如 `#Ferguson` → `Ferguson`，hashtag本身是强特征）和 mention（`@user`）
- 处理 emoji（转为文本描述，如 😭 → `loudly crying face`，或使用 emoji 库）
- 统一小写（但保留大写词比例作为单独特征，因为全大写暗示情绪强度）
- 可选：去除停用词（对于 BERT 类模型通常不建议，因为预训练时保留了停用词的语义）

**1.3 特征工程（为基线模型和LLM解释提供补充信号）**
- **语义特征**：预训练语言模型的 [CLS] embedding（RoBERTa / BERTweet）
- **词汇特征**：TF-IDF unigram + bigram + trigram
- **情感特征**：VADER 情感得分（正/负/中/复合）
- **元特征**：是否含 URL、hashtag 数量、mention 数量、大写词比例、问号/感叹号数量、推文长度
- **主题特征**：LDA 或 BERTopic 提取的主题分布

### 阶段2：深度学习检测模型设计与训练

#### 推荐方案：BERTweet / RoBERTa 微调 + 事件对抗训练

```
输入 Tweet → BERTweet/RoBERTa → [CLS] embedding → 分类头(MLP) → rumor/non-rumor
```

**关键步骤：**

1. **模型选择**：
   - 首选 `vinai/bertweet-base`：专为英文Twitter文本预训练，能理解 `@user`、`#hashtag`、emoji等Twitter特定语言
   - 备选 `roberta-base`：通用性强，community支持好
   - 如要轻量化可选 `distilroberta-base`

2. **输入构造**：
   - 格式：`[CLS] text [SEP]`，max_len 设为 64–128（文本很短，不需要512）
   - 如使用 BERTweet，需用其自带的 tokenizer（处理 `@` 和 `#`）

3. **训练策略**：
   - **分层 K 折交叉验证**：使用 `StratifiedKFold` 按 event 分层（确保每折包含所有event），K=5
   - **类别加权**：在损失函数中对谣言/非谣言按样本比例加权，缓解轻度不平衡
   - **Focal Loss**（可选）：`FocalLoss(gamma=2)` 进一步降低易分样本权重，聚焦难分样本
   - **域对抗训练（关键）**：在 [CLS] embedding 后加一个梯度反转层(GRL) + 事件分类器，让编码器学到的表示不被 event 影响。这是提升泛化能力的核心手段。
   - **优化器**：AdamW，learning_rate=2e-5，warmup_ratio=0.1，linear decay
   - **正则化**：Dropout=0.1，weight_decay=0.01，early_stopping patience=3

4. **评估指标**：
   - 主要：Accuracy（评分标准明确要求）
   - 辅助：Macro-F1（应对类别不平衡）、Per-event Accuracy（评估跨事件表现）

#### 训练伪代码框架

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold

# 加载数据
df = pd.read_csv("train.csv")

# 分层K折（按event分层）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['event'])):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # tokenize
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    train_enc = tokenizer(train_df['text'].tolist(), truncation=True, 
                          padding=True, max_length=128)
    val_enc = tokenizer(val_df['text'].tolist(), truncation=True, 
                        padding=True, max_length=128)
    
    # 训练（使用 HuggingFace Trainer 或 纯 PyTorch 训练循环）
    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/bertweet-base", num_labels=2
    )
    # ... 训练代码 ...
    
    # 记录每折的 val 指标
```

### 阶段3：可解释性——判断依据生成

这是本项目与普通分类任务的核心区别。建议采用 **两阶段流水线**：

#### 3.1 第一阶段：从检测模型中提取关键信号

| 方法 | 原理 | 优势 | 劣势 |
|------|------|------|------|
| **Integrated Gradients** | 从基线到输入的梯度积分 | 理论完备，忠实度高 | 计算略慢 |
| **Attention 权重** | 取最后一层多头注意力的平均权重 | 直观，计算快 | 忠实度有争议 |
| **SHAP (Partition SHAP)** | 基于Shapley值的特征归因 | 公认可靠 | 速度较慢 |
| **LIME** | 局部线性代理模型 | 灵活 | 对短文本不够稳定 |

**推荐组合**：Integrated Gradients（Captum库）提取 token 级重要性 + 情感特征作为补充信号。

```python
from captum.attr import IntegratedGradients

def explain_prediction(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                       max_length=128).to(device)
    ig = IntegratedGradients(model)
    attributions = ig.attribute(inputs['input_ids'], target=predicted_class)
    # 获取每个token的归因分数
    scores = attributions.sum(dim=-1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    # 返回 (token, score) 对，按重要性排序
    return sorted(zip(tokens, scores.tolist()), key=lambda x: abs(x[1]), reverse=True)
```

#### 3.2 第二阶段：用 LLM 生成自然语言解释

**工作流程：**

```
输入推文 → [检测模型] → (预测标签, 置信度, 关键token及重要性分数, 情感特征)
→ 构造 Prompt → [SJTU LLM API] → 自然语言解释
```

**Prompt 模板（中文，输出判断依据）：**

```
你是一个社交媒体谣言检测专家。请根据以下信息，用中文给出你的判断依据。

【推文内容】
{tweet_text}

【模型预测结果】
分类：{谣言/非谣言}（置信度：{confidence:.2f}）

【模型关注的关键词（按重要性排序）】
{key_tokens_with_scores}

【辅助特征】
- 情感极性：{sentiment}（范围-1到1）
- 是否含有外部链接：{has_url}
- Hashtag 数量：{hashtag_count}

请用2-3句话解释判断依据，要求：
1. 指出推文中支持该判断的关键语言特征
2. 分析这些特征为什么指向谣言或非谣言
3. 语言简洁清晰，面向普通社交媒体用户
4. 使用中文回答

判断依据：
```

**进阶做法（推荐）——思维链验证：**

1. 先让 LLM **不看模型预测结果**，独立对推文进行谣言判断并给出理由
2. 再让 LLM **看到模型预测和关键token**，生成最终解释
3. 如果两者判断一致 → 高置信度解释，可直接使用
4. 如果两者不一致 → 标记为"需要人工复核"，解释中说明分歧点

这样做的好处：
- 提升解释的可靠性
- 可发现模型的潜在错误模式
- 体现了"可解释AI"的研究深度

**关于 SJTU API 集成：**

按照 PDF 要求，LLM 部分必须使用 `https://claw.sjtu.edu.cn/guide/sjtu-api/` 提供的接口。需要注意：

- 提前测试 API 的可达性和速率限制
- 封装一个统一的 `call_llm(prompt, system_prompt)` 函数
- 对 val.csv 的 401 条样本做批量调用时控制并发（建议并发数 ≤ 5）
- 将 LLM 输出缓存到本地文件，方便复现和调试
- 如果 API 不可用，准备一个 fallback 方案（如基于规则模板的解释生成）

### 阶段4：模型评估与泛化性验证

#### 4.1 标准评估（在 val.csv 上）

- **分类准确率 (Accuracy)**：这是评分标准15分的直接依据
- **混淆矩阵**：展示各类别的分类情况
- **精确率/召回率/Macro-F1**：全面衡量模型性能
- **ROC-AUC**：衡量模型在不同阈值下的整体表现

#### 4.2 跨事件泛化评估（关键加分项）

这是展示"泛化能力"的核心实验。建议做 **Leave-One-Event-Out (LOEO) 交叉验证**：

```
For each event e in {0, 1, 2, 3, 4, 5, 6}:
    train_model(on all events EXCEPT e)
    evaluate_model(on event e)
    report per-event accuracy
```

报告一张表格：

| 留出事件 | 训练样本数 | 测试样本数 | Accuracy | F1 |
|----------|-----------|-----------|----------|-----|
| 0 | 2,774 | 66 | — | — |
| 1 | 2,041 | 799 | — | — |
| ... | ... | ... | ... | ... |
| **平均** | — | — | — | — |

#### 4.3 消融实验（Ablation Study）

| 实验 | 配置 | 目的 |
|------|------|------|
| 完整模型 | BERTweet + 域对抗 + 元特征 | 基准性能 |
| - 域对抗 | 去掉梯度反转层 | 验证对抗训练对泛化的贡献 |
| - 元特征 | 只使用文本输入 | 验证元特征的贡献 |
| BERT vs RoBERTa vs BERTweet | 替换预训练模型 | 验证模型选择的合理性 |
| 只用传统特征 | TF-IDF + 逻辑回归 | 基线对比，证明深度学习的必要性 |

#### 4.4 可解释性评估

- **忠实度 (Faithfulness)**：随机修改模型关注的前K个关键词，观察预测是否改变（改变越多 = 解释越忠实）
- **人工评估**：每组评估20-30条解释，按1-5分对"合理性""有用性""清晰度"打分
- **与LLM独立判断的一致性**：如前所述，一致性高说明解释可信

### 阶段5：报告撰写与代码整理

按照 `人工智能导论大作业模板2026.doc` 的格式撰写报告。报告建议结构：

1. **问题描述**（~200字）：谣言检测的定义和意义
2. **数据分析**（~300字）：数据集统计、事件分布、关键挑战
3. **模型设计**（~400字）：整体架构图、检测模型设计、解释生成设计
4. **实验与结果**（~500字）：主实验结果表、LOEO结果、消融实验、解释样例
5. **分工与总结**（~200字）：成员贡献、经验教训

代码仓库目录结构建议：

```
.
├── README.md                 # 环境配置、运行说明
├── report.pdf                # 大作业报告
├── requirements.txt          # Python 依赖
├── data/
│   ├── train.csv             # 原始训练数据
│   └── val.csv               # 原始验证数据
├── src/
│   ├── preprocess.py         # 数据预处理
│   ├── eda.py                # 探索性数据分析
│   ├── model.py              # 检测模型定义
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   ├── explain.py            # 可解释性模块（特征归因 + LLM解释生成）
│   ├── llm_api.py            # SJTU LLM API 封装
│   └── utils.py              # 工具函数
├── experiments/
│   ├── loeo_cv.py            # Leave-One-Event-Out 交叉验证
│   └── ablation.py           # 消融实验
├── outputs/
│   ├── models/               # 保存的训练模型
│   ├── explanations/         # LLM 生成的解释结果
│   └── figures/              # 可视化图表
└── docs/
    └── design_notes.md       # 设计决策记录
```

---

## 三、四人分工建议

| 角色 | 成员 | 核心职责 | 关键技术栈 | 产出物 |
|------|------|----------|------------|--------|
| **组长：项目经理 + 算法架构** | 成员A | ①任务统筹、进度管理、代码仓库维护 ②模型整体架构设计与流水线搭建 ③BERTweet/RoBERTa 检测模型的训练与调优 ④各模块接口定义与集成 ⑤报告统筹与最终提交 | PyTorch, HuggingFace Transformers, Git | 模型训练代码、集成流水线、报告终稿 |
| **数据工程师 + 特征工程** | 成员B | ①数据EDA与可视化（图表用于报告）②文本预处理管线实现 ③传统特征提取（TF-IDF、n-gram、情感分析、元特征）④数据增强（应对Event 2/3的不平衡）⑤LOEO跨事件评估框架实现 ⑥报告数据分析章节 | Pandas, NLTK/spaCy, VADER, Matplotlib/Seaborn | 预处理模块、EDA图表、LOEO评估结果 |
| **可解释性 + LLM 集成** | 成员C | ①Integrated Gradients/Attention特征归因实现 ②LLM Prompt设计与迭代优化 ③SJTU API适配与调用封装 ④自然语言解释的批量生成与缓存 ⑤解释质量的人工评估设计 ⑥报告可解释性方法章节 | Captum/SHAP, LLM API, Prompt Engineering | 解释生成模块、Prompt模板、解释质量评估报告 |
| **模型评估 + 消融实验 + 文档** | 成员D | ①全维度模型评估（Accuracy、F1、混淆矩阵、ROC-AUC）②消融实验设计与执行 ③不同预训练模型的对比实验 ④README.md 撰写与部署说明 ⑤实验记录整理与结果可视化 ⑥报告实验与分析章节 | scikit-learn, Captum, Weights & Biases (可选) | 评估报告、消融实验结果、README、部署文档 |

### 分工时间线（建议4周）

```
第1周 ─── 全员EDA与数据理解 ───
  A: 搭建Git仓库，初始化项目结构，跑通基线RoBERTa
  B: 数据分析脚本 + 可视化，输出EDA报告
  C: 调研SJTU API文档，跑通一个LLM调用demo
  D: 搭建评估框架（评估函数 + 可视化），撰写README大纲

第2周 ─── 核心模块并行开发 ───
  A: 训练检测模型（K折交叉验证），加入域对抗训练
  B: 完成预处理管线 + 元特征提取 + LOEO框架
  C: 实现特征归因模块（Integrated Gradients），设计第一版Prompt
  D: 跑基线模型评估，开始对比实验（BERT vs RoBERTa）

第3周 ─── 集成与迭代优化 ───
  A: 模型调优（超参搜索），与C对接模型输出接口
  B: 完成数据增强，支撑LOEO实验
  C: 批量生成解释，进行解释质量人工评估，迭代Prompt
  D: 完成所有消融实验，整理最终实验结果表格和图表

第4周 ─── 收尾与提交 ───
  A: 最终集成测试，确保流水线可运行，审核所有代码
  B: 完善报告数据分析章节 + 图表排版
  C: 完善报告可解释性章节 + 解释样例展示
  D: 完成报告实验分析章节，检查README可复现性
  全员: 交叉审阅报告，组长最终提交
```

### 协作关键接口点

| 接口 | 涉及成员 | 约定内容 |
|------|----------|----------|
| **模型输出 → 解释模块** | A ↔ C | 模型输出的数据格式：`{pred_label, confidence, token_attributions: [(token, score), ...], sentiment, has_url}` |
| **数据预处理 → 模型输入** | B → A | 预处理后的数据格式：`{text_clean, features_dict, label}` |
| **LLM API 封装** | C → 全员 | 统一的 `call_llm(prompt: str) → str` 函数签名 |
| **评估函数接口** | D → A | `evaluate(model, dataloader) → dict` 返回所有指标 |
| **LOEO实验** | B ↔ D | B提供数据切分，D运行评估脚本 |

---

## 四、关键技术风险与应对

| 风险 | 概率 | 应对策略 |
|------|------|----------|
| Event 2 仅9条样本且全为谣言，模型无法学到区分 | 高 | 将Event 2与主题相似的Event合并（或直接不作为独立事件的训练对象）；使用回译(back-translation)数据增强生成非谣言变体；训练时使用Focal Loss降低简单样本的影响 |
| Event 3 几乎全是谣言（99%），缺乏非谣言样本 | 高 | 同上；手动撰写2-3条非谣言变体作为few-shot补充；在解释中标注"该event训练数据偏向谣言" |
| LLM API不可达或速率限制过严 | 中 | 实现本地缓存机制（每个tweet+prediction的hash → 解释文本）；准备基于规则模板的fallback解释方案 |
| 跨事件泛化不佳（依赖event-specific词汇） | 中 | 域对抗训练；去除event-specific命名实体；多任务学习中加入event分类作为正则化 |
| 四人代码冲突频繁 | 低 | 使用Git分支策略：每人一个feature分支，组长负责merge + code review；定期（每周2次）同步会议 |
| 推文太短，解释缺乏充分依据 | 中 | LLM解释中加入"由于推文长度有限，该判断主要基于X和Y两个关键信号，建议结合更多上下文综合判断"的免责说明 |

---

## 五、评分标准对应的注意事项

| 评分项 | 分值 | 拿分关键 |
|--------|------|----------|
| 大作业报告叙述清楚 | 30 | 遵循模板结构，图表清晰，逻辑通畅，不超过2000字 |
| 代码可运行，部署说明清楚 | 25 | README中写清楚环境（Python版本、依赖库版本）、一行命令安装依赖、一行命令运行 |
| 分类准确率 | 15 | 在val.csv上达到较高准确率；建议目标 > 0.75（考虑到部分事件不平衡，0.75-0.85是合理区间） |
| 可解释性 | 15 | 不仅要有解释，还要评估解释的质量和忠实度；展示2-3个有代表性的解释样例（好/中/差各一个并分析） |
| 小组分工协作 | 15 | Git commit历史清晰反映各成员贡献；报告中明确列出分工；团队成员互评 |

---

## 六、推荐技术栈速查

| 用途 | 推荐库/工具 | 版本 |
|------|------------|------|
| 深度学习框架 | PyTorch | ≥ 2.0 |
| 预训练模型 | HuggingFace Transformers | ≥ 4.30 |
| Twitter预训练模型 | `vinai/bertweet-base` | — |
| 特征归因 | Captum | ≥ 0.6 |
| 情感分析 | VADER (nltk.sentiment) | — |
| 传统特征 | scikit-learn | ≥ 1.3 |
| LLM API | requests / httpx / openai SDK | — |
| 数据可视化 | Matplotlib + Seaborn | — |
| 实验管理 | Weights & Biases (可选) | — |
| Python | 3.9 / 3.10 / 3.11 | — |
