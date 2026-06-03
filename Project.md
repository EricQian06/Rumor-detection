# Project 过程记录

> 本文件记录项目从方案设计到最终交付的完整过程，包括代码框架、实验结果、调整方法等。**每次修改后必须同步更新此文件。**

---

## 一、项目背景与方案

### 1.1 任务目标
基于 `train.csv` / `val.csv` 构建可解释的谣言检测模型：
- 输入：英文推文（`text`）
- 输出1：二分类标签（`0`=非谣言，`1`=谣言）
- 输出2：自然语言解释（判断依据）

### 1.2 技术方案：复合流水线

```
输入文本
    │
    ▼
[RoBERTa 分类器] ──→ label ∈ {0, 1}
    │
    ▼
[Captum IG 归因] ──→ evidence_tokens (Top-K)
    │
    ▼
[CLAW API LLM] ──→ explanation (str)
    │
    ▼
输出: {"label": int, "explanation": str}
```

### 1.3 四人分工

| 同学 | 职责 | 核心文件 |
|------|------|----------|
| 荣 | 数据与 Baseline | `src/data_loader.py`, `eda.ipynb`, `baseline.py` |
| 刘 | 核心分类模型 | `src/model.py`, `src/train.py`, `src/evaluate.py` |
| 钱 | 可解释性与 LLM | `src/explainer.py`, `src/llm_client.py` |
| 贺 | 系统集成与报告 | `inference.py`, `README.md`, `report.pdf` |

---

## 二、代码框架（v1.0）

### 2.1 文件清单

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 项目依赖（torch, transformers, captum, sklearn 等） |
| `src/data_loader.py` | `RumorDataset`, `get_dataloaders`, 文本清洗 |
| `src/model.py` | `RumorClassifier`（RoBERTa backbone + Linear head） |
| `src/train.py` | 训练脚本：AdamW + warmup + 早停 + 保存最佳模型 |
| `src/evaluate.py` | 评估脚本：准确率、F1、混淆矩阵、错误样例分析 |
| `src/explainer.py` | `IGExplainer`：Captum Integrated Gradients 归因 |
| `src/llm_client.py` | `CLAWClient`：封装 CLAW API + 重试 + 模板降级 |
| `inference.py` | 端到端推理入口（单条/批量） |
| `test_sample.py` | 随机 10 条样本快速测试脚本 |
| `README.md` | 部署说明、Git 规范、CLAUDE.md 使用说明 |
| `分工.md` | 详细技术方案与模块接口契约 |
| `CLAUDE.md` | Claude Code 仓库级提示词（含开发规范） |

### 2.2 关键设计决策

- **模型选择**：`roberta-base`（预训练于海量英文文本，微调即可适应谣言检测）
- **可解释性**：先用 Captum IG 提取关键 token（数学可验证），再用 LLM 润色（可读性强）
- **鲁棒性**：LLM 失效时自动降级为模板解释，批量推理限速防限流
- **安全**：API Key 必须走环境变量，禁止硬编码

---

## 三、模型训练与测试记录

### 3.1 运行命令

```bash
# 训练
python -m src.train --train_csv train.csv --val_csv val.csv --epochs 5 --batch_size 32 --lr 2e-5

# 评估
python -m src.evaluate --model_dir checkpoints --val_csv val.csv
```

### 3.2 实验结果

**v1.0 Baseline（roberta-base，CPU 训练，2 epochs 后停止）**

| epoch | train_loss | val_loss | val_acc    | 备注      |
| ----- | ---------- | -------- | ---------- | ------- |
| 1     | 0.5704     | 0.4642   | **0.7955** | 最佳模型已保存 |
| 2     | 0.3763     | 0.3888   | **0.8529** | 最佳模型已更新 |

**最终评估结果（val.csv，n=401）**：

| 指标 | 值 |
|------|-----|
| Accuracy | **85.29%** |
| F1 Score (rumor) | **0.8300** |
| Precision (non-rumor) | 0.8646 |
| Recall (non-rumor) | 0.8761 |
| Precision (rumor) | 0.8372 |
| Recall (rumor) | 0.8229 |

---

**v2.0 优化（roberta-base，GPU 训练，max_len=256 + 类别加权）**

运行命令：
```bash
python -m src.train --train_csv train.csv --val_csv val.csv --epochs 5 --batch_size 32 --lr 2e-5 --num_workers 4
```

关键改动：
- `max_len=256`（原 128），避免长推文截断
- 类别加权：`CrossEntropyLoss(weight=[0.8875, 1.1452])`，提高 rumor 类惩罚以减少假阴性
- GPU 训练（RTX 4060，CUDA 12.4）

| epoch | train_loss | val_loss | val_acc    | 备注      |
| ----- | ---------- | -------- | ---------- | ------- |
| 1     | 0.5893     | 0.5088   | 0.7706     | 初期权重干扰，acc 略降 |
| 2     | 0.3554     | 0.4093   | **0.8404** | 快速收敛 |
| 3     | 0.2472     | 0.3328   | **0.8653** | 持续上升 |
| 4     | 0.1543     | 0.3838   | **0.8778** | **最佳模型** |
| 5     | 0.1077     | 0.3896   | 0.8703     | 轻微过拟合 |

**最终评估结果（val.csv，n=401）**：

| 指标 | v1.0 | v2.0 | 变化 |
|------|------|------|------|
| Accuracy | **85.29%** | **87.78%** | **+2.49%** |
| F1 Score (rumor) | 0.8300 | **0.8665** | **+3.65%** |
| Precision (non-rumor) | 0.8646 | 0.9234 | +5.88% |
| Recall (non-rumor) | 0.8761 | 0.8540 | -2.21% |
| Precision (rumor) | 0.8372 | 0.8281 | -0.91% |
| Recall (rumor) | 0.8229 | **0.9086** | **+8.57%** |

混淆矩阵图：`results/confusion_matrix.png`
错误样例：`results/error_cases.json`

> **结论**：准确率从 85.29% 提升至 **87.78%**（+2.49%）。**rumor recall 大幅提升至 90.86%**（+8.57%），假阴性显著减少，类别加权策略效果显著。当前性能处于课程项目**优秀区间上限**，后续如需进一步提升可尝试 `deberta-v3-base`。

### 3.3 错误分析

从 `results/error_cases.json` 抽查发现，模型的主要错误模式是：

| 错误类型 | 典型案例 | 原因分析 |
|----------|----------|----------|
| **假阴性**（rumor→non-rumor） | "#Ferguson police are embarking on what can only be described as an elaborate smear campaign..." | 模型将"基于真实事件的批评性报道"误判为非谣言，因为这些文本使用了事实性语言、具体人名/地点，缺少夸张词汇。 |
| **假阴性**（rumor→non-rumor） | "Lawyers for police in bad shootings often advise shooters not to write reports..." | 类似地，带有法律/制度分析色彩的文本被误判。 |

**核心问题**：模型学到的"rumor"模式偏向"夸张、未证实、情绪化"，而对"基于事实的阴谋论/质疑"识别不足。

### 3.4 调整方法与记录

**当前状态**：准确率 **87.78%**（GPU + max_len=256 + 类别加权）。以下优化方向可进一步提升至 **90%+**：

| 优化方向           | 预期提升    | 具体做法                                                                   |
| -------------- | ------- | ---------------------------------------------------------------------- |
| **换更强模型**      | +2~4%   | 使用 `roberta-large` 或 `deberta-v3-base` 替代 `roberta-base`。参数量更大，语义理解更深。 |
| **加长 max_len** | +1~2%   | 当前 `max_len=128`，部分推文被截断。可尝试 `256`，让模型看到完整上下文。                         |
| **类别加权**       | +1~2%   | 使用 `CrossEntropyLoss(weight=[1.0, 1.2])` 提高对 rumor 类的惩罚，减少假阴性。         |
| **数据增强**       | +1~2%   | 对 rumor 类做同义词替换、回译（back-translation），扩充少数类样本。                          |
| **集成学习**       | +1~3%   | 训练 3 个不同 seed 的模型，投票决定最终标签。                                            |
| **特征工程**       | +0.5~1% | 加入额外特征：文本长度、感叹号数量、URL 数量、是否全大写等。                                       |

**优先级建议**：
1. **高优先级**：换 `roberta-large`（效果最明显，但需 GPU，显存 > 8G）
2. **中优先级**：加长 `max_len` 到 256 + 类别加权（易实现，几乎无额外成本）
3. **低优先级**：数据增强、集成学习（投入产出比相对较低）

**已尝试 / 未尝试**：
- [x] roberta-base baseline（85.29%）
- [x] max_len=256 + 类别加权（87.78%）
- [ ] roberta-large
- [ ] deberta-v3-base
- [ ] 数据增强
- [ ] 集成学习

---

## 四、文档更新日志

| 日期 | 更新内容 | 更新人 |
|------|----------|--------|
| 2026/05/31 | 创建项目方案、分工.md、README.md、CLAUDE.md、.gitignore | Claude |
| 2026/05/31 | 创建完整代码框架（8个源码文件 + requirements.txt） | Claude |
| 2026/05/31 | 创建 Project.md，汇总全过程 | Claude |
| 2026/05/31 | 运行训练与评估：roberta-base 基线达到 85.29% 准确率，完成错误分析 | Claude |
| 2026/05/31 | 安装 PyTorch CUDA 版，修复 Windows segfault（torch 先于 pandas 导入），实施 max_len=256 + 类别加权，GPU 训练 5 epochs 达到 **87.78%** | Claude |
| 2026/05/31 | 实现 LLM 可解释性模块：对接 SJTU CLAW API（minimax 模型），Prompt Engineering 加入正反证据 + 置信度 + event 类别，解释支持正反两面论述 | Claude |
| 2026/05/31 | 从 val.csv 随机抽取 10 条样本进行端到端测试，准确率 **90%**（9/10），LLM 解释质量良好（含正反两面论述、事实核查、置信度分析） | Claude |
| 2026/06/02 | **安全改造**：移除 `inference.py` / `test_sample.py` 中硬编码的 API Key fallback；引入 `python-dotenv` + `.env` 文件支持；`.gitignore` 新增 `.env`；未配置 Key 时自动降级为模板解释 | Claude |
| 2026/06/02 | 端到端验证 `.env` 配置生效：2 条 val.csv 样本推理全部正确，LLM 生成解释质量显著高于模板降级 | Claude |

---

## 五、迁移到另一台电脑快速开始

> 若你需要在另一台电脑（如实验室服务器、队友电脑）上继续优化，按以下步骤操作，**30 分钟内可跑通训练和评估**。

### 5.1 环境准备

```bash
# 1. 克隆仓库（或复制项目文件夹）
git clone <your-repo-url>
# 或直接从 U 盘 / 网盘复制

# 2. 进入目录
cd Rumor-detection

# 3. 创建虚拟环境（强烈推荐，避免依赖冲突）
python -m venv venv

# 4. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 5. 安装依赖（~5-10 分钟，首次会自动下载 roberta-base 权重）
pip install -r requirements.txt
```

### 5.2 数据与文件确认

确保以下文件存在于项目根目录：

```
Rumor-detection/
├── train.csv               # 训练集
├── val.csv                 # 验证集
├── src/                    # 源码目录
├── inference.py            # 推理入口
├── requirements.txt        # 依赖
└── checkpoints/            # （训练后生成）模型权重
```

### 5.3 一键运行训练 + 评估

```bash
# 训练（GPU 约 8 分钟/epoch，CPU 约 30 分钟/epoch）
python -m src.train --train_csv train.csv --val_csv val.csv --epochs 5 --batch_size 32 --lr 2e-5 --num_workers 4

# 评估（输出混淆矩阵、错误样例）
python -m src.evaluate --model_dir checkpoints --val_csv val.csv --max_len 256

# 单条推理测试
python inference.py --text "BREAKING: Unconfirmed reports say aliens landed in Shanghai." --event 0

# 快速测试（随机 10 条样本）
python test_sample.py
```

### 5.4 GPU 加速（强烈推荐）

如果另一台电脑有 NVIDIA GPU：

```bash
# 先卸载 CPU 版 torch，安装 CUDA 版（以 CUDA 12.4 为例）
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

### 5.5 常见问题排查

| 问题 | 解决 |
|------|------|
| `ModuleNotFoundError` | 确认虚拟环境已激活，`pip install -r requirements.txt` 已执行 |
| `CUDA out of memory` | 减小 `--batch_size`（如 16 或 8），或换 `roberta-base` |
| 下载 HuggingFace 模型慢 | 设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`（Linux）或 `set HF_ENDPOINT=https://hf-mirror.com`（Windows CMD） |
| 训练极慢（CPU） | 这是正常的，5 epochs 可能需要 1-2 小时。建议找有 GPU 的机器 |

---

## 六、后续优化详细方案

以下是从当前 **87.78%** 提升到 **90%+** 的具体优化路径，**按投入产出比排序**。

### 6.1 优化一：加长 max_len（最易实现，预期 +1~2%）

**问题**：当前 `max_len=128`，部分长推文被截断，丢失关键尾部信息。  
**做法**：修改训练、评估、推理三处代码。

```bash
# 训练时
python -m src.train --max_len 256 ...

# 评估时
python -m src.evaluate --max_len 256 ...

# inference.py 中也要改：将默认 128 改为 256
```

**修改位置**：
- `src/train.py` 中 `parser.add_argument("--max_len", type=int, default=256)`
- `src/evaluate.py` 中 `parser.add_argument("--max_len", type=int, default=256)`
- `inference.py` 中两处 `max_length=128` 改为 `max_length=256`
- `src/explainer.py` 中 `max_length=128` 改为 `max_length=256`

---

### 6.2 优化二：类别加权（易实现，预期 +1~2%）

**问题**：rumor 类假阴性较多（模型倾向于判非谣言）。  
**做法**：给 rumor 类更高损失权重。

**修改位置**：`src/train.py`

```python
# 在 train() 函数中，定义 loss 时加入权重
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train(args):
    # ... 加载数据后，计算类别权重 ...
    labels = pd.read_csv(args.train_csv)["label"].values
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 替换原来的 loss 计算
    # outputs = model(input_ids, attention_mask, labels=labels)
    # loss = outputs["loss"]
    # 改为手动计算加权 loss：
    outputs = model(input_ids, attention_mask)
    logits = outputs["logits"]
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(logits, labels)
```

---

### 6.3 优化三：换更强模型（效果最明显，预期 +2~4%，需 GPU）

**做法**：将 `roberta-base` 替换为 `roberta-large` 或 `deberta-v3-base`。

```bash
# 训练时指定模型名称
python -m src.train --model_name roberta-large --batch_size 16 ...
```

**注意**：
- `roberta-large` 参数量是 base 的 3 倍（355M vs 125M），推理更慢，需要 **显存 > 8G**
- 如果显存不足，batch_size 需降到 8 或 4，并开启 `fp16`

**进阶**：`deberta-v3-base` 在 NLP 分类任务上通常优于 roberta-base，且参数量相同：
```bash
python -m src.train --model_name microsoft/deberta-v3-base ...
```

---

### 6.4 优化四：集成学习（稳健提升，预期 +1~3%，但训练成本高）

**做法**：训练 3 个不同随机种子的模型，用投票决定最终标签。

```bash
# 训练 3 个模型
python -m src.train --seed 42 --output_dir checkpoints/seed42
python -m src.train --seed 2024 --output_dir checkpoints/seed2024
python -m src.train --seed 1234 --output_dir checkpoints/seed1234
```

然后在 `inference.py` 中加入投票逻辑（取 3 个模型预测的平均概率）。

---

### 6.5 优化五：数据增强（针对少数类，预期 +1~2%）

**做法**：对 `train.csv` 中 label=1（rumor）的样本做同义词替换、回译。

```bash
pip install nlpaug
```

在 `src/data_loader.py` 中加入简单的同义词替换增强（EDA 方法）：

```python
import random

def eda_synonym_replacement(text, n=2):
    """随机替换 n 个词为同义词。"""
    words = text.split()
    # 使用简单的同义词词典或 WordNet（需 nltk）
    # 这里仅作示例，实际可引入 nlpaug
    return text
```

---

### 6.6 优化优先级总结

| 优先级 | 优化项 | 预期提升 | 实现难度 | 硬件要求 |
|--------|--------|----------|----------|----------|
| 🔴 高 | 加长 max_len 到 256 | +1~2% | ⭐ 极易 | 无 |
| 🔴 高 | 类别加权（加权 CrossEntropy） | +1~2% | ⭐ 极易 | 无 |
| 🟡 中 | 换 deberta-v3-base | +2~3% | ⭐⭐ 容易 | 推荐 GPU |
| 🟡 中 | 换 roberta-large | +2~4% | ⭐⭐⭐ 中等 | **必须 GPU + 显存 > 8G** |
| 🟢 低 | 集成学习（3 模型投票） | +1~3% | ⭐⭐⭐⭐ 较复杂 | 训练成本 ×3 |
| 🟢 低 | 数据增强 | +1~2% | ⭐⭐⭐ 中等 | 无 |

**建议执行顺序**：先完成「加长 max_len + 类别加权」（无需 GPU，30 分钟可完成），若效果仍不满意再换 `deberta-v3-base`。

---

## 七、已完成工作汇总

| 阶段 | 内容 | 状态 |
|------|------|------|
| 方案设计 | 确定复合流水线（RoBERTa + Captum + CLAW LLM） | ✅ 完成 |
| 分工文档 | 编写 `分工.md`，明确四人职责与接口契约 | ✅ 完成 |
| 仓库文档 | `README.md`、`CLAUDE.md`、`.gitignore` | ✅ 完成 |
| 代码框架 | 9 个源码文件 + `requirements.txt` + `inference.py` + `test_sample.py` | ✅ 完成 |
| Baseline 训练 | `roberta-base` 训练 2 epochs，val_acc=**85.29%** | ✅ 完成 |
| 评估与错误分析 | 混淆矩阵、错误样例、假阴性分析 | ✅ 完成 |
| 可解释性模块 | `explainer.py` + `llm_client.py` 框架 | ✅ 完成 |
| 端到端推理 | `inference.py` 支持单条/批量推理 | ✅ 完成 |
| 模型优化 | 加长 max_len=256、GPU 训练、类别加权，val_acc **87.78%** | ✅ 完成 |
| LLM 解释生成 | 对接 CLAW API（minimax），Prompt 含正反证据 + 置信度 + event | ✅ 完成 |
| 安全改造 | 移除硬编码 API Key，统一使用 `.env` + `python-dotenv`，未配置时自动模板降级 | ✅ 完成 |
| 解释质量抽查 | 人工检查 30 条解释样例 | ⏳ 待执行 |
| 报告撰写 | `report.pdf`（≤2000 字） | ⏳ 待执行 |

---

## 八、待办事项

- [x] 运行训练脚本，记录 baseline 正确率（85.29%）
- [x] 完成错误分析，识别假阴性为主
- [x] 执行优化：加长 max_len=256 + 类别加权 + GPU 训练，val_acc **87.78%**
- [x] 实现 LLM 解释生成：对接 CLAW API（minimax），Prompt 含正反证据 + 置信度 + event
- [x] 随机抽取 10 条样本测试，端到端准确率 **90%**，LLM 解释正反两面论述效果良好
- [x] API Key 安全改造：移除硬编码 fallback，引入 `.env` 支持，验证 LLM 解释正常生成
- [ ] 人工抽查 30 条解释样例，验证可解释性质量
- [ ] 撰写 report.pdf
- [ ] 最终检查 README.md 完整性并提交
