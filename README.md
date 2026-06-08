# Explainable Rumor Detection

基于深度学习的社交媒体谣言检测与可解释分析。本项目为 SJTU《人工智能导论》2026 大作业。

## 项目概述

本项目构建了一个**可解释的谣言检测系统**，输入一条英文推文，输出：
1. **分类结果**：rumor（1）或 non-rumor（0），val.csv 上准确率 **87.78%**
2. **自然语言解释**：由 LLM 生成的判断依据，支持**正反两面论述**（既说明支持预测的证据，也分析可能的反方证据）

技术栈：RoBERTa 分类器（`roberta-base`）→ Captum Integrated Gradients 归因 → SJTU CLAW API（`minimax` 模型）生成解释。

**关键特性**：
- 预训练模型托管于 [HuggingFace](https://huggingface.co/charchar2333/Rumor-detection)，**可跳过训练直接使用**
- GPU 训练（CUDA 12.4），5 epochs 约 40 分钟
- 加长 max_len=256 + 类别加权，rumor recall 达 **90.86%**
- LLM 解释含置信度、事件类别、正反证据引用
- 支持单条推理、批量推理、快速随机测试
- API 失效时自动降级为模板解释，确保流水线不中断

---

## 仓库导航

| 文件 | 说明 |
|------|------|
| [CLAUDE.md](./CLAUDE.md) | 面向 Claude Code 的仓库工作指南 |
| [分工.md](./分工.md) | 组内四人技术方案、模块接口与交付标准 |
| [Project.md](./Project.md) | **项目全过程记录**：实验结果、错误分析、优化方案、迁移指南 |

> **📋 完整项目记录**：实验结果、错误分析、后续优化方案、迁移到新电脑的操作步骤，请参阅 **[Project.md](./Project.md)**。

---

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

> 推荐 Python 3.10。若使用 CUDA，请按官方说明安装对应版本的 PyTorch。

### 快速使用预训练模型（推荐，无需训练）

预训练模型已上传至 [HuggingFace Hub](https://huggingface.co/charchar2333/Rumor-detection)：

```python
from transformers import AutoModel, AutoTokenizer

model_name = "charchar2333/Rumor-detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

> 下载较慢时，可设置镜像：`HF_ENDPOINT=https://hf-mirror.com`

### 训练模型（可选）

```bash
# GPU 训练（推荐，约 8 分钟/epoch）
python -m src.train --train_csv train.csv --val_csv val.csv --epochs 5 --batch_size 32 --lr 2e-5 --num_workers 4

# CPU 训练（约 10-30 分钟/epoch）
python -m src.train --train_csv train.csv --val_csv val.csv --epochs 5 --batch_size 32 --lr 2e-5 --num_workers 0

# 最佳模型将保存到 checkpoints/best_model.pt
```

### 评估模型

```bash
python -m src.evaluate --model_dir checkpoints --val_csv val.csv
```

输出：准确率、F1、混淆矩阵图（`results/confusion_matrix.png`）、错误样例（`results/error_cases.json`）。

**当前 Baseline 结果**（`roberta-base`，GPU + max_len=256 + 类别加权，详见 [Project.md](./Project.md)）：

| 指标 | 值 |
|------|-----|
| Accuracy | **87.78%** |
| F1 Score (rumor) | **0.8665** |
| Recall (rumor) | **0.9086** |

> 相较 v1.0（85.29%），v2.0 通过加长 max_len 和类别加权，**rumor recall 提升 8.57%**，假阴性显著减少。详细分析请参阅 **[Project.md](./Project.md)**。

### 单条推理

```bash
python inference.py --text "Here is a sample tweet text." --event 0
```

输出示例：
```json
{
  "label": 1,
  "confidence": 0.9905,
  "explanation": "## Classification Analysis\n\n**Predicted Label: Rumor (Label 1) with 99.05% confidence**\n\nThis tweet is classified as a rumor primarily due to the explicit language stating 'Sources unverified' and 'Unconfirmed reports,' which directly signal the information lacks credible verification. However, there is reasonable counter-evidence: the tweet does use proper journalistic attribution and names a specific location ('Shanghai'), which are hallmarks of legitimate news reporting. The distinction is that while the structure resembles credible news, the tweet's own admission that sources are 'unverified' makes the claim fundamentally unverified.",
  "evidence": {
    "predicted": [{"token": " reports", "score": 0.2915}, ...],
    "opposite": [{"token": "BRE", "score": 0.1835}, ...]
  },
  "event": 0
}
```

解释特点：
- **正反两面论述**：既说明支持预测标签的证据，也分析可能被误判为对立标签的原因
- **引用证据词**：解释中明确引用 Captum IG 归因提取的关键 token
- **置信度与事件类别**：包含模型 confidence 和 event 上下文

跳过 LLM（使用模板降级）：
```bash
python inference.py --text "..." --no_llm
```

### 批量推理

```bash
python inference.py --input val.csv --output results/predictions.csv
```

### 快速测试（随机 10 条样本）

从 `val.csv` 随机抽取 10 条（5 条 non-rumor + 5 条 rumor）进行端到端测试，查看分类结果和 LLM 解释：

```bash
python test_sample.py
```

预期输出（示例）：
```
测试完成！准确率: 9/10 (90%)
```

### GPU 加速（强烈推荐）

若电脑有 NVIDIA GPU，安装 CUDA 版 PyTorch 以大幅加速训练：

```bash
# 卸载 CPU 版 torch，安装 CUDA 11.8 版
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

训练时脚本会自动检测并使用 GPU。若显存不足（< 8G），请减小 `--batch_size`（如 16 或 8）。

### 常见问题

| 问题 | 解决 |
|------|------|
| `ModuleNotFoundError` | 确认虚拟环境已激活，`pip install -r requirements.txt` 已执行 |
| `CUDA out of memory` | 减小 `--batch_size`（16 / 8 / 4），或冻结更多层 |
| HuggingFace 下载慢/失败 | `set HF_ENDPOINT=https://hf-mirror.com`（Windows CMD）后重试 |
| CPU 训练极慢 | 正常，5 epochs 约 1-2 小时。建议换有 GPU 的机器 |

---

### CLAW API 配置（LLM 解释生成）

> 本组使用 Claude Code 作为编程辅助前端，但**模型推理与 LLM 解释均通过 Python 代码调用 CLAW API**，而非直接通过 Claude Code 对话窗口请求解释。

#### 1. 推荐：`.env` 文件（安全、方便）

在项目根目录创建 `.env` 文件：

```env
CLAW_API_KEY=your_api_key_here
CLAW_BASE_URL=https://models.sjtu.edu.cn/api
```

`.env` 已被加入 `.gitignore`，**不会提交到 GitHub**，适合本地开发。

#### 2. 备选：系统环境变量

Linux/macOS:

```bash
export CLAW_API_KEY="your_api_key_here"
export CLAW_BASE_URL="https://models.sjtu.edu.cn/api"
```

Windows PowerShell:
```powershell
$env:CLAW_API_KEY="your_api_key_here"
$env:CLAW_BASE_URL="https://models.sjtu.edu.cn/api"
```

> 默认使用 `minimax` 模型，通过标准 OpenAI 兼容的 `/chat/completions` 端点调用。

#### 3. 程序内接入方式

由 `src/llm_client.py` 统一封装所有 CLAW API 调用，其他模块禁止直接构造请求：

```python
import os
from dotenv import load_dotenv
from src.llm_client import CLAWClient

# 自动加载 .env 文件（如存在）
load_dotenv()

client = CLAWClient(
    api_key=os.getenv("CLAW_API_KEY"),
    base_url=os.getenv("CLAW_BASE_URL", "https://models.sjtu.edu.cn/api"),
    model="minimax",
)

# 生成带正反两面论述的解释
explanation = client.generate_explanation(
    text=tweet_text,
    label=predicted_label,
    confidence=predicted_confidence,
    event=event_id,
    evidence={
        "predicted": [{"token": "breaking", "score": 0.85}, ...],
        "opposite": [{"token": "confirmed", "score": 0.42}, ...],
    }
)
```

关键约定：

- **API Key 必须走环境变量或 `.env` 文件**，禁止硬编码在源码或配置文件中。
- **调用端点**：请参照 [SJTU CLAW API 文档](https://claw.sjtu.edu.cn/guide/sjtu-api/) 确认当前支持的 chat/completions 端点。
- **降级策略**：若 API Key 未配置、超时、限流或返回异常，`CLAWClient` 自动返回模板解释，确保流水线不中断。详见 `src/llm_client.py` 实现。
- **批量限速**：批量生成解释时，建议在 `llm_client.py` 内加入 `time.sleep(0.5)` 避免触发限流。

---

## 目录结构

```
.
├── train.csv               # 训练集
├── val.csv                 # 验证集
├── checkpoints/            # 训练好的模型权重
├── results/                # 实验结果、图表、预测输出
├── src/                    # 核心源码
│   ├── data_loader.py      # 数据加载与清洗
│   ├── model.py            # 模型定义
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   ├── explainer.py        # 可解释性归因
│   └── llm_client.py       # CLAW API 封装
├── inference.py            # 端到端推理入口
├── test_sample.py          # 随机 10 条样本快速测试
├── requirements.txt        # Python 依赖
├── README.md               # 本文件
├── CLAUDE.md               # Claude Code 工作指南
├── 分工.md                 # 项目分工与模块细节
└── report.pdf              # 最终报告
```

---

## Git 使用规范

### 分支策略

- `master`（或 `main`）为主分支，存放可运行、稳定的代码。
- 每人开发时从 `master` 检出功能分支，完成后通过 Pull Request 或合并提交合并回 `master`。

```bash
# 示例：刘开发分类模型
git checkout -b feat/classifier
# ... 开发、提交 ...
git checkout master
git pull origin master
git merge feat/classifier
git push origin master
```

### Commit 规范

提交信息建议格式：`<type>: <subject>`

| type | 含义 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `docs` | 文档更新（README、报告） |
| `refactor` | 代码重构 |
| `test` | 增加测试或 Baseline |

示例：
```bash
git commit -m "feat: add RoBERTa classifier with early stopping"
git commit -m "fix: llm retry logic when API timeout"
git commit -m "docs: update README with inference examples"
```

### 提交前检查

- 不提交数据集、模型权重、密钥、个人缓存。
- 确保 `.gitignore` 已生效：`git status` 中不应出现 `__pycache__`、`.claude/`、`checkpoints/` 等。

---

## CLAUDE.md 使用说明

[CLAUDE.md](./CLAUDE.md) 是为 [Claude Code](https://claude.ai/code) 编写的仓库级提示词，包含：

- 项目背景与目标
- 数据集格式说明
- 作业约束（如必须使用 SJTU CLAW API）
- 当前仓库状态与待办事项

**何时使用**：
- 当你使用 Claude Code 辅助编写或调试本仓库代码时，Claude 会自动读取 `CLAUDE.md`，从而快速理解项目背景，无需重复说明。
- 若项目结构或约束发生变化（如新增依赖、更换模型），请同步更新 `CLAUDE.md`，保证后续 Claude 会话的上下文准确。

---

## 评分对应

| 评分项 | 对应内容 | 负责人 |
|--------|----------|--------|
| 报告叙述清楚（30分） | `report.pdf`（数据洞察、实验图表素材） | 贺（整合）、荣（素材） |
| 代码可运行、部署说明清楚（25分） | `README.md`、`requirements.txt`、`inference.py` | 贺 |
| val.csv 分类准确率（15分） | `train.py`、`evaluate.py`、最佳模型 | 刘 |
| 检测依据可解释性（15分） | `explainer.py`、`llm_client.py` | 钱 |
| 小组分工协作（15分） | 全员 commit 记录 + `分工.md` | 全员（荣 负责 Baseline 与数据流水线） |

---

## 参考与致谢

- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Captum: https://captum.ai/
- SJTU CLAW API: https://claw.sjtu.edu.cn/guide/sjtu-api/
