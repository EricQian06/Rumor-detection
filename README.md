# Explainable Rumor Detection

基于深度学习的社交媒体谣言检测与可解释分析。本项目为 SJTU《人工智能导论》2026 大作业。

---

## 仓库导航

| 文件 | 说明 |
|------|------|
| [CLAUDE.md](./CLAUDE.md) | 面向 Claude Code 的仓库工作指南 |
| [分工.md](./分工.md) | 组内四人技术方案、模块接口与交付标准 |
| `作业要求.txt` | 课程组发布的原始作业要求 |
| `作业模板.txt` | 课程组发布的报告模板 |

---

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

> 推荐 Python 3.10。若使用 CUDA，请按官方说明安装对应版本的 PyTorch。

### 模型权重

（待补充：下载链接、放置路径、文件大小说明）

### 单条推理

```bash
python inference.py --text "Here is a sample tweet text."
```

输出示例：
```json
{
  "label": 1,
  "explanation": "This tweet is classified as rumor because it contains suspicious keywords such as 'breaking', 'unverified'."
}
```

### 批量推理

```bash
python inference.py --input data/val.csv --output results/predictions.csv
```

### CLAW API 配置（LLM 解释生成）

> 本组使用 Claude Code 作为编程辅助前端，但**模型推理与 LLM 解释均通过 Python 代码调用 CLAW API**，而非直接通过 Claude Code 对话窗口请求解释。

#### 1. 环境变量设置

运行任何涉及 LLM 的脚本前，先设置环境变量：

```bash
export CLAW_API_KEY="your_api_key_here"
export CLAW_BASE_URL="https://claw.sjtu.edu.cn/v1"
```

Windows PowerShell:
```powershell
$env:CLAW_API_KEY="your_api_key_here"
$env:CLAW_BASE_URL="https://claw.sjtu.edu.cn/v1"
```

#### 2. 程序内接入方式

由 `src/llm_client.py` 统一封装所有 CLAW API 调用，其他模块禁止直接构造请求：

```python
import os
from src.llm_client import CLAWClient

client = CLAWClient(
    api_key=os.getenv("CLAW_API_KEY"),
    base_url=os.getenv("CLAW_BASE_URL", "https://claw.sjtu.edu.cn/v1")
)

explanation = client.generate_explanation(
    text=tweet_text,
    label=predicted_label,
    evidence=[{"token": "breaking", "score": 0.85}, ...]
)
```

关键约定：
- **API Key 必须走环境变量**，禁止硬编码在源码或配置文件中。
- **调用端点**：请参照 [SJTU CLAW API 文档](https://claw.sjtu.edu.cn/guide/sjtu-api/) 确认当前支持的 chat/completions 端点。
- **降级策略**：若 API 超时、限流或返回异常，`CLAWClient` 自动返回模板解释，确保流水线不中断。详见 `src/llm_client.py` 实现。
- **批量限速**：批量生成解释时，建议在 `llm_client.py` 内加入 `time.sleep(0.5)` 避免触发限流。

---

## 目录结构

```
.
├── data/                   # 数据集
│   ├── train.csv
│   └── val.csv
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
# 示例：同学B开发分类模型
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
| 报告叙述清楚（30分） | `report.pdf`（数据洞察、实验图表素材） | D（整合）、A（素材） |
| 代码可运行、部署说明清楚（25分） | `README.md`、`requirements.txt`、`inference.py` | D |
| val.csv 分类准确率（15分） | `train.py`、`evaluate.py`、最佳模型 | B |
| 检测依据可解释性（15分） | `explainer.py`、`llm_client.py` | C |
| 小组分工协作（15分） | 全员 commit 记录 + `分工.md` | 全员（A 负责 Baseline 与数据流水线） |

---

## 参考与致谢

- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Captum: https://captum.ai/
- SJTU CLAW API: https://claw.sjtu.edu.cn/guide/sjtu-api/
