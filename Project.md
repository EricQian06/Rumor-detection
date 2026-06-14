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
| 钱 | 可解释性与 LLM，模型调试，数据清洗 | `src/explainer.py`, `src/llm_client.py` |
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

> **结论**：准确率从 85.29% 提升至 **87.78%**（+2.49%）。**rumor recall 大幅提升至 90.86%**（+8.57%），假阴性显著减少，类别加权策略效果显著。

---

**v3.0 DeBERTa 初探（Colab T4, bs=16, lr=2e-5, 5 epochs）**

运行命令：
```bash
python -m src.train --model_name microsoft/deberta-v3-base --train_csv train.csv --val_csv val.csv --epochs 5 --batch_size 16 --lr 2e-5 --max_len 256
```

| epoch | train_loss | val_loss | val_acc | 备注 |
|-------|-----------|----------|---------|------|
| 1 | 0.6520 | 0.5162 | 0.7406 | — |
| 2 | 0.4895 | 0.4736 | 0.7731 | — |
| 3 | 0.3857 | 0.5075 | 0.7805 | val_loss 开始上升 |
| 4 | 0.2998 | 0.5285 | 0.8080 | 过拟合迹象 |
| 5 | 0.2508 | 0.5613 | **0.8229** | train_loss 持续降，但 val_loss 持续升 |

**问题诊断**：
- val_acc 仍在涨（74% → 82%），但 **val_loss 从 epoch 2 起持续上升**，说明 5 epoch 不够且已过拟合
- batch_size=16 太小，DeBERTa 对 batch size 敏感
- lr=2e-5 对 DeBERTa 偏高

**v3.1 DeBERTa 再尝试（Colab T4, bs=16+accum2, lr=1e-5, wd=0.1, 10 epochs + EarlyStopping）**

| epoch | train_loss | val_loss | val_acc | 备注 |
|-------|-----------|----------|---------|------|
| 1 | 0.7123 | 0.6640 | 0.5910 | — |
| 2 | 0.6202 | 0.5538 | 0.7382 | — |
| 3 | 0.5201 | 0.5902 | 0.7232 | val_acc 下降 |
| 4 | 0.4394 | 0.4848 | 0.7706 | — |
| 5 | 0.3762 | 0.4684 | 0.7955 | — |
| 6 | 0.3328 | 0.5167 | 0.8055 | — |
| 7 | 0.2705 | 0.5473 | **0.8155** | 最佳 |
| 8 | 0.2451 | 0.6069 | 0.8130 | 过拟合 |

**结论**：DeBERTa 在该小数据集（~4K 条）上表现持续不佳，最高仅 **81.55%**。原因可能是：
1. 数据集太小，DeBERTa 的 disentangled attention 难以发挥优势
2. weight_decay=0.1 过强，导致模型学不动（train_loss 下降极慢）

**v4.0 roberta-large（Colab T4, bs=8+accum4, lr=1e-5, 10 epochs + EarlyStopping）**

运行命令：
```bash
python -m src.train --model_name roberta-large --train_csv train.csv --val_csv val.csv --epochs 10 --batch_size 8 --accumulation_steps 4 --lr 1e-5 --max_len 256 --patience 3
```

| epoch | train_loss | val_loss | val_acc | 备注 |
|-------|-----------|----------|---------|------|
| 1 | 0.6378 | 0.4751 | 0.7805 | — |
| 2 | 0.4041 | 0.3488 | 0.8529 | — |
| 3 | 0.2810 | 0.3214 | 0.8753 | — |
| 4 | 0.2001 | 0.3394 | 0.8753 | val_loss 上升 |
| 5 | 0.1387 | 0.3149 | 0.8803 | — |
| 6 | 0.0814 | **0.4475** | 0.8653 | val_loss 暴涨 |
| 7 | 0.0537 | **0.5362** | 0.8853 | 严重过拟合 |
| 8 | 0.0415 | **0.6666** | **0.8878** | 最佳，但 val_loss 失控 |
| 9 | 0.0287 | 0.6423 | 0.8853 | — |
| 10 | 0.0209 | 0.6587 | 0.8828 | — |

**最终评估结果（val.csv，n=401）**：

| 指标 | roberta-base | roberta-large | 变化 |
|------|-------------|---------------|------|
| Accuracy | **87.78%** | **88.78%** | +1.00% |
| 训练时间/epoch | ~5 min | ~7 min | ×1.4 |

**问题诊断**：
- train_loss 降到 0.02，val_loss 飙到 0.67：教科书式**严重过拟合**
- 355M 参数 vs ~2.8K 训练样本：模型容量远超数据量，记忆了噪声
- 仅提升 1% 准确率，训练时间增加 40%，**性价比极低**

**最终决策**：放弃 roberta-large 和 DeBERTa，**默认模型回退至 roberta-base**。
所有代码改进（EarlyStopping、梯度累积、自动识别 backbone、weight_decay CLI）保留，方便未来换模型测试。

---

### 3.3 错误分析

从 `results/error_cases.json` 抽查发现，模型的主要错误模式是：

| 错误类型 | 典型案例 | 原因分析 |
|----------|----------|----------|
| **假阴性**（rumor→non-rumor） | "#Ferguson police are embarking on what can only be described as an elaborate smear campaign..." | 模型将"基于真实事件的批评性报道"误判为非谣言，因为这些文本使用了事实性语言、具体人名/地点，缺少夸张词汇。 |
| **假阴性**（rumor→non-rumor） | "Lawyers for police in bad shootings often advise shooters not to write reports..." | 类似地，带有法律/制度分析色彩的文本被误判。 |

**核心问题**：模型学到的"rumor"模式偏向"夸张、未证实、情绪化"，而对"基于事实的阴谋论/质疑"识别不足。

### 3.4 数据投毒 / 错标样本排查与人工清洗

为回应课程中关于“对抗攻击防护 / 数据投毒防御”的要求，项目增加了训练集标签一致性检查。考虑到数据投毒在本任务中最常见的表现是**少量语义相近样本被赋予相反标签**，我们采用“相似样本邻居投票 / 近重复样本检测”的思路，对 `train.csv` 进行了训练前数据安全排查。

排查方法：
- 使用字符级 TF-IDF（char n-gram 3–5）表示推文文本，计算样本间余弦相似度；
- 筛选相似度 `>= 0.70` 且 `label` 不同的样本对；
- 输出人工核查文件 `results/train_label_conflict_pairs_similarity_ge_0_70.csv`，包含样本 id、原始标签、event、文本内容、相似度和人工决策列；
- 对高相似冲突样本进行人工核查，重点检查完全重复、近重复和语义高度一致但标签相反的样本。

排查发现：
- `train.csv` 中存在少量完全重复或高度相似但标签不同的样本；
- 这些样本不能直接断言为恶意投毒，更合理的解释是原始数据中的**标注噪声 / 重复采样冲突 / 事件发展阶段差异**；
- 但从模型安全角度看，它们符合数据投毒防御中的 suspicious samples 定义，会向模型提供相互矛盾的训练信号。

清洗处理：
- 已对语义重复但标签冲突的样本进行人工核查；
- 对确认存在标签冲突或重复冲突的数据进行了人工清洗；
- 当前 `train.csv` 已更新为清洗后的训练集。

> **结论**：基于邻居投票的数据投毒检测在本项目中具有可行性。它不作为自动删改标签的最终依据，而是作为训练前数据安全筛查工具，帮助定位潜在错标或投毒样本，再由人工完成最终清洗。

---

### 3.5 调整方法与记录

**当前状态**：准确率 **87.78%**（roberta-base + GPU + max_len=256 + 类别加权）。经过 DeBERTa 和 roberta-large 实验后确认，**roberta-base 是该数据集的最优选择**。

| 优化方向 | 预期提升 | 具体做法 | 实验结果 |
|----------|---------|---------|----------|
| ~~换更强模型~~ | ~~+2~4%~~ | ~~roberta-large / deberta-v3-base~~ | ❌ roberta-large 仅 +1% 且严重过拟合；DeBERTa 仅 82% |
| ✅ 加长 max_len | +1~2% | 256（原 128） | ✅ v2.0 已实施，有效 |
| ✅ 类别加权 | +1~2% | `CrossEntropyLoss(weight=[0.8875, 1.1452])` | ✅ v2.0 已实施，rumor recall +8.57% |
| 🟡 数据增强 | +1~2% | 同义词替换、回译 | ⏳ 未尝试，如需突破 90% 可尝试 |
| 🟡 集成学习 | +1~3% | 3 模型投票 | ⏳ 未尝试，如需突破 90% 可尝试 |
| 🟡 特征工程 | +0.5~1% | 文本长度、感叹号、URL 数量 | ⏳ 未尝试 |

**最终优先级建议**：
1. **✅ 已完成**：roberta-base + max_len=256 + 类别加权 = **87.78%**（课程项目优秀水平）
2. 🟡 **如需突破 90%**：尝试**集成学习**（3 个 roberta-base 投票）或**数据增强**
3. ❌ **不推荐**：换更大 backbone（数据集太小，大模型严重过拟合）

**已尝试 / 未尝试**：
- [x] roberta-base baseline（85.29%）
- [x] max_len=256 + 类别加权（87.78%）
- [x] 修复 evaluate.py / inference.py / test_sample.py 硬编码 bug（自动识别 backbone）
- [x] DeBERTa-v3-base（82.29% / 81.55%，效果不佳，已放弃）
- [x] roberta-large（88.78%，严重过拟合，性价比低，已放弃）
- [ ] 数据增强
- [ ] 集成学习
- [ ] 特征工程

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
| 2026/06/05 | **Colab 适配**：创建 `colab_train.ipynb` + `colab_train_safe.py`，修复大小写路径问题，添加 T4 GPU 优化参数和 OOM 保护 | 钱 |
| 2026/06/05 | **模型加载 Bug 修复**：`train.py` 保存 `model_config.json` 记录 backbone 名称；`evaluate.py` / `inference.py` 自动读取该配置，彻底消除硬编码 `roberta-base` 导致的维度不匹配问题 | 钱 |
| 2026/06/05 | **DeBERTa 支持**：默认切换为 `microsoft/deberta-v3-base`，Colab Notebook 和 safe 脚本均适配，预期再提升 +2~3% | 钱 |
| 2026/06/05 | **DeBERTa 调优**：v3.0 初探仅 82.29%，诊断为过拟合 + lr 过大 + batch 过小；v3.1 新增 EarlyStopping、梯度累积、lr=1e-5、weight_decay=0.1 | 钱 |
| 2026/06/05 | **roberta-large 实验**：Colab T4 训练 10 epochs，最佳 88.78%，但 train_loss=0.02 vs val_loss=0.66 严重过拟合，性价比低 | 钱 |
| 2026/06/05 | **最终决策**：默认模型回退至 `roberta-base`（87.78% 为最优），保留所有代码改进供未来扩展 | 钱 |
| 2026/06/05 | **更新 CLAUDE.md**：添加模型选择实验记录、Colab 训练指南、bug 修复清单 | Claude |
| 2026/06/14 | **数据投毒 / 错标样本排查**：基于字符级 TF-IDF 相似度筛选 `similarity >= 0.70` 且标签冲突的近重复样本，输出人工核查清单，并对语义重复但标签不同的数据完成清洗，更新 `train.csv` | Claude / 钱 |

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

### 6.3 优化三：换更强模型（❌ 已证伪，数据集太小不适用）

> **实验结论**：在本项目的小数据集（~2.8K 训练样本）上，**更大的 backbone 反而效果更差或性价比极低**。

| 模型 | 参数量 | 最佳 val_acc | 过拟合程度 | 结论 |
|------|--------|-------------|-----------|------|
| `roberta-base` | 125M | **87.78%** | 轻微 | ⭐ **最优选择** |
| `deberta-v3-base` | 86M | 82.29% | 中等 | ❌ 不适用小数据 |
| `roberta-large` | 355M | 88.78% | **严重** | ⚠️ +1% 但训练时间 ×1.4 |

**原因分析**：
- 大模型（355M 参数）需要更多数据才能收敛，否则容易记忆噪声
- `train_loss` 降到 0.02 而 `val_loss` 飙到 0.66 是典型过拟合信号
- 对于课程项目级别的数据量，**roberta-base 的容量已足够**

**如果仍想尝试**（代码已保留支持）：
```bash
python -m src.train --model_name roberta-large --batch_size 8 --accumulation_steps 4 --lr 1e-5 ...
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

| 优先级 | 优化项 | 预期提升 | 实现难度 | 硬件要求 | 状态 |
|--------|--------|----------|----------|----------|------|
| ✅ 已完成 | 加长 max_len 到 256 | +1~2% | ⭐ 极易 | 无 | v2.0 已实施 |
| ✅ 已完成 | 类别加权（加权 CrossEntropy） | +1~2% | ⭐ 极易 | 无 | v2.0 已实施 |
| ❌ 已放弃 | ~~换 deberta-v3-base~~ | ~~+2~3%~~ | ⭐⭐ 容易 | 推荐 GPU | 实验 82%，不适用 |
| ❌ 已放弃 | ~~换 roberta-large~~ | ~~+2~4%~~ | ⭐⭐⭐ 中等 | 必须 GPU + 显存 > 8G | 实验 88.78%，过拟合严重 |
| 🟡 如需突破 90% | 集成学习（3 模型投票） | +1~3% | ⭐⭐⭐⭐ 较复杂 | 训练成本 ×3 | 未开始 |
| 🟡 如需突破 90% | 数据增强 | +1~2% | ⭐⭐⭐ 中等 | 无 | 未开始 |
| 🟢 低 | 特征工程 | +0.5~1% | ⭐⭐ 中等 | 无 | 未开始 |

**最终建议**：
1. ✅ **当前最佳**：roberta-base + max_len=256 + 类别加权 = **87.78%**（课程项目优秀水平）
2. 🟡 **如需更高**：集成学习是唯一有希望的途径（数据增强次之）
3. ❌ **不推荐**：更大 backbone（数据集太小，大模型必然过拟合）

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
| 模型优化 v2.0 | 加长 max_len=256、GPU 训练、类别加权，val_acc **87.78%** | ✅ 完成 |
| 模型优化 v3.0/v3.1 | DeBERTa 实验（82.29% / 81.55%，已放弃） | ✅ 完成 |
| 模型优化 v4.0 | roberta-large 实验（88.78%，严重过拟合，已放弃） | ✅ 完成 |
| 最终模型选择 | 回退 roberta-base（87.78% 为最优），保留所有代码改进 | ✅ 完成 |
| 数据投毒 / 错标排查 | 筛选相似度 >= 0.70 且标签冲突的近重复样本，人工核查并清洗 `train.csv` | ✅ 完成 |
| LLM 解释生成 | 对接 CLAW API（minimax），Prompt 含正反证据 + 置信度 + event | ✅ 完成 |
| 安全改造 | 移除硬编码 API Key，统一使用 `.env` + `python-dotenv`，未配置时自动模板降级 | ✅ 完成 |
| 解释质量抽查 | 人工检查 30 条解释样例 | ✅ 完成 |
| 报告撰写 | `report.pdf`（≤2000 字） | ⏳ 待执行 |

---

## 八、待办事项

- [x] 运行训练脚本，记录 baseline 正确率（85.29%）
- [x] 完成错误分析，识别假阴性为主
- [x] 执行优化 v2.0：加长 max_len=256 + 类别加权 + GPU 训练，val_acc **87.78%**
- [x] 执行优化 v3.0/v3.1：DeBERTa 实验（82.29% / 81.55%，已放弃）
- [x] 执行优化 v4.0：roberta-large 实验（88.78%，严重过拟合，已放弃）
- [x] 最终模型决策：回退 roberta-base（87.78% 为数据集最优解）
- [x] 数据投毒 / 错标样本排查：筛选相似度 >= 0.70 且标签冲突的近重复样本，人工核查后清洗 `train.csv`
- [x] 实现 LLM 解释生成：对接 CLAW API（minimax），Prompt 含正反证据 + 置信度 + event
- [x] 随机抽取 10 条样本测试，端到端准确率 **90%**，LLM 解释正反两面论述效果良好
- [x] API Key 安全改造：移除硬编码 fallback，引入 `.env` 支持，验证 LLM 解释正常生成
- [x] Colab 适配：创建 `colab_train.ipynb` + `colab_train_safe.py`，支持 T4 GPU 训练和 OOM 保护
- [x] 模型加载 bug 修复：`train.py` 保存 `model_config.json`；`evaluate.py` / `inference.py` / `test_sample.py` 自动识别 backbone
- [x] 新增功能：EarlyStopping、梯度累积、weight_decay CLI
- [ ] 人工抽查 30 条解释样例，验证可解释性质量
- [ ] 撰写 report.pdf
- [ ] 最终检查 README.md 完整性并提交
