# Explainable Rumor Detection

基于 RoBERTa 的社交媒体谣言检测与可解释分析系统。本项目为上海交通大学《人工智能导论》课程大作业，目标是对英文推文进行二分类，并给出可读的判断依据。

- **输入**：英文推文文本，可选 `event` 事件编号
- **输出**：`label`、`confidence`、`explanation`、`evidence`
- **标签含义**：`0 = non-rumor`，`1 = rumor`
- **当前主模型**：`roberta-base`
- **验证集准确率**：`87.78%`
- **预训练模型地址**：[charchar2333/Rumor-detection](https://huggingface.co/charchar2333/Rumor-detection)

---

## 1. 项目工程说明

本项目采用“分类模型 + 可解释归因 + LLM 解释生成”的复合流水线：

```text
Input tweet
   ↓
Text cleaning / tokenization
   ↓
RoBERTa classifier
   ↓
Predicted label + confidence
   ↓
Integrated Gradients evidence extraction
   ↓
CLAW API / template fallback explanation
   ↓
Final output
```

### 核心模块

| 文件 / 目录 | 说明 |
|---|---|
| `train.csv` | 训练集，包含 `id,text,label,event` 四列 |
| `val.csv` | 验证集 / 测试集，格式同训练集 |
| `src/data_loader.py` | 数据清洗、`Dataset`、`DataLoader` |
| `src/model.py` | `RumorClassifier` 模型定义与模型加载函数 |
| `src/train.py` | 训练脚本，支持类别加权、早停、梯度累积 |
| `src/evaluate.py` | 模型评估脚本，输出 Accuracy、F1、混淆矩阵、错误样本 |
| `src/explainer.py` | Captum Integrated Gradients 归因，提取支持 / 反方证据 token |
| `src/llm_client.py` | SJTU CLAW API 封装，生成自然语言解释，失败时模板降级 |
| `inference.py` | 端到端推理入口，支持单条文本和 CSV 批量推理 |
| `test_sample.py` | 从 `val.csv` 随机抽取 10 条样本进行端到端测试 |
| `src/adversarial_analysis.py` | 字符级输入扰动鲁棒性测试脚本 |
| `Project.md` | 项目全过程记录，包括实验、调参、安全分析与数据清洗 |
| `requirements.txt` | Python 依赖 |

---

## 2. 环境部署与安装

### 2.1 Python 环境

推荐使用：

- Python 3.10+
- Windows / Linux / macOS 均可
- 有 NVIDIA GPU 时推荐使用 CUDA 版 PyTorch

建议创建虚拟环境：

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 主要包含：

- `torch`
- `transformers`
- `pandas`
- `scikit-learn`
- `captum`
- `matplotlib`
- `seaborn`
- `tqdm`
- `python-dotenv`

### 2.3 GPU 加速，可选

如果本机有 NVIDIA GPU，建议安装匹配 CUDA 版本的 PyTorch。例如 CUDA 12.4：

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

验证 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

输出 `True` 表示 GPU 可用。

---

## 3. 模型准备

项目支持两种使用方式：

1. **直接使用 HuggingFace 上的预训练模型**，推荐；
2. **本地重新训练模型**。

### 3.1 方式一：直接使用 HuggingFace 模型，推荐

无需本地训练，直接指定 HuggingFace 模型路径：

```bash
python inference.py --model_dir charchar2333/Rumor-detection --text "Breaking: unconfirmed reports say the bridge collapsed." --event 0
```

批量推理：

```bash
python inference.py --model_dir charchar2333/Rumor-detection --input val.csv --output results/predictions.csv
```

快速随机测试：

```bash
python test_sample.py --model_dir charchar2333/Rumor-detection
```

如果 HuggingFace 下载较慢，可以设置镜像：

```bash
# Windows CMD
set HF_ENDPOINT=https://hf-mirror.com

# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
```

### 3.2 方式二：下载模型到本地 checkpoints/

```bash
huggingface-cli download charchar2333/Rumor-detection --local-dir checkpoints
```

下载完成后，可直接使用默认本地路径：

```bash
python inference.py --text "Breaking: unconfirmed reports say the bridge collapsed." --event 0
```

默认模型目录为：

```text
checkpoints/
```

其中应包含：

```text
best_model.pt
model_config.json
tokenizer.json
vocab.json
merges.txt
...
```

---

## 4. 如何运行模型

### 4.1 单条文本推理

```bash
python inference.py --text "Here is a sample tweet text." --event 0
```

使用 HuggingFace 模型：

```bash
python inference.py --model_dir charchar2333/Rumor-detection --text "Here is a sample tweet text." --event 0
```

示例输出：

```json
{
  "label": 1,
  "confidence": 0.9905,
  "explanation": "The tweet is classified as rumor because ...",
  "evidence": {
    "predicted": [
      {"token": " reports", "score": 0.2915}
    ],
    "opposite": [
      {"token": "confirmed", "score": 0.1835}
    ]
  },
  "event": 0
}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `label` | 预测标签，`0=non-rumor`，`1=rumor` |
| `confidence` | 模型对预测标签的置信度 |
| `explanation` | 自然语言解释 |
| `evidence.predicted` | 支持预测类别的关键 token |
| `evidence.opposite` | 支持相反类别的关键 token |
| `event` | 输入事件编号 |

### 4.2 批量 CSV 推理

输入 CSV 至少应包含：

```text
text
```

如果包含 `id` 和 `event`，会一并保留或使用。

运行：

```bash
python inference.py --input val.csv --output results/predictions.csv
```

使用 HuggingFace 模型：

```bash
python inference.py --model_dir charchar2333/Rumor-detection --input val.csv --output results/predictions.csv
```

输出 CSV 包含：

```text
id,text,label,confidence,explanation,event
```

### 4.3 不使用 LLM，仅使用模板解释

如果没有配置 CLAW API Key，程序会自动降级为模板解释。也可以显式跳过 LLM：

```bash
python inference.py --text "Some tweet text." --no_llm
```

批量模式同样支持：

```bash
python inference.py --input val.csv --output results/predictions.csv --no_llm
```

---

## 5. CLAW API 配置，可选但推荐

课程要求如使用大语言模型，需使用 SJTU 提供的 CLAW API。本项目所有 LLM 调用都集中封装在：

```text
src/llm_client.py
```

API Key 不允许硬编码，必须通过环境变量或 `.env` 文件提供。

### 5.1 使用 .env 文件

在项目根目录新建 `.env`：

```env
CLAW_API_KEY=your_api_key_here
CLAW_BASE_URL=https://models.sjtu.edu.cn/api
```

`.env` 已加入 `.gitignore`，不会提交到 GitHub。

### 5.2 使用临时环境变量

Linux / macOS：

```bash
export CLAW_API_KEY="your_api_key_here"
export CLAW_BASE_URL="https://models.sjtu.edu.cn/api"
```

Windows PowerShell：

```powershell
$env:CLAW_API_KEY="your_api_key_here"
$env:CLAW_BASE_URL="https://models.sjtu.edu.cn/api"
```

如果未配置 API Key，系统仍可正常运行，只是解释会使用模板 fallback。

---

## 6. 训练模型

如果需要重新训练模型，运行：

```bash
python -m src.train \
  --train_csv train.csv \
  --val_csv val.csv \
  --model_name roberta-base \
  --epochs 5 \
  --batch_size 32 \
  --lr 2e-5 \
  --max_len 256 \
  --output_dir checkpoints
```

GPU 训练可增加 `num_workers`：

```bash
python -m src.train --train_csv train.csv --val_csv val.csv --epochs 5 --batch_size 32 --lr 2e-5 --max_len 256 --num_workers 4
```

显存不足时可减小 batch size：

```bash
python -m src.train --train_csv train.csv --val_csv val.csv --batch_size 16
```

训练脚本会自动：

- 加载 tokenizer 与 RoBERTa backbone；
- 使用类别加权 CrossEntropyLoss；
- 使用 AdamW + linear warmup；
- 进行梯度裁剪；
- 根据验证集准确率保存最佳模型；
- 保存 tokenizer、`best_model.pt`、`training_history.json`、`model_config.json`。

训练输出目录默认为：

```text
checkpoints/
```

---

## 7. 模型评估

```bash
python -m src.evaluate --model_dir checkpoints --val_csv val.csv --max_len 256
```

输出：

- Accuracy
- F1 Score
- Classification Report
- Confusion Matrix
- 错误样本

结果文件：

```text
results/evaluation_results.json
results/error_cases.json
results/confusion_matrix.png
```

当前主模型在 `val.csv` 上的结果：

| 指标 | 值 |
|---|---:|
| Accuracy | 87.78% |
| F1 Score, rumor | 0.8665 |
| Recall, rumor | 90.86% |

---

## 8. 快速随机测试

从 `val.csv` 随机抽取 10 条样本，进行完整端到端测试：

```bash
python test_sample.py
```

使用 HuggingFace 模型：

```bash
python test_sample.py --model_dir charchar2333/Rumor-detection
```

该脚本会输出：

- 原文
- 真实标签
- 预测标签
- 置信度
- LLM / 模板解释
- 支持预测的证据 token
- 支持对立类别的证据 token

---

## 9. 对抗扰动鲁棒性测试

项目包含字符级输入扰动测试，用于评估模型面对简单规避攻击时的鲁棒性。

### 9.1 Basic 扰动

规则：

```text
o/O → 0
l/L → 1
```

运行：

```bash
python -m src.adversarial_analysis --model_dir checkpoints --val_csv val.csv --preset basic
```

### 9.2 Extended 扰动

规则：

```text
o/O → 0
l/L → 1
i/I → 1
e/E → 3
a/A → @
s/S → $
```

运行：

```bash
python -m src.adversarial_analysis --model_dir checkpoints --val_csv val.csv --preset extended
```

输出文件：

```text
results/adversarial_basic_metrics.json
results/adversarial_basic_examples.csv
results/adversarial_extended_metrics.json
results/adversarial_extended_examples.csv
```

已测结果显示：

| 扰动类型 | Clean Acc | Perturbed Acc | Accuracy Drop | Attack Success Rate |
|---|---:|---:|---:|---:|
| basic | 87.78% | 77.56% | 10.22% | 17.05% |
| extended | 87.78% | 56.61% | 31.17% | 44.89% |

说明模型对字符级规避扰动较敏感，可作为后续输入规范化和对抗训练的改进方向。

---

## 10. 数据投毒 / 错标样本清洗

项目还进行了训练集标签一致性检查，用于发现潜在数据投毒或错标样本。

方法：

1. 使用字符级 TF-IDF 表示推文；
2. 计算样本间余弦相似度；
3. 筛选 `similarity >= 0.70` 且 `label` 不同的近重复样本；
4. 输出人工核查表；
5. 对语义重复但标签冲突的样本进行人工清洗。

人工核查文件：

```text
results/train_label_conflict_pairs_similarity_ge_0_70.csv
```

清洗后的 `train.csv` 已作为当前训练集使用。

---

## 11. 数据格式

`train.csv` 和 `val.csv` 均包含四列：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | int / str | 推文 ID |
| `text` | str | 推文文本 |
| `label` | int | 标签，`0=non-rumor`，`1=rumor` |
| `event` | int | 事件类别编号 |

批量推理时，输入 CSV 至少需要 `text` 列；如果有 `id` 和 `event`，会保留在输出中。

---

## 12. 常见问题

### Q1: 没有 `CLAW_API_KEY` 能运行吗？

可以。模型分类和证据提取不依赖 CLAW API。未配置 Key 时，解释部分会自动使用模板 fallback。

### Q2: 没有 GPU 能运行吗？

可以。推理和评估可以在 CPU 上运行，只是速度较慢。训练建议使用 GPU。

### Q3: HuggingFace 下载慢怎么办？

设置镜像：

```bash
set HF_ENDPOINT=https://hf-mirror.com          # Windows CMD
export HF_ENDPOINT=https://hf-mirror.com       # Linux / macOS
```

### Q4: 重新训练后模型加载维度不匹配怎么办？

训练脚本会保存：

```text
checkpoints/model_config.json
```

评估和推理脚本会自动读取 backbone 名称。若手动指定模型，请确保 `--model_name` 与训练时一致。

### Q5: 如何快速验证项目是否可运行？

推荐顺序：

```bash
pip install -r requirements.txt
python test_sample.py --model_dir charchar2333/Rumor-detection
```

如果能输出 10 条样本的预测和解释，说明环境和模型加载正常。

---

## 13. 参考

- HuggingFace Transformers: https://huggingface.co/docs/transformers
- PyTorch: https://pytorch.org/
- Captum Integrated Gradients: https://captum.ai/
- SJTU CLAW API: https://claw.sjtu.edu.cn/guide/sjtu-api/
- 预训练模型: https://huggingface.co/charchar2333/Rumor-detection
