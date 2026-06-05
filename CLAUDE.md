# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a course project for SJTU's "Introduction to Artificial Intelligence" (人工智能导论). The task is to build an **explainable rumor detection** model for social media text. The model must classify tweets as rumor (1) or non-rumor (0) and provide a textual explanation for its decision.

## Repository Contents

- `train.csv` — Training dataset (~401KB)
- `val.csv` — Validation/test dataset (~56KB)
- `作业要求.txt` — Detailed assignment requirements (Chinese)
- `作业模板.txt` — Project report template (Chinese)

## Dataset Schema

Both CSV files contain four columns:

1. `id` — Numeric tweet identifier
2. `text` — Tweet content
3. `label` — Binary classification target (`0` = non-rumor, `1` = rumor)
4. `event` — Event category/grouping identifier

## Assignment Constraints

The following constraints come from `作业要求.txt` and must be respected:

- **Classification output**: Binary (0 or 1). Accuracy on `val.csv` contributes 15% of the grade.
- **Explanation output**: A human-readable text string justifying the classification decision. Explanation quality contributes 15% of the grade.
- **LLM usage**: If a large language model is used for any part of the pipeline (e.g., generating explanations), it **must** use the SJTU-provided CLAW API. Do not hard-code API keys for external services. See https://claw.sjtu.edu.cn/guide/sjtu-api/.
- **Allowed techniques**: Deep learning, LLMs, RAG, or composite architectures (e.g., a DL model for classification + an LLM for explanation).
- **Submission requirements**: The final GitHub repo must include:
  - `readme.md` — Project description, deployment instructions, and how to run the model.
  - `report.pdf` — Final report (max ~2000 words) following the template in `作业模板.txt`.
  - All code and supporting files.

## Common Commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the classifier (saves best checkpoint to `checkpoints/`):
```bash
python -m src.train --train_csv train.csv --val_csv val.csv --epochs 5 --batch_size 32 --lr 2e-5
```

Colab 训练（推荐）：
```bash
# 在 Colab 中运行以下命令
%cd /content
!git clone https://github.com/EricQian06/Rumor-detection.git
%cd Rumor-detection
!pip install -q -r requirements.txt
%run colab_train_safe.py
```

Evaluate on `val.csv` (outputs metrics, confusion matrix plot, and error cases):
```bash
python -m src.evaluate --model_dir checkpoints --val_csv val.csv
```

Run single-text inference with explanation:
```bash
# 方式1：在项目根目录创建 .env 文件（推荐，已加入 .gitignore）
# CLAW_API_KEY=your_key
# CLAW_BASE_URL=https://models.sjtu.edu.cn/api
python inference.py --text "some tweet text" --event 0

# 方式2：临时环境变量
export CLAW_API_KEY="your_key"
export CLAW_BASE_URL="https://models.sjtu.edu.cn/api"
python inference.py --text "some tweet text" --event 0
```

Run batch inference:
```bash
python inference.py --input val.csv --output results/predictions.csv
```

Skip LLM explanations and use template fallback:
```bash
python inference.py --text "some tweet text" --no_llm
```

## High-Level Architecture

The system is a composite pipeline with three stages:

1. **Classification** (`src/model.py`, `src/train.py`, `src/evaluate.py`)
   - `RumorClassifier` wraps a pretrained Transformer (`roberta-base` by default) and appends `Dropout(0.1)` + `Linear(hidden_size, 2)` on the `[CLS]` token embedding.
   - `train.py` uses `AdamW` with linear warmup (10% of total steps) and gradient clipping (`max_norm=1.0`). Best model is saved by validation accuracy.
   - `evaluate.py` computes accuracy, F1, and a confusion matrix; it also extracts up to 20 misclassified examples into `results/error_cases.json`.

2. **Attribution** (`src/explainer.py`)
   - `IGExplainer` uses Captum's `IntegratedGradients` on the **embedding layer** to attribute the predicted label back to input tokens.
   - It handles different backbone architectures (BERT, RoBERTa, DeBERTa) via `hasattr` fallbacks when locating the embedding layer.
   - Output is a list of `{"token": str, "score": float}` sorted by absolute attribution, filtered to remove special tokens and padding, then normalized to percentages.

3. **Attribution (Both Sides)** (`src/explainer.py`)
   - `IGExplainer` uses Captum's `IntegratedGradients` on the **embedding layer** to attribute the predicted label back to input tokens.
   - It handles different backbone architectures (BERT, RoBERTa, DeBERTa) via `hasattr` fallbacks when locating the embedding layer.
   - `get_evidence_both_sides(text, predicted_label, top_k=5)` returns evidence for **both** the predicted label and the opposite label, enabling balanced explanations.
   - Output is a dict `{"predicted": [...], "opposite": [...]}` where each list contains `{"token": str, "score": float}` sorted by absolute attribution, filtered to remove special tokens and padding, then normalized to percentages.

4. **Explanation Generation** (`src/llm_client.py`)
   - `CLAWClient` is the **only** module permitted to make HTTP calls to the LLM API. All other code must import and use it.
   - It wraps the SJTU CLAW API (`https://models.sjtu.edu.cn/api/chat/completions`, `minimax` model) with retry (3 attempts, exponential backoff), timeout (30s), and a **template fallback** if all retries fail.
   - Prompt temperature is fixed at `0.3` for consistency; max tokens at `512`.
   - The prompt includes: tweet text, predicted label, confidence score, event category, supporting evidence tokens, and counter-evidence tokens — producing balanced, two-sided explanations.

5. **Orchestration** (`inference.py`)
   - Loads the trained `RumorClassifier`, instantiates `IGExplainer`, and `CLAWClient`.
   - For each input, it runs classification → both-sides attribution → LLM explanation, returning `{"label", "confidence", "explanation", "evidence", "event"}`.
   - Batch mode iterates with `tqdm` and writes a CSV containing the original columns plus `label`, `confidence`, and `explanation`.

## Cross-Module Interface Contracts

These contracts are defined in `分工.md`. If you change any of them, update `分工.md` and notify the module owner.

- `src.data_loader.get_dataloaders(train_csv, val_csv, tokenizer, batch_size, max_len)` → `(train_loader, val_loader)`. Each batch contains `input_ids`, `attention_mask`, `label`, `raw_text`, and `event`.
- `src.explainer.IGExplainer.get_evidence_both_sides(text, predicted_label, top_k=5)` → `dict` with keys `"predicted"` and `"opposite"`, each mapping to `list[dict]` with keys `token` and `score`.
- `src.llm_client.CLAWClient.generate_explanation(text, label, confidence, event, evidence)` → `str`.

## Development Guidelines

- **Dependencies**: After adding any new package import, update `requirements.txt` with the package name and a minimum version. Do not pin exact patch versions unless necessary.
- **Documentation sync**: If you change a public function signature, a script's CLI arguments, or the directory structure, update both `README.md` and `分工.md` (or notify the owner to do so).
- **Secrets**: The CLAW API key must be read from environment variables (`CLAW_API_KEY`).
- **Code ownership**: Respect module boundaries defined in `分工.md`. If you need to change another member's interface, discuss it first and update the interface contract in `分工.md`.
- **Results & checkpoints**: Save experiment outputs (plots, metrics, model weights) under `results/` and `checkpoints/`. These directories are gitignored; share large files via external storage links in `README.md` rather than committing them.

## Model Selection & Experiment History

This section records backbone experiments so future contributors do not repeat dead ends.

| Model | Params | Best val_acc | Training time/epoch (T4) | Conclusion |
|-------|--------|--------------|--------------------------|------------|
| `roberta-base` | 125M | **87.78%** | ~5 min | ⭐ **Default.** Best cost-benefit for this small (~4K) dataset. |
| `microsoft/deberta-v3-base` | 86M | 82.29% | ~6 min | ❌ Poor fit. Disentangled attention overfits on small data; val_loss rises from epoch 2. |
| `roberta-large` | 355M | 88.78% | ~7 min | ⚠️ Marginal gain (+1%) but severe overfit (train_loss 0.02 vs val_loss 0.66). Not worth the doubled training time. |

**Decision**: Stay on `roberta-base`. All improvements (EarlyStopping, gradient accumulation, auto-detect `model_name`) are kept in code so other backbones can still be tested via CLI flags.

## Fixed Bugs & New Features

| Change | Files | Details |
|--------|-------|---------|
| Hard-coded `roberta-base` in `evaluate.py` / `inference.py` / `test_sample.py` | `src/evaluate.py`, `inference.py`, `test_sample.py` | `train.py` now saves `model_config.json`; loaders auto-read the backbone name. No more dimension mismatch when switching models. |
| DeBERTa `float16` vs `float32` dtype mismatch | `src/model.py` | Added `self.to(torch.float32)` after classifier init. |
| Missing `weight_decay` CLI arg | `src/train.py` | Added `--weight_decay` (default 0.01). |
| Scheduler step count ignored gradient accumulation | `src/train.py` | `total_steps` now divides by `accumulation_steps`. |
| EarlyStopping + gradient accumulation | `src/train.py`, `colab_train_safe.py` | `--patience` (default 3) stops when val_acc stalls. `--accumulation_steps` simulates larger batches on memory-constrained GPUs. |
