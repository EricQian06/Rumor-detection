# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a course project for SJTU's "Introduction to Artificial Intelligence" (人工智能导论). The task is to build an **explainable rumor detection** model for social media text. The model must classify tweets as rumor (1) or non-rumor (0) and provide a textual explanation for its decision.

## Repository Contents

- `train.csv` — Training dataset (~401KB)
- `val.csv` — Validation/test dataset (~56KB)
- `作业要求.txt` — Detailed assignment requirements (Chinese)
- `作业模板.txt` — Project report template (Chinese)

No build system, dependency manifest, or test suite exists yet.

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

## Architecture Notes

The codebase is currently empty except for data and documentation. There is no existing architecture to extend. When implementing:

- The core interface is: **input a single text → output a label and an explanation**.
- A composite architecture is explicitly encouraged by the assignment (e.g., a neural classifier paired with a separate explanation generator).
- Keep runtime reasonable; excessive inference time may be penalized.

## Development Guidelines

When working in this repository, follow these conventions so the project stays consistent and reproducible:

- **Dependencies**: After adding any new package import, update `requirements.txt` with the package name and a minimum version. Do not pin exact patch versions unless necessary.
- **Documentation sync**: If you change a public function signature, a script's CLI arguments, or the directory structure, update both `README.md` and `分工.md` (or notify the owner to do so).
- **Secrets**: Never hard-code API keys, tokens, or passwords. The CLAW API key must be read from environment variables (`CLAW_API_KEY`).
- **Git hygiene**: Commit via Git, not file uploads. Write meaningful commit messages (`feat:`, `fix:`, `docs:`). Before committing, run `git status` and make sure `__pycache__`, `.claude/`, `checkpoints/`, and `data/` are ignored.
- **Code ownership**: Respect module boundaries defined in `分工.md`. If you need to change another member's interface, discuss it first and update the interface contract in `分工.md`.
- **Results & checkpoints**: Save experiment outputs (plots, metrics, model weights) under `results/` and `checkpoints/`. These directories are gitignored; share large files via external storage links in `README.md` rather than committing them.
