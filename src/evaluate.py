"""
evaluate.py — 模型评估脚本
职责：刘
用法：python -m src.evaluate --model_dir checkpoints --val_csv val.csv
"""

import os
import argparse
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.model import RumorClassifier, build_model_and_tokenizer
from src.data_loader import get_dataloaders


def evaluate_model(model, dataloader, device):
    """在验证集上进行完整评估。"""
    model.eval()
    all_preds = []
    all_labels = []
    all_texts = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            texts = batch["raw_text"]

            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_texts.extend(texts)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    report = classification_report(all_labels, all_preds, target_names=["non-rumor", "rumor"], digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    # 找出错误样本
    errors = []
    for text, pred, label in zip(all_texts, all_preds, all_labels):
        if pred != label:
            errors.append({
                "text": text,
                "predicted": int(pred),
                "actual": int(label),
            })

    return {
        "accuracy": acc,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm,
        "errors": errors,
    }


def plot_confusion_matrix(cm, output_path="results/confusion_matrix.png"):
    """绘制并保存混淆矩阵热力图。"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["non-rumor", "rumor"],
                yticklabels=["non-rumor", "rumor"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate rumor detection model")
    parser.add_argument("--model_dir", default="checkpoints", help="Directory containing saved model")
    parser.add_argument("--val_csv", default="val.csv", help="Path to validation CSV")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--model_name", type=str, default=None, help="Override backbone name (auto-detected from model_config.json if omitted)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 自动识别训练时使用的 backbone 名称
    model_name = args.model_name
    if model_name is None:
        config_path = os.path.join(args.model_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_name = json.load(f).get("model_name", "roberta-base")
            print(f"Auto-detected model_name: {model_name}")
        else:
            model_name = "roberta-base"
            print(f"Warning: model_config.json not found, defaulting to {model_name}")

    # 加载 tokenizer 和模型
    _, tokenizer = build_model_and_tokenizer(model_name)
    model = RumorClassifier(model_name)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt"), map_location=device))
    model.to(device)

    _, val_loader = get_dataloaders(
        args.val_csv, args.val_csv, tokenizer,
        batch_size=args.batch_size, max_len=args.max_len
    )

    results = evaluate_model(model, val_loader, device)

    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print("\nClassification Report:")
    print(results["report"])

    # 保存结果
    os.makedirs("results", exist_ok=True)
    plot_confusion_matrix(results["confusion_matrix"])

    with open("results/evaluation_results.json", "w") as f:
        json.dump({
            "accuracy": results["accuracy"],
            "f1": results["f1"],
            "report": results["report"],
        }, f, indent=2)

    with open("results/error_cases.json", "w") as f:
        json.dump(results["errors"][:20], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
