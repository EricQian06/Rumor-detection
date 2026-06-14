"""
adversarial_analysis.py — 输入扰动鲁棒性评估脚本
职责：安全鲁棒性测试
用法：python -m src.adversarial_analysis --model_dir checkpoints --val_csv val.csv
"""

import os
import argparse
import json
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.model import build_model_and_tokenizer
from src.data_loader import clean_text


BASIC_SUBSTITUTIONS = {
    "o": "0",
    "O": "0",
    "l": "1",
    "L": "1",
}

EXTENDED_SUBSTITUTIONS = {
    **BASIC_SUBSTITUTIONS,
    "i": "1",
    "I": "1",
    "e": "3",
    "E": "3",
    "a": "@",
    "A": "@",
    "s": "$",
    "S": "$",
}


PRESETS = {
    "basic": BASIC_SUBSTITUTIONS,
    "extended": EXTENDED_SUBSTITUTIONS,
}


def load_model(model_dir: str, device, model_name: str | None = None):
    """加载训练好的模型和 tokenizer。"""
    if model_name is None:
        config_path = os.path.join(model_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_name = json.load(f).get("model_name", "roberta-base")
            print(f"Auto-detected model_name: {model_name}")
        else:
            model_name = "roberta-base"
            print(f"Warning: model_config.json not found, defaulting to {model_name}")

    model, tokenizer = build_model_and_tokenizer(model_name)
    model_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run training first.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer


def perturb_text(text: str, substitutions: dict[str, str]) -> tuple[str, int]:
    """按字符替换规则生成扰动文本，并返回替换次数。"""
    chars = []
    changed = 0
    for char in text:
        if char in substitutions:
            chars.append(substitutions[char])
            changed += 1
        else:
            chars.append(char)
    return "".join(chars), changed


def predict_texts(model, tokenizer, texts, labels, device, batch_size: int, max_len: int):
    """批量预测文本，返回预测标签、预测置信度和真实标签置信度。"""
    preds = []
    pred_confidences = []
    true_confidences = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[start:start + batch_size]
            batch_labels = labels[start:start + batch_size]
            inputs = tokenizer(
                batch_texts,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs["logits"], dim=1).cpu()
            batch_preds = torch.argmax(probs, dim=1)

            preds.extend(batch_preds.tolist())
            pred_confidences.extend(probs.gather(1, batch_preds.unsqueeze(1)).squeeze(1).tolist())
            true_confidences.extend(probs[range(len(batch_labels)), batch_labels].tolist())

    return preds, pred_confidences, true_confidences


def summarize_results(labels, clean_preds, clean_true_conf, adv_preds, adv_true_conf):
    """汇总 clean 与扰动后的鲁棒性指标。"""
    clean_acc = accuracy_score(labels, clean_preds)
    adv_acc = accuracy_score(labels, adv_preds)
    clean_f1 = f1_score(labels, clean_preds, average="binary")
    adv_f1 = f1_score(labels, adv_preds, average="binary")

    clean_correct = [pred == label for pred, label in zip(clean_preds, labels)]
    adv_wrong = [pred != label for pred, label in zip(adv_preds, labels)]
    flips = [clean_pred != adv_pred for clean_pred, adv_pred in zip(clean_preds, adv_preds)]
    attackable_count = sum(clean_correct)
    attack_success_count = sum(c and w for c, w in zip(clean_correct, adv_wrong))

    return {
        "num_samples": len(labels),
        "clean_accuracy": clean_acc,
        "perturbed_accuracy": adv_acc,
        "accuracy_drop": clean_acc - adv_acc,
        "clean_f1": clean_f1,
        "perturbed_f1": adv_f1,
        "f1_drop": clean_f1 - adv_f1,
        "flip_rate": sum(flips) / len(labels),
        "attack_success_rate": attack_success_count / attackable_count if attackable_count else 0.0,
        "attack_success_count": attack_success_count,
        "clean_correct_count": attackable_count,
        "avg_true_label_confidence_clean": sum(clean_true_conf) / len(clean_true_conf),
        "avg_true_label_confidence_perturbed": sum(adv_true_conf) / len(adv_true_conf),
        "avg_true_label_confidence_drop": (
            sum(c - a for c, a in zip(clean_true_conf, adv_true_conf)) / len(labels)
        ),
        "clean_report": classification_report(labels, clean_preds, target_names=["non-rumor", "rumor"], digits=4),
        "perturbed_report": classification_report(labels, adv_preds, target_names=["non-rumor", "rumor"], digits=4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate robustness under character-level adversarial perturbations")
    parser.add_argument("--model_dir", default="checkpoints", help="Directory containing saved model")
    parser.add_argument("--val_csv", default="val.csv", help="Path to validation CSV")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--model_name", type=str, default=None, help="Override backbone name")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="basic",
                        help="basic: o/O->0 and l/L->1; extended: add i/e/a/s leetspeak")
    parser.add_argument("--output_dir", default="results", help="Directory to save adversarial metrics and examples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_model(args.model_dir, device, args.model_name)

    df = pd.read_csv(args.val_csv)
    labels = df["label"].astype(int).tolist()
    clean_texts = df["text"].astype(str).apply(clean_text).tolist()

    substitutions = PRESETS[args.preset]
    perturbed = [perturb_text(text, substitutions) for text in clean_texts]
    perturbed_texts = [item[0] for item in perturbed]
    changed_counts = [item[1] for item in perturbed]

    print("\nEvaluating clean validation texts...")
    clean_preds, clean_pred_conf, clean_true_conf = predict_texts(
        model, tokenizer, clean_texts, labels, device, args.batch_size, args.max_len
    )

    print("\nEvaluating perturbed validation texts...")
    adv_preds, adv_pred_conf, adv_true_conf = predict_texts(
        model, tokenizer, perturbed_texts, labels, device, args.batch_size, args.max_len
    )

    metrics = summarize_results(labels, clean_preds, clean_true_conf, adv_preds, adv_true_conf)
    metrics["attack_preset"] = args.preset
    metrics["substitutions"] = substitutions
    metrics["changed_sample_count"] = sum(count > 0 for count in changed_counts)
    metrics["avg_changed_chars"] = sum(changed_counts) / len(changed_counts)

    print("\n=== Character Perturbation Robustness ===")
    print(f"Attack preset: {args.preset} ({substitutions})")
    print(f"Changed samples: {metrics['changed_sample_count']}/{metrics['num_samples']}")
    print(f"Clean Accuracy:      {metrics['clean_accuracy']:.4f}")
    print(f"Perturbed Accuracy:  {metrics['perturbed_accuracy']:.4f}")
    print(f"Accuracy Drop:       {metrics['accuracy_drop']:.4f}")
    print(f"Flip Rate:           {metrics['flip_rate']:.4f}")
    print(f"Attack Success Rate: {metrics['attack_success_rate']:.4f}")
    print(f"True Conf Drop:      {metrics['avg_true_label_confidence_drop']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"adversarial_{args.preset}_metrics.json")
    examples_path = os.path.join(args.output_dir, f"adversarial_{args.preset}_examples.csv")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    out_df = pd.DataFrame({
        "id": df.get("id", pd.Series(range(len(df)))),
        "label": labels,
        "event": df.get("event", pd.Series([0] * len(df))),
        "text": clean_texts,
        "perturbed_text": perturbed_texts,
        "changed_chars": changed_counts,
        "clean_pred": clean_preds,
        "perturbed_pred": adv_preds,
        "clean_confidence": clean_pred_conf,
        "perturbed_confidence": adv_pred_conf,
        "clean_true_confidence": clean_true_conf,
        "perturbed_true_confidence": adv_true_conf,
        "prediction_flipped": [c != a for c, a in zip(clean_preds, adv_preds)],
        "attack_succeeded": [c == y and a != y for c, a, y in zip(clean_preds, adv_preds, labels)],
    })
    out_df.to_csv(examples_path, index=False)

    print(f"\nMetrics saved to {metrics_path}")
    print(f"Examples saved to {examples_path}")


if __name__ == "__main__":
    main()
