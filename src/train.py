"""
train.py — 模型训练脚本
职责：刘
用法：python -m src.train --train_csv train.csv --val_csv val.csv --epochs 5
"""

import os
import argparse
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight

from src.model import build_model_and_tokenizer
from src.data_loader import get_dataloaders


def evaluate(model, dataloader, device):
    """在验证集上评估，返回平均 loss 和准确率。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 构建模型和 tokenizer
    model, tokenizer = build_model_and_tokenizer(args.model_name, num_labels=2)
    model.to(device)

    # 2. 数据加载
    train_loader, val_loader = get_dataloaders(
        args.train_csv,
        args.val_csv,
        tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers,
    )

    # 2.5 计算类别权重（针对 rumor 类假阴性过多的问题）
    labels = pd.read_csv(args.train_csv)["label"].values
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 4. 训练循环
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

            # 梯度累积：等效大 batch_size
            if args.accumulation_steps > 1:
                loss = loss / args.accumulation_steps

            loss.backward()

            # 每 accumulation_steps 步更新一次参数
            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                # 梯度裁剪，防止爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * (args.accumulation_steps if args.accumulation_steps > 1 else 1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 保存最佳模型 + Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            tokenizer.save_pretrained(args.output_dir)
            print(f"  → Best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  → val_acc did not improve ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    # 5. 保存训练历史
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # 保存模型配置，供 evaluate / inference 自动识别 backbone
    model_config = {"model_name": args.model_name}
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\nTraining finished. Best val_acc: {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train rumor detection model")
    parser.add_argument("--train_csv", default="train.csv", help="Path to training CSV")
    parser.add_argument("--val_csv", default="val.csv", help="Path to validation CSV")
    parser.add_argument("--model_name", default="roberta-base", help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--output_dir", default="checkpoints", help="Directory to save model")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps (simulate larger batch)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
