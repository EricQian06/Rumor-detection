"""
data_loader.py — 数据加载与预处理
职责：荣
"""

import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


def clean_text(text: str) -> str:
    """清洗推文文本。"""
    # 去除 URL
    text = re.sub(r"http\S+", "", text)
    # 去除多余空白
    text = re.sub(r"\s+", " ", text)
    # 去除前后空白
    return text.strip()


class RumorDataset(Dataset):
    """PyTorch Dataset，加载 CSV 并返回模型需要的张量。"""

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 128,
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 清洗文本
        self.df["text_clean"] = self.df["text"].astype(str).apply(clean_text)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = row["text_clean"]
        label = int(row["label"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        event = int(row["event"])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "raw_text": text,  # 保留原始文本供解释器使用
            "event": event,
        }


def get_dataloaders(
    train_csv: str,
    val_csv: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_len: int = 128,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """创建训练与验证 DataLoader。"""
    train_dataset = RumorDataset(train_csv, tokenizer, max_len)
    val_dataset = RumorDataset(val_csv, tokenizer, max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def get_label_distribution(csv_path: str) -> pd.Series:
    """EDA 用：查看标签分布。"""
    df = pd.read_csv(csv_path)
    return df["label"].value_counts()
