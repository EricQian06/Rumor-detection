"""
model.py — 谣言分类模型定义
职责：刘
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class RumorClassifier(nn.Module):
    """
    基于预训练 Transformer 的谣言分类器。
    输入文本 → [CLS]向量 → Dropout → Linear → 2类logits
    """

    def __init__(self, model_name: str = "roberta-base", num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        # 某些模型（如 DeBERTa）的 safetensors 权重默认以 float16 加载，
        # 而 classifier 是 float32，会导致 matmul dtype 不匹配。统一为 float32。
        self.to(torch.float32)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] 位置的向量 (batch_size, hidden_size)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


def build_model_and_tokenizer(model_name: str = "roberta-base", num_labels: int = 2):
    """
    便捷函数：同时构建模型和分词器。

    支持两种加载方式：
    - 本地路径：如 "checkpoints"
    - HuggingFace：如 "charchar2333/Rumor-detection"，从缓存加载自定义权重

    """
    # 判断是否从 HuggingFace 加载
    is_hf = "/" in str(model_name) and (model_name.startswith("charchar") or model_name.startswith("hf-"))

    if is_hf:
        # 从 HuggingFace 加载
        cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")

        # 在 snapshots 目录下查找文件（HuggingFace 缓存结构）
        snapshots_dir = os.path.join(cache_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            # 获取最新的 snapshot 目录
            snapshot_folders = os.listdir(snapshots_dir)
            if snapshot_folders:
                snapshot_dir = os.path.join(snapshots_dir, snapshot_folders[0])
            else:
                snapshot_dir = cache_dir
        else:
            snapshot_dir = cache_dir

        # 查找 model_config.json 获取 backbone 名称
        model_config_path = os.path.join(snapshot_dir, "model_config.json")
        if os.path.exists(model_config_path):
            with open(model_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                base_model_name = config.get("model_name", "roberta-base")
        else:
            base_model_name = "roberta-base"

        # 加载 tokenizer 和 base model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = RumorClassifier(base_model_name, num_labels)

        # 从缓存加载权重
        import glob
        # 查找 .pt 或 .bin 文件（在 snapshot 目录）
        pt_files = glob.glob(os.path.join(snapshot_dir, "*.pt"))
        bin_files = glob.glob(os.path.join(snapshot_dir, "*.bin"))
        weight_files = pt_files + bin_files
        if weight_files:
            state_dict = torch.load(weight_files[0], map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Loaded weights from: {os.path.basename(weight_files[0])}")
    else:
        # 本地加载
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = RumorClassifier(model_name, num_labels)

    return model, tokenizer


def load_model_from_path(model_dir: str, model_name: str = None):
    """
    从本地 checkpoint 目录加载模型和分词器。

    Args:
        model_dir: 包含 best_model.pt 和 model_config.json 的目录
        model_name: 可选，强制指定 backbone 名称
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 自动识别 backbone 名称
    if model_name is None:
        config_path = os.path.join(model_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_name = json.load(f).get("model_name", "roberta-base")
        else:
            model_name = "roberta-base"

    model, tokenizer = build_model_and_tokenizer(model_name)
    model_path = os.path.join(model_dir, "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer, device
