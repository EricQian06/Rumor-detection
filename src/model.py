"""
model.py — 谣言分类模型定义
职责：刘
"""

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
    """便捷函数：同时构建模型和分词器。"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RumorClassifier(model_name, num_labels)
    return model, tokenizer
