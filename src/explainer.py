"""
explainer.py — 基于 Captum 的可解释性归因模块
职责：钱

使用 Integrated Gradients 对分类结果进行归因，提取对预测影响最大的 token。
"""

import torch
import numpy as np
from captum.attr import IntegratedGradients


class IGExplainer:
    """
    基于 Integrated Gradients 的解释器。
    输入：文本、预测标签、模型、tokenizer
    输出：归因后的关键 token 列表
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.ig = IntegratedGradients(self._forward_func)

        # 获取嵌入层（不同模型名称不同）
        if hasattr(model.backbone, "embeddings"):
            self.embeddings = model.backbone.embeddings.word_embeddings
        elif hasattr(model.backbone, "roberta"):
            self.embeddings = model.backbone.roberta.embeddings.word_embeddings
        elif hasattr(model.backbone, "bert"):
            self.embeddings = model.backbone.bert.embeddings.word_embeddings
        else:
            # 通用 fallback
            self.embeddings = model.backbone.get_input_embeddings()

    def _forward_func(self, embeddings):
        """
        Captum 需要的前向函数，输入 embedding，输出 logits。
        """
        # 构造 attention_mask（全1，因为长度固定）
        attention_mask = torch.ones(
            embeddings.shape[:2], dtype=torch.long, device=embeddings.device
        )
        outputs = self.model.backbone(inputs_embeds=embeddings, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.model.classifier(self.model.dropout(pooled))
        return logits

    @torch.no_grad()
    def get_evidence(self, text: str, label: int, top_k: int = 5) -> list[dict]:
        """
        对输入文本进行归因，返回对预测标签贡献最大的 top_k 个 token。

        Returns:
            list[dict]: [{"token": str, "score": float}, ...]
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # 编码文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # 获取 embeddings
        embeddings = self.embeddings(input_ids)
        embeddings.requires_grad_(True)

        # 基准：全 [PAD] token 的 embedding
        pad_id = self.tokenizer.pad_token_id
        baseline_ids = torch.full_like(input_ids, pad_id)
        baseline_embeds = self.embeddings(baseline_ids)

        # 计算 Integrated Gradients
        attributions, delta = self.ig.attribute(
            embeddings,
            baselines=baseline_embeds,
            target=label,  # 对预测类别进行归因
            n_steps=50,
            return_convergence_delta=True,
        )

        # 对每个 token 的归因值求和（绝对值）
        token_importance = attributions.squeeze(0).sum(dim=-1).detach().cpu().numpy()
        token_ids = input_ids.squeeze(0).cpu().numpy()
        mask = attention_mask.squeeze(0).cpu().numpy()

        # 过滤 [PAD]、特殊符号，按重要性排序
        special_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }

        evidence = []
        for idx in range(len(token_ids)):
            if mask[idx] == 0 or token_ids[idx] in special_tokens:
                continue
            token = self.tokenizer.decode([token_ids[idx]], skip_special_tokens=True)
            score = abs(float(token_importance[idx]))
            if token.strip() and score > 1e-6:
                evidence.append({"token": token, "score": score})

        # 按分数排序，取 top_k
        evidence.sort(key=lambda x: x["score"], reverse=True)
        top_evidence = evidence[:top_k]

        # 归一化分数（百分比）
        total = sum(e["score"] for e in top_evidence) or 1.0
        for e in top_evidence:
            e["score"] = round(e["score"] / total, 4)

        return top_evidence

    @torch.no_grad()
    def get_evidence_both_sides(self, text: str, predicted_label: int, top_k: int = 5) -> dict:
        """
        同时提取对预测标签和对立标签的证据。
        Returns:
            dict: {
                "predicted": [{"token": str, "score": float}, ...],
                "opposite": [{"token": str, "score": float}, ...],
            }
        """
        opposite_label = 1 - predicted_label
        return {
            "predicted": self.get_evidence(text, predicted_label, top_k),
            "opposite": self.get_evidence(text, opposite_label, top_k),
        }

    def format_evidence_for_prompt(self, evidence: list[dict]) -> str:
        """将 evidence 列表格式化为逗号分隔的字符串。"""
        return ", ".join([f"'{e['token']}'" for e in evidence])
