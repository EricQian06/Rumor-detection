"""
explainer.py — 基于 Captum 的可解释性归因模块
职责：钱

使用 Integrated Gradients 对分类结果进行归因，提取对预测影响最大的 token。
"""

import os
import json
import torch
import numpy as np
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz


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

    def get_token_attributions(self, text: str, target_label: int) -> dict:
        """返回单条文本的 token 级有符号归因，用于可视化。"""
        self.model.eval()
        device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs["logits"], dim=1)
            pred_label = int(torch.argmax(probs, dim=1).item())
            pred_prob = float(probs[0, pred_label].item())
            target_prob = float(probs[0, target_label].item())

        embeddings = self.embeddings(input_ids)
        embeddings.requires_grad_(True)

        pad_id = self.tokenizer.pad_token_id
        baseline_ids = torch.full_like(input_ids, pad_id)
        baseline_embeds = self.embeddings(baseline_ids)

        attributions, delta = self.ig.attribute(
            embeddings,
            baselines=baseline_embeds,
            target=target_label,
            n_steps=50,
            return_convergence_delta=True,
        )

        token_attributions = attributions.squeeze(0).sum(dim=-1).detach().cpu().numpy()
        token_ids = input_ids.squeeze(0).detach().cpu().numpy()
        mask = attention_mask.squeeze(0).detach().cpu().numpy()
        special_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }

        tokens = []
        scores = []
        for idx, token_id in enumerate(token_ids):
            if mask[idx] == 0 or token_id in special_tokens:
                continue
            token = self.tokenizer.decode([int(token_id)], skip_special_tokens=True)
            if not token.strip():
                continue
            tokens.append(token)
            scores.append(float(token_attributions[idx]))

        scores = np.array(scores, dtype=float)
        denom = np.linalg.norm(scores, ord=2)
        if denom > 0:
            scores = scores / denom

        return {
            "tokens": tokens,
            "scores": scores.tolist(),
            "pred_label": pred_label,
            "pred_prob": pred_prob,
            "target_label": int(target_label),
            "target_prob": target_prob,
            "delta": float(delta.detach().cpu().item()),
        }

    def visualize_text_attribution(
        self,
        text: str,
        target_label: int | None = None,
        output_dir: str = "results/explanation_examples",
        prefix: str = "example",
    ) -> dict:
        """使用 Captum visualization 保存单条文本的红绿高亮归因图。"""
        if target_label is None:
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=256,
                padding="max_length",
                truncation=True,
            )
            with torch.no_grad():
                outputs = self.model(
                    inputs["input_ids"].to(device),
                    inputs["attention_mask"].to(device),
                )
                target_label = int(torch.argmax(outputs["logits"], dim=1).item())

        attribution = self.get_token_attributions(text, target_label)
        label_name = "rumor" if attribution["pred_label"] == 1 else "non-rumor"
        target_name = "rumor" if attribution["target_label"] == 1 else "non-rumor"

        record = viz.VisualizationDataRecord(
            word_attributions=np.array(attribution["scores"]),
            pred_prob=attribution["pred_prob"],
            pred_class=label_name,
            true_class=target_name,
            attr_class=target_name,
            attr_score=sum(attribution["scores"]),
            raw_input_ids=attribution["tokens"],
            convergence_score=attribution["delta"],
        )

        os.makedirs(output_dir, exist_ok=True)
        html = viz.visualize_text([record])
        html_path = os.path.join(output_dir, f"{prefix}_attribution.html")
        json_path = os.path.join(output_dir, f"{prefix}_attribution.json")
        html_str = html.data if hasattr(html, "data") else str(html)

        # Captum 默认配色即正向绿色、负向红色；这里保存 HTML 供浏览器查看。
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_str)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"text": text, **attribution}, f, indent=2, ensure_ascii=False)

        return {"html_path": html_path, "json_path": json_path, **attribution}

    def format_evidence_for_prompt(self, evidence: list[dict]) -> str:
        """将 evidence 列表格式化为逗号分隔的字符串。"""
        return ", ".join([f"'{e['token']}'" for e in evidence])
