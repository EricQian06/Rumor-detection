"""
llm_client.py — SJTU CLAW API 封装
职责：钱

所有 CLAW API 调用必须经过此模块，禁止在其他文件中直接构造 HTTP 请求。
"""

import os
import time
import requests
from typing import Optional


class CLAWClient:
    """
    SJTU CLAW API 的 Python 客户端。
    支持重试、超时、异常降级。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "minimax",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.api_key = api_key or os.getenv("CLAW_API_KEY")
        if not self.api_key:
            raise ValueError(
                "CLAW API key is required. Set CLAW_API_KEY env var or pass api_key."
            )

        self.base_url = (base_url or os.getenv("CLAW_BASE_URL", "https://models.sjtu.edu.cn/api")).rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _call_api(self, messages: list[dict]) -> str:
        """底层 API 调用，带重试机制。"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # 低温度，保证解释稳定一致
            "max_tokens": 1024,  # 足够生成完整解释，避免中途截断
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                # 标准 OpenAI 格式
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        # 所有重试失败
        raise RuntimeError(f"CLAW API failed after {self.max_retries} retries: {last_error}")

    def generate_explanation(
        self,
        text: str,
        label: int,
        confidence: float,
        event: int,
        evidence: dict,
    ) -> str:
        """
        根据文本、预测标签、置信度、事件类别和正反证据，调用 LLM 生成自然语言解释。
        若 API 失败，自动降级为模板解释。

        Args:
            text: 原始推文文本
            label: 预测标签（0=non-rumor, 1=rumor）
            confidence: 模型置信度（0~1）
            event: 事件类别 ID
            evidence: dict，包含 "predicted" 和 "opposite" 两个证据列表
        """
        label_name = "rumor" if label == 1 else "non-rumor"
        opposite_name = "non-rumor" if label == 1 else "rumor"

        pred_evidence = evidence.get("predicted", [])
        opp_evidence = evidence.get("opposite", [])

        pred_str = ", ".join([f"'{e['token']}'" for e in pred_evidence]) if pred_evidence else "N/A"
        opp_str = ", ".join([f"'{e['token']}'" for e in opp_evidence]) if opp_evidence else "N/A"

        # 构造 Prompt
        system_prompt = (
            "You are a rumor detection assistant with expertise in social media analysis. "
            "Your task is to explain why a tweet was classified as rumor or non-rumor. "
            "Provide a balanced, well-reasoned explanation that: "
            "(1) states the predicted label with confidence, "
            "(2) explains the key evidence supporting the prediction, "
            "(3) acknowledges any counter-evidence that might suggest the opposite label, "
            "(4) references specific words from the tweet as evidence. "
            "Keep the explanation concise but thorough (3–5 sentences). Respond in English."
        )

        user_prompt = (
            f"Tweet: \"{text}\"\n"
            f"Event category: {event}\n"
            f"Predicted label: {label} ({label_name})\n"
            f"Model confidence: {confidence:.2%}\n"
            f"Evidence supporting {label_name}: {pred_str}\n"
            f"Evidence that might suggest {opposite_name}: {opp_str}\n\n"
            f"Explain why this tweet is classified as {label_name}, while also addressing "
            f"why it might have been considered {opposite_name}. Reference specific words "
            f"from the evidence above. Keep your explanation balanced and factual."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            return self._call_api(messages)
        except Exception as e:
            # 降级策略：返回模板解释
            print(f"[CLAWClient] LLM call failed, using fallback template. Error: {e}")
            if label == 1:
                return (
                    f"This tweet is classified as rumor with {confidence:.1%} confidence. "
                    f"It contains suspicious keywords such as {pred_str}, which often "
                    f"appear in unverified claims. However, words like {opp_str} might "
                    f"suggest factual reporting if taken out of context."
                )
            else:
                return (
                    f"This tweet is classified as non-rumor with {confidence:.1%} confidence. "
                    f"Its content is consistent with factual reporting, as indicated by words "
                    f"such as {pred_str}. Nonetheless, terms like {opp_str} could raise "
                    f"minor suspicion in isolation."
                )

    def generate_explanation_batch(
        self,
        items: list[dict],
        sleep_interval: float = 0.5,
    ) -> list[str]:
        """
        批量生成解释，每次调用后 sleep 避免限流。
        items: list of {"text": str, "label": int, "confidence": float,
                         "event": int, "evidence": dict}
        """
        explanations = []
        for i, item in enumerate(items):
            exp = self.generate_explanation(
                item["text"],
                item["label"],
                item["confidence"],
                item["event"],
                item["evidence"],
            )
            explanations.append(exp)
            if i < len(items) - 1:
                time.sleep(sleep_interval)
        return explanations
