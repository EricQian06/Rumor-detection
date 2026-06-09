"""
从 val.csv 随机抽取 10 条样本进行端到端测试，输出结果到终端。
用法：python test_sample.py
"""

import os
import sys
import random
import json
import argparse

import pandas as pd
import torch
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import build_model_and_tokenizer, load_model_from_path
from src.explainer import IGExplainer
from src.llm_client import CLAWClient


def load_model(model_dir: str = "checkpoints", device=None, model_name: str = None):
    """加载模型，支持本地路径或 HuggingFace。

    Args:
        model_dir: 本地目录或 HuggingFace 模型名（如 "charchar2333/Rumor-detection"）
        device: 计算设备
        model_name: 强制指定 backbone 名称
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 判断是否从 HuggingFace 加载
    is_hf = "/" in str(model_dir) or (model_name and "/" in str(model_name)) or str(model_dir).startswith("charchar")

    if is_hf:
        hf_name = model_name if model_name else str(model_dir)
        print(f"Loading model from HuggingFace: {hf_name}")
        model, tokenizer = build_model_and_tokenizer(hf_name)
        model.to(device)
        model.eval()
        return model, tokenizer, device
    else:
        return load_model_from_path(model_dir, model_name)


def predict_single(text: str, model, tokenizer, device, explainer, llm_client, event: int = 0):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        label = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][label].item())

    evidence = explainer.get_evidence_both_sides(text, label, top_k=5)
    explanation = llm_client.generate_explanation(
        text=text,
        label=label,
        confidence=confidence,
        event=event,
        evidence=evidence,
    )

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "explanation": explanation,
        "evidence": evidence,
        "event": event,
    }


def main():
    parser = argparse.ArgumentParser(description="随机测试谣言检测模型")
    parser.add_argument("--model_dir", type=str, default="checkpoints",
                        help="模型路径（本地目录或 HuggingFace 模型名）")
    args = parser.parse_args()

    print("=" * 80)
    print("随机抽取 10 条 val.csv 样本进行端到端测试")
    print("=" * 80)

    # 1. 加载模型
    print("\n[1/3] 加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    model, tokenizer, device = load_model(args.model_dir)
    explainer = IGExplainer(model, tokenizer)

    # 2. 初始化 LLM 客户端
    print("[2/3] 初始化 CLAW API 客户端...")
    api_key = os.getenv("CLAW_API_KEY")
    base_url = os.getenv("CLAW_BASE_URL", "https://models.sjtu.edu.cn/api")
    if not api_key:
        print("    Warning: CLAW_API_KEY not found in environment or .env file. "
              "Using template fallback for explanations.")
        llm_client = CLAWClient(api_key="dummy", max_retries=0)
    else:
        llm_client = CLAWClient(api_key=api_key, base_url=base_url, model="minimax")
        print("    客户端就绪")

    # 3. 读取数据并随机抽样
    print("[3/3] 读取 val.csv 并随机抽取 10 条（5 条 non-rumor + 5 条 rumor）...")
    df = pd.read_csv("val.csv")
    df_0 = df[df["label"] == 0].sample(n=5, random_state=42)
    df_1 = df[df["label"] == 1].sample(n=5, random_state=42)
    samples = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. 逐条推理
    print("\n" + "=" * 80)
    print("开始推理...")
    print("=" * 80)

    correct = 0
    for idx, row in samples.iterrows():
        text = str(row["text"]).replace('\xa0', ' ').replace('​', '')
        true_label = int(row["label"])
        event = int(row["event"])

        result = predict_single(text, model, tokenizer, device, explainer, llm_client, event=event)
        pred_label = result["label"]
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1

        status = "[OK] 正确" if is_correct else "[X] 错误"
        label_map = {0: "non-rumor", 1: "rumor"}

        print(f"\n{'-' * 80}")
        print(f"【样本 {idx + 1}/10】{status}")
        print(f"{'─' * 80}")
        print(f"ID:        {row['id']}")
        print(f"Event:     {event}")
        print(f"原文:      {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"真实标签:  {true_label} ({label_map[true_label]})")
        print(f"预测标签:  {pred_label} ({label_map[pred_label]})")
        print(f"置信度:    {result['confidence']:.2%}")
        print(f"\n[解释]\n{result['explanation']}")
        print(f"\n[证据 - 支持预测]:")
        for e in result["evidence"]["predicted"]:
            print(f"    - '{e['token']}': {e['score']:.4f}")
        print(f"\n[证据 - 支持对立]:")
        for e in result["evidence"]["opposite"]:
            print(f"    - '{e['token']}': {e['score']:.4f}")

    print(f"\n{'=' * 80}")
    print(f"测试完成！准确率: {correct}/10 ({correct * 10}%)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
