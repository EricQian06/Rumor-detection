"""
inference.py — 端到端推理入口
职责：贺

整合 荣/刘/钱 的模块，提供单条/批量推理接口。
用法：
  python inference.py --text "some tweet text" [--event 0]
  python inference.py --input val.csv --output results/predictions.csv
  python inference.py --text "..." --no_llm  # 跳过 LLM，使用模板降级
"""

import os
import sys
import argparse
import json
import pandas as pd
import torch
from tqdm import tqdm
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import RumorClassifier, build_model_and_tokenizer, load_model_from_path
from src.explainer import IGExplainer
from src.llm_client import CLAWClient


def load_model(model_dir: str = "checkpoints", device=None, model_name: str = None):
    """加载训练好的模型和 tokenizer。

    支持两种加载方式：
    - 本地路径：指定 model_dir，从 best_model.pt 加载
    - HuggingFace：如 "charchar2333/Rumor-detection"，直接下载并加载

    Args:
        model_dir: checkpoint 目录（本地模式）或 HuggingFace 模型名
        device: 计算设备
        model_name: 强制指定 backbone 名称；为 None 时从 model_config.json 读取（仅本地模式）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 判断是否从 HuggingFace 加载（模型名包含 "/" 或以 charchar2333 开头）
    is_hf = "/" in str(model_dir) or (model_name and "/" in str(model_name)) or str(model_dir).startswith("charchar")

    if is_hf:
        # 从 HuggingFace 加载
        hf_name = model_name if model_name else str(model_dir)
        print(f"Loading model from HuggingFace: {hf_name}")
        model, tokenizer = build_model_and_tokenizer(hf_name)
        model.to(device)
        model.eval()
        return model, tokenizer, device
    else:
        # 从本地加载
        return load_model_from_path(model_dir, model_name)


def predict_single(
    text: str,
    model,
    tokenizer,
    device,
    explainer: IGExplainer,
    llm_client: CLAWClient,
    event: int = 0,
):
    """对单条文本进行完整推理：分类 + 正反证据提取 + LLM 解释。"""
    # 1. 分类
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

    # 2. 提取正反两方面证据
    evidence = explainer.get_evidence_both_sides(text, label, top_k=5)

    # 3. 生成自然语言解释（含置信度、事件类别、正反证据）
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


def predict_batch(csv_path: str, output_path: str, model, tokenizer, device, explainer, llm_client):
    """批量推理 CSV 文件。"""
    df = pd.read_csv(csv_path)
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inferencing"):
        text = str(row["text"])
        event = int(row.get("event", 0))
        result = predict_single(text, model, tokenizer, device, explainer, llm_client, event=event)

        record = {
            "id": row.get("id", idx),
            "text": text,
            "label": result["label"],
            "confidence": result["confidence"],
            "explanation": result["explanation"],
            "event": event,
        }
        results.append(record)

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"\nBatch inference finished. Results saved to {output_path}")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Rumor Detection Inference")
    parser.add_argument("--text", type=str, default=None, help="Single tweet text to classify")
    parser.add_argument("--event", type=int, default=0, help="Event category ID (for single text)")
    parser.add_argument("--input", type=str, default=None, help="Input CSV for batch inference")
    parser.add_argument("--output", type=str, default="results/predictions.csv", help="Output CSV path")
    parser.add_argument("--model_dir", type=str, default="checkpoints",
                        help="Model path: local directory or HuggingFace model name")
    parser.add_argument("--no_llm", action="store_true", help="Skip LLM explanation, use template fallback")
    args = parser.parse_args()

    if not args.text and not args.input:
        parser.print_help()
        sys.exit(1)

    # 加载模型
    print(f"Loading model from: {args.model_dir}...")
    model, tokenizer, device = load_model(args.model_dir)
    explainer = IGExplainer(model, tokenizer)

    # 初始化 LLM 客户端
    if args.no_llm:
        llm_client = CLAWClient(api_key="dummy", max_retries=0)
    else:
        api_key = os.getenv("CLAW_API_KEY")
        base_url = os.getenv("CLAW_BASE_URL", "https://models.sjtu.edu.cn/api")
        if not api_key:
            print("Warning: CLAW_API_KEY not found in environment or .env file. "
                  "Explanations will use template fallback.")
            llm_client = CLAWClient(api_key="dummy", max_retries=0)
        else:
            llm_client = CLAWClient(api_key=api_key, base_url=base_url, model="minimax")

    # 单条推理
    if args.text:
        result = predict_single(args.text, model, tokenizer, device, explainer, llm_client, event=args.event)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # 批量推理
    if args.input:
        predict_batch(args.input, args.output, model, tokenizer, device, explainer, llm_client)


if __name__ == "__main__":
    main()
