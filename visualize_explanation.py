"""
visualize_explanation.py — 单条文本归因可视化脚本
职责：钱

用法：
  python visualize_explanation.py --text "some tweet" --model_dir checkpoints
  python visualize_explanation.py --text "some tweet" --model_dir charchar2333/Rumor-detection
"""

import os
import sys
import argparse
import json
import torch
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import load_model
from src.explainer import IGExplainer


def main():
    parser = argparse.ArgumentParser(description="Visualize token attribution for one tweet")
    parser.add_argument("--text", type=str, required=True, help="Tweet text to explain")
    parser.add_argument("--model_dir", type=str, default="checkpoints",
                        help="Model path: local directory or HuggingFace model name")
    parser.add_argument("--target_label", type=int, choices=[0, 1], default=None,
                        help="Attribution target label; omitted means predicted label")
    parser.add_argument("--output_dir", type=str, default="results/explanation_examples",
                        help="Directory to save attribution HTML and JSON")
    parser.add_argument("--prefix", type=str, default="single_text",
                        help="Output filename prefix")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, device = load_model(args.model_dir, device=device)

    explainer = IGExplainer(model, tokenizer)
    result = explainer.visualize_text_attribution(
        text=args.text,
        target_label=args.target_label,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )

    print(json.dumps({
        "pred_label": result["pred_label"],
        "pred_prob": round(result["pred_prob"], 4),
        "target_label": result["target_label"],
        "target_prob": round(result["target_prob"], 4),
        "html_path": result["html_path"],
        "json_path": result["json_path"],
    }, indent=2, ensure_ascii=False))
    print("\nOpen the HTML file in a browser to view green positive and red negative token attributions.")


if __name__ == "__main__":
    main()
