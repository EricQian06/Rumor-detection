"""
colab_train_safe.py — 在 Colab 中稳定训练的脚本
用法（在 Colab 单元格中）:
    %run colab_train_safe.py

相比 !python -m src.train 的优势：
- 同一进程运行，子进程 OOM 不会导致运行时断开
- 自动监控 GPU 内存，OOM 前主动报错
- 清晰的错误堆栈，不会只显示 [object CloseEvent]
"""

import os
import sys
import gc
import argparse
import json
import torch

# 打印 GPU 信息
print("=" * 50)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_vram:.2f} GB")
print("=" * 50)

# 清理显存缓存
torch.cuda.empty_cache()
gc.collect()

# 设置更保守的参数默认值（防止 OOM）
class SafeArgs:
    train_csv = "train.csv"
    val_csv = "val.csv"
    # 优化模型选择（按 Colab T4 显存安全排序）：
    #   roberta-base       (125M, 最安全, bs=32, lr=2e-5)
    #   microsoft/deberta-v3-base (86M, bs=16 + accum=2, lr=1e-5)
    #   roberta-large      (355M, bs=8 + accum=2, lr=1e-5)
    model_name = "microsoft/deberta-v3-base"
    epochs = 10            # 增加 epoch 上限，EarlyStopping 会在收敛后自动停止
    batch_size = 16        # DeBERTa attention 显存占用大，T4 上 16 安全
    accumulation_steps = 2 # 梯度累积，等效 batch_size=32，稳定训练
    lr = 1e-5              # DeBERTa 对学习率更敏感，1e-5 通常比 2e-5 更好
    max_len = 256
    output_dir = "checkpoints"
    num_workers = 2
    patience = 3           # EarlyStopping：val_acc 连续 3 轮不提升则停止

args = SafeArgs()

# 你也可以在这里手动覆盖参数，例如：
# args.model_name = "roberta-large"
# args.batch_size = 16
# args.epochs = 3

print("\n训练参数:")
for k, v in vars(args).items():
    print(f"  {k}: {v}")
print()

try:
    # 直接导入训练函数，避免子进程
    from src.train import train
    train(args)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n" + "=" * 50)
        print("ERROR: GPU 显存不足 (OOM)！")
        print("=" * 50)
        print("\n解决方案（按顺序尝试）：")
        print("1. 修改本脚本中的 args.batch_size = 8（甚至 4）")
        print("2. 修改 args.accumulation_steps = 1（降低等效 batch_size）")
        print("3. 修改 args.max_len = 128")
        print("4. 重启运行时：代码执行程序 → 重新启动代码执行程序")
        print("=" * 50)
        # 清空显存，防止后续单元格也失败
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        raise
except Exception as e:
    print("\n训练过程中发生错误:")
    print(type(e).__name__, ":", e)
    import traceback
    traceback.print_exc()
    raise
finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n显存已清理，可以继续执行后续单元格。")
