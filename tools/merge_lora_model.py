"""
merge_lora_weights.py
将 baseline LoRA 微调权重合并到基础模型，导出完整的微调后模型。

用法:
    python merge_lora_weights.py \
        --base_model_path /path/to/CogVideoX-5b-I2V \
        --lora_path /path/to/checkpoint-xxx \
        --output_path /path/to/merged_model
"""

import argparse
import torch
import shutil
from pathlib import Path
from diffusers import (
    CogVideoXTransformer3DModel,
    CogVideoXImageToVideoPipeline,
)
from peft import PeftModel, LoraConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="原始预训练模型路径，如 CogVideoX-5b-I2V")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="LoRA checkpoint 目录，包含 pytorch_lora_weights.safetensors")
    parser.add_argument("--output_path", type=str, required=True,
                        help="合并后模型的输出路径")
    args = parser.parse_args()

    base_path = Path(args.base_model_path)
    lora_path = Path(args.lora_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========== 1. 加载完整 Pipeline ==========
    print("[1/4] Loading base pipeline...")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        str(base_path),
        torch_dtype=torch.float16,
    )

    # ========== 2. 加载 LoRA 权重 ==========
    print("[2/4] Loading LoRA weights...")
    # load_lora_weights 会自动找 pytorch_lora_weights.safetensors
    pipe.load_lora_weights(str(lora_path))

    # ========== 3. 合并 LoRA 到基础权重 ==========
    print("[3/4] Merging LoRA weights into base model...")
    # fuse_lora 将 LoRA 的 delta 权重加到原始权重上
    pipe.fuse_lora()
    # unload_lora 移除 LoRA adapter 结构，只保留合并后的权重
    pipe.unload_lora_weights()

    # ========== 4. 保存合并后的完整模型 ==========
    print("[4/4] Saving merged model...")
    pipe.save_pretrained(str(output_path))

    print(f"\n 合并完成！模型已保存到: {output_path}")
    print(f"   目录结构:")
    for item in sorted(output_path.rglob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   {item.relative_to(output_path)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
