# scripts/infer_baseline.py
# Desc: Inference script for running the baseline model without any adapters.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import argparse
import warnings
import logging
import importlib.util
import random
from typing import List, Dict, Any, Optional, Tuple
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

# --- 基本设置 ---
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Transformers ---
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

# --- 提示词 ---
SYSTEM_PROMPT = (
    "你是一个严格的结构化评估助手。只输出一个合法的 JSON 对象，不得包含任何解释性文字、前后缀、代码块或多余字符。"
    "JSON 必须且只能包含这些字段：risk_level, hazards, reasoning, metrics, recommendation, evidence。"
)
DEFAULT_PROMPT = (
    "仅依据按时间顺序给出的多帧行车画面，输出驾驶风险评估 JSON。必须包含字段: risk_level, hazards[], reasoning[], metrics{}, "
    "recommendation, evidence[]。不得编造未在画面中出现的信息。"
)


# --- 工具函数 ---

class AtomicJSONLWriter:
    """原子写入，防止中断时生成空文件或不完整文件"""

    def __init__(self, final_path: str):
        self.final = final_path
        self.tmp = final_path + ".tmp"
        self.fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.final) or ".", exist_ok=True)
        self.fh = open(self.tmp, "w", encoding="utf-8")
        return self

    def write(self, obj: dict):
        self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def __exit__(self, exc_type, exc, tb):
        if self.fh:
            self.fh.close()
            if exc_type is None:
                os.replace(self.tmp, self.final)
            else:
                try:
                    os.remove(self.tmp)
                except OSError:
                    pass


def load_images(paths: list[str], longest: int):
    """加载并缩放图片"""
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img.thumbnail((longest, longest), Image.BICUBIC)
        imgs.append(img)
    return imgs


def extract_json_block(text: str):
    """从可能包含额外文本的字符串中提取最外层的JSON块"""
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1] if s != -1 and e != -1 and e > s else None


def try_parse_json(text: str):
    """尝试解析JSON，如果失败则尝试提取JSON块再解析"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_block = extract_json_block(text)
        if json_block:
            try:
                return json.loads(json_block)
            except json.JSONDecodeError:
                return {"_raw_text": text, "_error": "Failed to parse extracted JSON block."}
        return {"_raw_text": text, "_error": "No valid JSON found."}


# --- 模型加载与生成 ---

def load_model(base_path, device, attn_impl, use_compile):
    """加载基础模型、Tokenizer和Processor"""
    device_map = {"": device}
    torch_dtype = torch.bfloat16 if "cuda" in device else torch.float32

    # 只加载基础预训练模型
    model = AutoModelForVision2Seq.from_pretrained(
        base_path,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        use_safetensors=True
    )
    model.eval()

    if use_compile and "cuda" in device:
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[Info] torch.compile enabled.")
        except Exception as e:
            print(f"[Warning] torch.compile failed: {e}")

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=True)
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)

    return model, tokenizer, processor


@torch.no_grad()
def generate_one(model, tokenizer, processor, images, prompt_text, max_new_tokens):
    """执行一次模型生成"""
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]

    template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[template], images=[images], return_tensors="pt").to(model.device)

    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    output_ids = gen_ids[:, input_len:]

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


# --- 主程序 ---

def main():
    parser = argparse.ArgumentParser(description="Baseline Inference Engine for VLM Risk Assessment.")
    # 路径参数
    parser.add_argument("--input", required=True, help="Path to the input conversation JSONL file.")
    parser.add_argument("--out", required=True, help="Path to save the output prediction JSONL file.")
    parser.add_argument("--base", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model identifier.")

    # 推理参数
    parser.add_argument("--image_longest", type=int, default=448, help="Resize image's longest edge to this size.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation.")

    # 性能参数
    parser.add_argument("--device", default="cuda:0", help="Device to run inference on.")
    parser.add_argument("--attn_impl", default="sdpa",
                        help="Attention implementation (e.g., 'sdpa', 'flash_attention_2').")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for optimization.")

    args = parser.parse_args()

    print("Loading BASELINE model...")
    model, tokenizer, processor = load_model(args.base, args.device, args.attn_impl, args.compile)

    with open(args.input, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    print(f"Starting BASELINE inference on {len(records)} samples...")
    with AtomicJSONLWriter(args.out) as writer:
        for record in tqdm(records, desc="Generating Baseline Predictions"):
            try:
                images = load_images(record["images"], args.image_longest)
                user_prompt = next((turn["content"] for turn in record["conversations"] if turn["role"] == "user"),
                                   DEFAULT_PROMPT)
                reply_text = generate_one(model, tokenizer, processor, images, user_prompt, args.max_new_tokens)
                pred_json = try_parse_json(reply_text)

                output_record = {
                    "images": record["images"],
                    "frame_idx_list": record.get("frame_idx_list"),
                    "prediction_text": reply_text,
                    "prediction_json": pred_json
                }
                writer.write(output_record)
            except Exception as e:
                print(f"Error processing record: {record.get('images', [])}. Reason: {e}", file=sys.stderr)
                writer.write({"error": str(e), "source_record": record})

    print(f"\nBaseline inference complete. Results saved to {args.out}")


if __name__ == "__main__":
    main()