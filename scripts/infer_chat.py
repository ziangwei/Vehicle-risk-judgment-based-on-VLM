# scripts/infer_chat.py
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

BASE   = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER= "../saves/qwen2_5_vl7b/bdd_multiframe_sft/checkpoint-1600"
IMGS   = [
  "/绝对路径/f1.jpg",
  "/绝对路径/f2.jpg",
  "/绝对路径/f3.jpg",
  "/绝对路径/f4.jpg",
  "/绝对路径/f5.jpg",
]
PROMPT = "仅依据按时间顺序给出的多帧行车画面，输出驾驶风险评估JSON。必须包含字段: risk_level, hazards[], reasoning[], metrics{}, recommendation, evidence[]。不得编造未在画面中出现的信息。"

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
pro = AutoProcessor.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER).eval()

images = [Image.open(p).convert("RGB") for p in IMGS]
# 构造多模态消息：5 个 image 占位 + 文本
messages = [{"role":"user","content":[*([{"type":"image"}]*len(images)),{"type":"text","text":PROMPT}]}]
inputs = pro.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)
pixel  = pro(images=images, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, **pixel, max_new_tokens=512, temperature=0.2)
print(tok.decode(out[0], skip_special_tokens=True))
