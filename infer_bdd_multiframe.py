#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL-7B-Instruct + LoRA 多帧推理（整洁JSON / V100兼容 / 原子写入）
"""

import os, sys, json, argparse, warnings, logging, importlib.util, random
from typing import List, Dict, Any, Optional, Tuple
import torch
from PIL import Image, ImageOps

# ---------- 安静一些 ----------
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # 关闭 xet 分块
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------- 禁用不兼容注意力（V100 强烈建议） ----------
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
except Exception:
    pass
os.environ.setdefault("USE_FLASH_ATTENTION_2", "0")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("PYTORCH_ENABLE_SDPA_HEURISTICS", "0")

# ---------- Transformers ----------
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore

from peft import PeftModel

# ---------- 提示词（只输出合法 JSON） ----------
SYSTEM_PROMPT = (
    "你是一个严格的结构化评估助手。只输出一个合法的 JSON 对象，不得包含任何解释性文字、前后缀、"
    "代码块或多余字符。JSON 必须且只能包含这些字段：risk_level, hazards, reasoning, metrics, "
    "recommendation, evidence。字段约束：risk_level ∈ {'low','medium','high'}；"
    "hazards 为数组，每项包含 {type:'vehicle', action:'cut_in'或'hard_brake'}，可选 obj_id；"
    "reasoning 为字符串数组；metrics 为对象，建议包含 dx_norm, area_ratio, cx_mid_norm, cy_last_norm；"
    "recommendation 为字符串；evidence 为数组，每项包含 {frame_idx:int, box2d:[x1,y1,x2,y2]}。"
)
DEFAULT_PROMPT = (
    "仅依据按时间顺序给出的多帧行车画面，输出驾驶风险评估 JSON。必须包含字段: "
    "risk_level, hazards[], reasoning[], metrics{}, recommendation, evidence[]。不得编造未在画面中出现的信息。"
)
REQUIRED_FIELDS = ["risk_level", "hazards", "reasoning", "metrics", "recommendation", "evidence"]


# ============== 工具函数 ==============
def bnb_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def load_images(paths: List[str], longest: int = 192, square: bool = False):
    imgs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        img = Image.open(p).convert("RGB")
        img.thumbnail((longest, longest), Image.BICUBIC)
        if square:
            img = ImageOps.pad(img, (longest, longest), method=Image.BICUBIC, color=(0, 0, 0))
        imgs.append(img)
    return imgs


def build_messages(images, prompt_text: str):
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": content},
    ]


def extract_json_block(text: str) -> Optional[str]:
    s, e = text.find("{"), text.rfind("}")
    return text[s:e+1] if (s != -1 and e != -1 and e > s) else None


def try_parse_json(text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    try:
        return json.loads(text), True
    except Exception:
        blk = extract_json_block(text)
        if blk is not None:
            try:
                return json.loads(blk), True
            except Exception:
                return {"_raw": text, "_json_block": blk}, False
        return {"_raw": text}, False


def ensure_fields(j: Dict[str, Any]) -> bool:
    return isinstance(j, dict) and all(k in j for k in REQUIRED_FIELDS)


def normalize_json(j: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(j, dict):
        return j
    out: Dict[str, Any] = {}

    rl = str(j.get("risk_level", "medium")).lower()
    if rl not in ("low", "medium", "high"):
        rl = "medium"
    out["risk_level"] = rl

    hazards = j.get("hazards", [])
    if not isinstance(hazards, list): hazards = []
    fixed_h = []
    for h in hazards:
        if not isinstance(h, dict): continue
        a = str(h.get("action", "")).lower().replace("cutin", "cut_in").replace("hardbrake", "hard_brake")
        if a not in ("cut_in", "hard_brake"): continue
        item = {"type": "vehicle", "action": a}
        if "obj_id" in h: item["obj_id"] = str(h["obj_id"])
        fixed_h.append(item)
    out["hazards"] = fixed_h

    reasoning = j.get("reasoning", [])
    if isinstance(reasoning, str): reasoning = [reasoning]
    if not isinstance(reasoning, list): reasoning = []
    out["reasoning"] = [str(x) for x in reasoning][:6]

    def tofloat(v, default=None):
        try: return float(v)
        except Exception: return default

    m = j.get("metrics", {})
    if not isinstance(m, dict): m = {}
    out["metrics"] = {k: tofloat(m.get(k), None)
                      for k in ["dx_norm", "area_ratio", "cx_mid_norm", "cy_last_norm"] if k in m}

    out["recommendation"] = str(j.get("recommendation", ""))

    evid = j.get("evidence", [])
    if not isinstance(evid, list): evid = []
    fixed_e = []
    for ev in evid:
        if not isinstance(ev, dict): continue
        fi = ev.get("frame_idx", ev.get("frame"))
        try: fi = int(fi)
        except Exception: continue
        box = ev.get("box2d", ev.get("bbox"))
        if isinstance(box, list) and len(box) == 4:
            try:
                box = [float(x) for x in box]
                fixed_e.append({"frame_idx": fi, "box2d": box})
            except Exception:
                pass
    out["evidence"] = fixed_e
    return out


# ---------- 原子写入，避免空文件 ----------
class AtomicJSONLWriter:
    def __init__(self, final_path: str):
        self.final = final_path
        self.tmp = final_path + ".tmp"
        self.fh = None
        self.n_written = 0

    def __enter__(self):
        if self.final:
            os.makedirs(os.path.dirname(self.final) or ".", exist_ok=True)
            self.fh = open(self.tmp, "w", encoding="utf-8")
        return self

    def write_line(self, obj: Dict[str, Any]):
        if self.fh:
            self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.fh.flush()
            self.n_written += 1

    def __exit__(self, exc_type, exc, tb):
        if not self.fh:
            return
        self.fh.close()
        if self.n_written > 0 and exc_type is None:
            os.replace(self.tmp, self.final)
        else:
            try: os.remove(self.tmp)
            except OSError: pass


# ============== 加载模型 ==============
def load_model(base_model: str, adapter_path: str,
               fourbit: bool = False, merge_adapter: bool = False,
               offline: bool = False, device: str = "cuda"):
    # 设备映射
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda 但当前不可用 CUDA。")
        torch.cuda.set_device(0)
        device_map = {"": "cuda:0"}
        torch_dtype = torch.float16
    elif device == "cpu":
        device_map = {"": "cpu"}
        torch_dtype = torch.float32   # CPU 用 fp32 更稳
        if fourbit:
            print("[WARN] CPU 上不支持 4bit 量化，自动回退到 FP32。")
            fourbit = False
    else:  # auto
        device_map = "auto"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 量化
    quant_cfg = None
    if fourbit:
        if bnb_available() and BitsAndBytesConfig is not None:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            # 量化时不要强制 dtype
            torch_dtype = None
        else:
            print("[WARN] --fourbit 指定了但未安装 bitsandbytes；自动回退到非量化。")
            fourbit = False

    local_only = {"local_files_only": True} if offline or os.environ.get("HF_HUB_OFFLINE") == "1" else {}

    # 重点：attn_implementation='eager'，禁止懒加载（low_cpu_mem_usage=False）
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
        use_safetensors=True,
        **local_only
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge_adapter:
            try:
                model = model.merge_and_unload()
            except Exception:
                pass

    try:
        model.config.attn_implementation = "eager"
        if hasattr(model.config, "use_flash_attn"):
            model.config.use_flash_attn = False
    except Exception:
        pass

    proc = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, use_fast=False, **local_only)
    tok  = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False, **local_only)
    model.eval()
    return model, tok, proc


# ============== 生成 ==============
@torch.no_grad()
def generate_once(model, tokenizer, processor, images: List[Image.Image], prompt_text: str,
                  max_new_tokens=128, do_sample=False, temperature=0.2, top_p=0.9) -> str:
    messages = build_messages(images, prompt_text)
    tmpl = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[tmpl], images=[images], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        use_cache=True
    )
    new_tokens = gen_ids[:, prompt_len:]
    answer = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return answer


# ============== IO ==============
def read_records(path: str, single: bool):
    with open(path, "r", encoding="utf-8") as f:
        if single:
            return [json.load(f)]
        return [json.loads(line) for line in f if line.strip()]


# ============== 主函数 ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--out", default="outputs/predictions.jsonl")
    ap.add_argument("--json_only_out", default="", help="若指定，仅包含规范化 JSON 的 JSONL 路径")
    ap.add_argument("--no_text", action="store_true", help="不保存 prediction_text")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offline", action="store_true", help="仅本地加载（不访问 Hub）")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"],
                    help="模型放置设备：cuda(单卡)、cpu、auto(Accelerate自动分片)；V100建议cuda或cpu。")

    # 生成/显存
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--fourbit", action="store_true")
    ap.add_argument("--merge_adapter", action="store_true")

    # 图像
    ap.add_argument("--image_longest", type=int, default=192)
    ap.add_argument("--square", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(args.seed); torch.manual_seed(args.seed)

    model, tok, proc = load_model(
        args.base, args.adapter,
        fourbit=args.fourbit,
        merge_adapter=args.merge_adapter,
        offline=args.offline,
        device=args.device
    )

    records = read_records(args.input, single=args.single)
    if args.limit > 0:
        records = records[:args.limit]

    total, ok = 0, 0
    with AtomicJSONLWriter(args.out) as w_main, AtomicJSONLWriter(args.json_only_out) as w_json:
        for ridx, rec in enumerate(records, 1):
            try:
                image_paths = rec["images"]
                images = load_images(image_paths, longest=args.image_longest, square=args.square)
                prompt_text = next((m["content"] for m in rec.get("conversations", [])
                                   if m.get("role")=="user" and isinstance(m.get("content"), str)), DEFAULT_PROMPT)

                # 第一次
                reply = generate_once(
                    model, tok, proc, images, prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=not args.greedy,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                pred_json, ok_json = try_parse_json(reply)

                # 失败重试（短一点、贪心）
                if not ok_json:
                    retry_prompt = DEFAULT_PROMPT + " 请只输出一个合法 JSON，不要包含任何多余字符。"
                    reply2 = generate_once(
                        model, tok, proc, images, retry_prompt,
                        max_new_tokens=max(96, args.max_new_tokens // 2),
                        do_sample=False
                    )
                    pred_json2, ok2 = try_parse_json(reply2)
                    if ok2:
                        reply, pred_json, ok_json = reply2, pred_json2, True

                norm_json = normalize_json(pred_json if isinstance(pred_json, dict) else {})
                fields_ok = ensure_fields(norm_json)

                out_item = {
                    "images": image_paths,
                    "prompt": prompt_text,
                    "prediction_json": norm_json,
                    "json_valid": ok_json and isinstance(pred_json, dict),
                    "fields_ok": fields_ok,
                    "gen_cfg": {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": (not args.greedy),
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "fourbit": args.fourbit,
                        "image_longest": args.image_longest,
                        "square": args.square,
                        "device": args.device
                    }
                }
                if not args.no_text:
                    out_item["prediction_text"] = reply

                w_main.write_line(out_item)
                if args.json_only_out and fields_ok:
                    w_json.write_line(norm_json)

                total += 1
                ok += int(fields_ok)
                print(f"[{ridx}/{len(records)}] fields_ok={fields_ok}")

            except Exception as e:
                w_main.write_line({"error": str(e), "record_index": ridx})
                print(f"[Error idx={ridx}] {e}", file=sys.stderr)

    print(f"[Done] total={total} ok(fields_ok)={ok} -> {args.out}")


if __name__ == "__main__":
    main()
