# =============================
# File: scripts/inference_final.py
# Desc: Minimally modified inference script based on infer_bdd_multiframe_h100.py
# =============================
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, json, argparse, warnings, logging, importlib.util, random
from typing import List, Dict, Any, Optional, Tuple
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---- H100 perf knobs ----
try:
    from torch.backends.cuda import sdp_kernel

    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
except Exception:
    pass
os.environ.setdefault("USE_FLASH_ATTENTION_2", "1")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore
from peft import PeftModel

SYSTEM_PROMPT = (
    "你是一个严格的结构化评估助手。只输出一个合法的 JSON 对象，不得包含任何解释性文字、前后缀、代码块或多余字符。"
    "JSON 必须且只能包含这些字段：risk_level, hazards, reasoning, metrics, recommendation, evidence。"
    "字段约束：risk_level ∈ {'low','medium','high'}；hazards 为数组，每项包含 {type:'vehicle', action:'cut_in'/'hard_brake'}；"
    "reasoning 为字符串数组；metrics 建议含 dx_norm, area_ratio, cx_mid_norm, cy_last_norm；"
    "evidence 为数组，每项包含 {frame_idx:int, box2d:[x1,y1,x2,y2]}。"
    "在`evidence`字段中，`box2d`的值是一个包含四个浮点数的数组，分别代表边界框左上角顶点的x和y坐标，以及右下角顶点的x和y坐标 (x1, y1, x2, y2)。"
)
DEFAULT_PROMPT = (
    "仅依据按时间顺序给出的多帧行车画面，输出驾驶风险评估 JSON。必须包含字段: risk_level, hazards[], reasoning[], metrics{}, "
    "recommendation, evidence[]。不得编造未在画面中出现的信息。"
)
REQUIRED_FIELDS = ["risk_level", "hazards", "reasoning", "metrics", "recommendation", "evidence"]


def bnb_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def parse_frame_idx_from_path(p: str) -> Optional[int]:
    m = re.search(r"/([^/]+)-(\d+)\.jpg$", p)
    return int(m.group(2)) if m else None


def allowed_frame_idxs(paths: List[str]) -> List[int]:
    out = []
    for p in paths:
        v = parse_frame_idx_from_path(p)
        if v is not None: out.append(v - 1)
    return out


def load_images(paths: List[str], longest: int = 384, square: bool = False):
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


def build_messages(images, prompt_text: str, allowed_idx: Optional[List[int]] = None):
    content = [{"type": "image", "image": img} for img in images]
    if allowed_idx and args.append_frame_idx_to_prompt:  # <--- 修正: 确保 args 可访问
        prefix = (
            f"本次图片对应的 frame_idx 依次为 {allowed_idx} 。请确保 evidence[].frame_idx 只能从该列表中选择，"
            f"并优先与图片顺序对齐。\n"
        )
        prompt_text = prefix + prompt_text
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def extract_json_block(text: str) -> Optional[str]:
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1] if (s != -1 and e != -1 and e > s) else None


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


def ensure_fields_strict(j: Dict[str, Any]) -> bool:
    if not isinstance(j, dict): return False
    if not all(k in j for k in REQUIRED_FIELDS): return False
    ok = True
    ok &= isinstance(j["hazards"], list) and len(j["hazards"]) >= 1
    ok &= isinstance(j["metrics"], dict) and len(j["metrics"]) >= 1
    ok &= isinstance(j["recommendation"], str) and len(j["recommendation"].strip()) > 0
    ok &= isinstance(j["reasoning"], list)
    return ok


def normalize_json(j: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(j, dict): return {}
    out: Dict[str, Any] = {}
    rl = str(j.get("risk_level", "medium")).lower()
    out["risk_level"] = rl if rl in ("low", "medium", "high") else "medium"
    hz = j.get("hazards", []);
    hh = []
    if isinstance(hz, list):
        for h in hz:
            if not isinstance(h, dict): continue
            a = str(h.get("action", ""))
            a = a.lower().replace("cutin", "cut_in").replace("hardbrake", "hard_brake")
            if a not in ("cut_in", "hard_brake"): continue
            item = {"type": "vehicle", "action": a}
            if "obj_id" in h: item["obj_id"] = str(h["obj_id"])
            hh.append(item)
    out["hazards"] = hh
    rs = j.get("reasoning", [])
    if isinstance(rs, str): rs = [rs]
    out["reasoning"] = [str(x) for x in rs][:6] if isinstance(rs, list) else []
    mout = {}
    if isinstance(j.get("metrics"), dict):
        for k in ["dx_norm", "area_ratio", "cx_mid_norm", "cy_last_norm"]:
            if k in j["metrics"]:
                try:
                    mout[k] = float(j["metrics"][k])
                except Exception:
                    pass
    out["metrics"] = mout
    out["recommendation"] = str(j.get("recommendation", ""))
    ee = [];
    ev = j.get("evidence", [])
    if isinstance(ev, list):
        for e in ev:
            if not isinstance(e, dict): continue
            box = e.get("box2d") or e.get("bbox")
            fi = e.get("frame_idx") if isinstance(e.get("frame_idx"), int) else e.get("frame")
            try:
                if isinstance(box, list) and len(box) == 4:
                    ee.append({"frame_idx": int(fi) if isinstance(fi, int) else None,
                               "box2d": [float(x) for x in box]})
            except Exception:
                pass
    out["evidence"] = ee
    return out


def clamp_evidence_frames(obj: Dict[str, Any], allowed: List[int], mode: str = "by_order") -> Dict[str, Any]:
    if not isinstance(obj, dict) or not allowed: return obj
    if not isinstance(obj.get("evidence"), list): return obj
    if mode == "off": return obj
    if mode == "by_order":
        for k, e in enumerate(obj["evidence"]):
            if isinstance(e, dict): e["frame_idx"] = allowed[min(k, len(allowed) - 1)]
    elif mode == "by_nearest":
        S = set(allowed)
        for k, e in enumerate(obj["evidence"]):
            if not isinstance(e, dict): continue
            fi = e.get("frame_idx")
            if isinstance(fi, int) and fi in S: continue
            base = fi if isinstance(fi, int) else allowed[min(k, len(allowed) - 1)]
            e["frame_idx"] = min(allowed, key=lambda x: abs(x - base))
    return obj


class AtomicJSONLWriter:
    def __init__(self, final_path: str):
        self.final = final_path;
        self.tmp = final_path + ".tmp";
        self.fh = None;
        self.n = 0

    def __enter__(self):
        if self.final:
            os.makedirs(os.path.dirname(self.final) or ".", exist_ok=True)
            self.fh = open(self.tmp, "w", encoding="utf-8")
        return self

    def write(self, obj: Dict[str, Any]):
        if self.fh:
            self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n");
            self.fh.flush();
            self.n += 1

    def __exit__(self, exc_type, exc, tb):
        if self.fh: self.fh.close()
        if self.n > 0 and exc_type is None:
            os.replace(self.tmp, self.final)
        else:
            try:
                os.remove(self.tmp)
            except OSError:
                pass


# ---------- model (保持不变) ----------
def load_model(base_model: str, adapter_path: str, fourbit: bool = False, merge_adapter: bool = False,
               offline: bool = False, device: str = "cuda", attn_impl: str = "sdpa", compile_model: bool = False):
    # (此函数保持不变)
    device_map = {"": "cuda:0"} if device == "cuda" and torch.cuda.is_available() else {"": "cpu"}
    torch_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_available() else torch.float32
    quant_cfg = None
    if fourbit and device == "cuda" and bnb_available() and BitsAndBytesConfig is not None:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                       bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        torch_dtype = None
    local_only = {"local_files_only": True} if (offline or os.environ.get("HF_HUB_OFFLINE") == "1") else {}

    model = AutoModelForVision2Seq.from_pretrained(
        base_model, device_map=device_map, trust_remote_code=True,
        torch_dtype=torch_dtype, quantization_config=quant_cfg,
        attn_implementation=("flash_attention_2" if attn_impl == "fa2" else "sdpa"),
        low_cpu_mem_usage=False, use_safetensors=True, **local_only
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge_adapter:
            try:
                model = model.merge_and_unload();
                print("[Info] LoRA merged & unloaded")
            except Exception:
                pass
    try:
        model.config.attn_implementation = ("flash_attention_2" if attn_impl == "fa2" else "sdpa")
    except Exception:
        pass

    proc = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, use_fast=True, **local_only)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True, **local_only)
    model.eval()
    if compile_model and device == "cuda" and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[Info] torch.compile enabled")
        except Exception as e:
            print(f"[Warn] torch.compile failed: {e}")
    return model, tok, proc


@torch.no_grad()
def generate_once(model, tokenizer, processor, images: List[Image.Image], prompt_text: str,
                  max_new_tokens=512, do_sample=False, temperature=0.2, top_p=0.9,
                  allowed_idx=None):  # <--- 修正: 增加 allowed_idx 参数
    messages = build_messages(images, prompt_text, allowed_idx=allowed_idx)
    tmpl = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[tmpl], images=[images], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id
    out_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=do_sample, temperature=(temperature if do_sample else None), top_p=(top_p if do_sample else None),
        eos_token_id=eos_id, pad_token_id=eos_id, use_cache=True
    )
    new_tokens = out_ids[:, prompt_len:]
    answer = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return answer


# ---------- IO ----------
def read_records(path: str, single: bool):
    with open(path, "r", encoding="utf-8") as f:
        if single: return [json.load(f)]
        return [json.loads(line) for line in f if line.strip()]


# ---------- main ----------
def main():
    global args  # <--- 修正: 声明 args 为全局变量，以便 build_messages 函数可以访问
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--out", default="outputs/predictions.jsonl")
    ap.add_argument("--json_only_out", default="")
    ap.add_argument("--no_text", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--fourbit", action="store_true")
    ap.add_argument("--merge_adapter", action="store_true")
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "fa2"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--image_longest", type=int, default=384)
    ap.add_argument("--square", action="store_true")
    ap.add_argument("--append_frame_idx_to_prompt", action="store_true")
    ap.add_argument("--force_evidence_frames", default="by_order", choices=["off", "by_order", "by_nearest"])
    ap.add_argument("--strict_fields", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(args.seed);
    torch.manual_seed(args.seed)

    model, tok, proc = load_model(
        args.base, args.adapter, fourbit=args.fourbit, merge_adapter=args.merge_adapter,
        offline=args.offline, device=args.device, attn_impl=args.attn_impl, compile_model=args.compile
    )

    records = read_records(args.input, single=args.single)
    if args.limit > 0: records = records[:args.limit]

    total, ok = 0, 0
    with AtomicJSONLWriter(args.out) as wm, AtomicJSONLWriter(args.json_only_out) as wj:
        for ridx, rec in enumerate(tqdm(records, desc="Generating Predictions"), 1):  # <--- 使用tqdm
            try:
                image_paths = rec["images"]
                images = load_images(image_paths, longest=args.image_longest, square=args.square)
                allowed = rec.get("frame_idx_list")
                if allowed:
                    allowed = [int(x) for x in allowed]  # 确保是整数

                user_text = None
                if isinstance(rec.get("conversations"), list):
                    for m in rec["conversations"]:
                        if m.get("role") == "user" and isinstance(m.get("content"), str):
                            user_text = m["content"];
                            break
                prompt_text = user_text or DEFAULT_PROMPT

                # 这里是核心逻辑，保持不变
                reply = generate_once(
                    model, tok, proc, images, prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(not args.greedy), temperature=args.temperature, top_p=args.top_p,
                    allowed_idx=allowed  # <--- 修正: 传递 allowed_idx
                )
                pred_json, ok_json = try_parse_json(reply)

                # 移除了 check_consistency 的调用
                norm_json = normalize_json(pred_json if isinstance(pred_json, dict) else {})

                if args.force_evidence_frames != "off" and allowed:
                    norm_json = clamp_evidence_frames(norm_json, allowed, mode=args.force_evidence_frames)

                fields_ok = ensure_fields_strict(norm_json) if args.strict_fields else (
                        isinstance(norm_json, dict) and all(k in norm_json for k in REQUIRED_FIELDS)
                )

                out_item = {
                    "images": image_paths,
                    "frame_idx_list": allowed,
                    "prediction_json": norm_json,
                    "json_valid": ok_json and isinstance(pred_json, dict),
                    "fields_ok": fields_ok,
                    "prediction_text": reply if not args.no_text else "",
                    "gen_cfg": {  # (保持不变)
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": (not args.greedy),
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "image_longest": args.image_longest,
                        "device": args.device,
                        "attn": args.attn_impl
                    }
                }
                if args.no_text:
                    del out_item["prediction_text"]

                wm.write(out_item)
                if args.json_only_out and fields_ok:
                    wj.write(norm_json)
                total += 1;
                ok += int(fields_ok)

            except Exception as e:
                wm.write({"error": str(e), "record_index": ridx});
                print(f"[Error idx={ridx}] {e}", file=sys.stderr)

    print(f"[Done] total={total} ok(fields_ok)={ok} -> {args.out}")


if __name__ == "__main__":
    main()