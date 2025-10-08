import json, argparse, random, sys
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

# ---------------- Args ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=5, help="frames per sample (odd number suggested)")
ap.add_argument("--min_gap", type=int, default=5, help="dedup: min distance between accepted centers for same seq+obj+event")
ap.add_argument("--max_train_per_event", type=int, default=10000, help="cap per event type for train")  # 控制规模
ap.add_argument("--max_val_per_event", type=int, default=1200, help="cap per event type for val")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed)

ROOT = Path(__file__).resolve().parent.parent
MAN_DIR = ROOT / "data/interim/manifests"
EVT_DIR = ROOT / "data/interim/events"
OUT_DIR = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取 manifest → 映射 seq_id -> frames 路径列表
def load_manifest(split):
    m = {}
    with open(MAN_DIR / f"manifest_{split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            m[rec["seq_id"]] = rec["frames"]
    return m

def read_label_frames(label_path: Path):
    data = json.load(open(label_path))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("frames") or data.get("video", {}).get("frames") or []
    return []

def find_box_for_obj(frame_rec, obj_id):
    labels = frame_rec.get("labels") if isinstance(frame_rec, dict) else None
    if not labels:
        return None
    for lab in labels:
        lid = lab.get("id", lab.get("track_id", lab.get("index")))
        if str(lid) == str(obj_id):
            b2d = lab.get("box2d")
            if isinstance(b2d, dict) and {"x1","y1","x2","y2"} <= b2d.keys():
                return [float(b2d["x1"]), float(b2d["y1"]), float(b2d["x2"]), float(b2d["y2"])]
    return None

def risk_from_metrics(ev):
    dx = abs(ev["metrics"].get("abs_dx_norm", ev["metrics"].get("dx_norm", 0.0)))
    ar = ev["metrics"].get("area_ratio", 1.0)
    # 简单规则：先看高风险，再看中等，否则低
    if dx >= 0.18 or ar >= 1.8:
        return "high", ["lateral_shift_fast" if dx >= 0.18 else "rapid_approach"]
    if dx >= 0.12 or ar >= 1.4:
        return "medium", ["noticeable_shift" if dx >= 0.12 else "distance_closing"]
    return "low", ["mild_change"]

def recommendation_from_event(ev_type, risk):
    if ev_type == "cut_in":
        return "prepare_brake_and_keep_gap" if risk != "low" else "monitor_adjacent_vehicle"
    if ev_type == "hard_brake":
        return "brake_then_increase_following_distance" if risk != "low" else "keep_alert_and_cover_brake"
    return "drive_normally"

def draw_bbox_on_image(original_path: str, box: list, save_path: str, color="red", width=3):
    """
    打开一张图片，在指定位置画一个框，然后保存到新的路径。
    """
    try:
        img = Image.open(original_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        # box 格式是 [x1, y1, x2, y2]
        draw.rectangle(box, outline=color, width=width)
        save_path_p = Path(save_path)
        save_path_p.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Error: Failed to draw bbox on {original_path}. Reason: {e}", file=sys.stderr)
        return False

def build_samples(split, cap_per_event):
    seq2frames = load_manifest(split)
    events_path = EVT_DIR / f"events_{split}.jsonl"
    out_path = OUT_DIR / f"bdd_multiframe_sft_{split}.jsonl"

    # 去重记忆：同 (seq,obj,event) 最近一次被采纳的 center_idx
    last_pick = {}
    # 计数：每类的已写条数
    written = {"cut_in": 0, "hard_brake": 0, "normal_driving": 0}
    kept = 0

    # 新建一个目录，用于存放我们画好框的新图片
    PROCESSED_IMG_DIR = ROOT / f"data/processed/images_with_bbox_{split}"
    PROCESSED_IMG_DIR.mkdir(parents=True, exist_ok=True)

    processed_keys = set()
    if out_path.exists():
        print(f"[{split}] Resuming. Reading existing records from {out_path}...")
        with open(out_path, "r", encoding="utf-8") as f_read:
            for line in f_read:
                try:
                    # 从已有的输出文件中解析出唯一的 "身份证"
                    # 这里我们通过解析 response 里的 obj_id 和 evidence 里的第一个 frame_idx
                    record = json.loads(line)
                    resp = json.loads(record["response"])
                    seq_id_from_img = Path(record["images"][0]).parts[-2]  # 从图片路径获取seq_id
                    obj_id = resp["hazards"][0]["obj_id"]
                    center_idx_surrogate = resp["evidence"][0]["frame_idx"]  # 用第一个frame_idx作为近似的center
                    key = (seq_id_from_img, str(obj_id), center_idx_surrogate)
                    processed_keys.add(key)
                except Exception:
                    continue  # 如果有坏行，跳过
        print(f"[{split}] Found {len(processed_keys)} completed records. Will skip them.")
    # 准备原始事件列表
    all_events = []
    with open(events_path, "r", encoding="utf-8") as f_in:
        all_events = [json.loads(line) for line in f_in]

    # 关键修改：使用 "a" (append) 模式打开文件
    with open(out_path, "a", encoding="utf-8") as f_out, \
            tqdm(total=len(all_events), desc=f"Processing {split}") as pbar:

        for ev in all_events:
            pbar.update(1)  # 更新进度条

            seq_id = ev["seq_id"]
            obj_id = ev["obj_id"]
            center = int(ev["center_idx"])

            # 构造近似key来检查是否已处理
            # 注意：这里的center_idx可能和我们从evidence里解析的不完全一样，但可以作为高效的跳过检查
            surrogate_key = (seq_id, str(obj_id), ev["window"][0])
            if surrogate_key in processed_keys:
                continue

            # ... (后续的逻辑完全不变)
            ev_type = ev["event"]
            if written.get(ev_type, 0) >= cap_per_event:
                continue

            frames_all = seq2frames.get(seq_id)
            if not frames_all: continue

            key = (seq_id, str(obj_id), ev_type)
            last = last_pick.get(key, -10 ** 9)
            if abs(center - last) < args.min_gap: continue

            win = ev["window"]
            if args.frames < len(win):
                mid = len(win) // 2
                half = args.frames // 2
                sel = win[max(0, mid - half): max(0, mid - half) + args.frames]
            else:
                sel = win

            labels = read_label_frames(Path(ev["label"]))
            imgs, evidence_boxes = [], []
            ok = True
            for t in sel:
                if t >= len(frames_all) or t >= len(labels): ok = False; break
                box = find_box_for_obj(labels[t], obj_id)
                if box is None: ok = False; break
                imgs.append(frames_all[t])
                evidence_boxes.append({"frame_idx": int(t), "box2d": [round(v, 2) for v in box]})

            if not ok or len(imgs) < 2: continue

            imgs_processed = []
            drawing_ok = True
            for i, original_path_str in enumerate(imgs):
                original_path = Path(original_path_str)
                box_to_draw = evidence_boxes[i]["box2d"]

                new_img_name = f"{original_path.stem}_obj_{obj_id}.jpg"
                save_path = PROCESSED_IMG_DIR / seq_id / new_img_name

                # 同样可以增加一个检查，如果图片已存在，就不用重复画了
                if save_path.exists():
                    imgs_processed.append(str(save_path))
                    continue

                if not draw_bbox_on_image(original_path_str, box_to_draw, str(save_path)):
                    drawing_ok = False
                    break
                imgs_processed.append(str(save_path))

            if not drawing_ok: continue

            if ev_type == "normal_driving":
                risk = "low"
                reasons = ["stable_movement", "no_notable_risk"]
                recommendation = "continue_driving"
                # 对于低风险事件，hazards列表为空
                hazards = []
            else:
                # 这里是您原来的逻辑，保持不变
                risk, reasons = risk_from_metrics(ev)
                recommendation = recommendation_from_event(ev_type, risk)
                hazards = [{"obj_id": str(obj_id), "type": "vehicle", "action": ev_type}]

            resp = {
                "risk_level": risk,
                "hazards": hazards,
                "reasoning": reasons,
                "metrics": {k: ev["metrics"].get(k) for k in
                            ["dx_norm", "area_ratio", "cx_mid_norm", "cy_last_norm"]},
                "recommendation": recommendation,
                "evidence": evidence_boxes
            }
            sample = {
                "images": imgs_processed,
                "prompt": ("请根据提供的时间连续帧图像，对图中高亮框出的车辆行为进行分析，并输出驾驶风险评估JSON。"
                           "必须包含字段: risk_level, hazards[], reasoning[], metrics{}, recommendation, evidence[].不得编造未在画面中出现的信息。"),
                "response": json.dumps(resp, ensure_ascii=False)
            }
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written[ev_type] = written.get(ev_type, 0) + 1
            last_pick[key] = center

    # 统计最终写入了多少
    kept = sum(1 for _ in open(out_path, "r", encoding="utf-8")) if out_path.exists() else 0
    print(f"[{split}] total samples in file: {kept} -> {out_path}")
    print(f"  per-event counts (in this run): {written}")

def main():
    build_samples("train", cap_per_event=args.max_train_per_event)
    build_samples("val", cap_per_event=args.max_val_per_event)

if __name__ == "__main__":
    main()