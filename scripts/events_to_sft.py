import json, argparse, random, sys
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

# ---------------- Args ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=5, help="frames per sample (odd number suggested)")
ap.add_argument("--min_gap", type=int, default=5,
                help="dedup: min distance between accepted centers for same seq+obj+event")
ap.add_argument("--max_train_per_event", type=int, default=10000, help="cap per event type for train")
ap.add_argument("--max_val_per_event", type=int, default=1200, help="cap per event type for val")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed)

ROOT = Path(__file__).resolve().parent.parent
MAN_DIR = ROOT / "data/interim/manifests"
EVT_DIR = ROOT / "data/interim/events"
OUT_DIR = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper Functions ---

def load_manifest(split):
    m = {}
    with open(MAN_DIR / f"manifest_{split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            m[rec["seq_id"]] = rec["frames"]
    return m


def read_label_frames(label_path: Path):
    try:
        data = json.load(open(label_path))
        if isinstance(data, list): return data
        if isinstance(data, dict): return data.get("frames") or []
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    return []


def find_label_for_obj(frame_rec, obj_id):
    labels = frame_rec.get("labels") if isinstance(frame_rec, dict) else None
    if not labels:
        return None, None
    for lab in labels:
        lid = lab.get("id", lab.get("track_id", lab.get("index")))
        if str(lid) == str(obj_id):
            category = lab.get("category", "unknown")
            b2d = lab.get("box2d")
            if isinstance(b2d, dict) and {"x1", "y1", "x2", "y2"} <= b2d.keys():
                box = [float(b2d["x1"]), float(b2d["y1"]), float(b2d["x2"]), float(b2d["y2"])]
                return box, category
    return None, None


def risk_from_metrics(ev):
    dx = abs(ev["metrics"].get("dx_norm", 0.0))
    ar = ev["metrics"].get("area_ratio", 1.0)
    if dx >= 0.18 or ar >= 1.8: return "high", ["lateral_shift_fast" if dx >= 0.18 else "rapid_approach"]
    if dx >= 0.12 or ar >= 1.4: return "medium", ["noticeable_shift" if dx >= 0.12 else "distance_closing"]
    return "low", ["mild_change"]


def recommendation_from_event(ev_type, risk, category):
    # 我们可以为不同类别的目标提供不同的建议
    if category == "person" and risk != "low":
        return "yield_to_pedestrian_and_prepare_to_brake"
    if ev_type == "cut_in":
        return "prepare_brake_and_keep_gap"
    if ev_type == "hard_brake":
        return "brake_then_increase_following_distance"
    return "continue_driving"


def draw_bbox_on_image(original_path: str, box: list, save_path: str, color="red", width=3):
    try:
        img = Image.open(original_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle(box, outline=color, width=width)
        save_path_p = Path(save_path)
        save_path_p.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Error drawing bbox on {original_path}: {e}", file=sys.stderr)
        return False


def build_samples(split, cap_per_event):
    seq2frames = load_manifest(split)
    events_path = EVT_DIR / f"events_{split}.jsonl"
    out_path = OUT_DIR / f"bdd_multiframe_sft_{split}.jsonl"

    last_pick = {}
    written = {}  # 初始化为空字典，可以动态接受任何事件类型

    PROCESSED_IMG_DIR = ROOT / f"data/processed/images_with_bbox_{split}"
    PROCESSED_IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_events = []
    if events_path.exists():
        with open(events_path, "r", encoding="utf-8") as f_in:
            all_events = [json.loads(line) for line in f_in]

    with open(out_path, "w", encoding="utf-8") as f_out:
        for ev in tqdm(all_events, desc=f"Processing {split}"):
            ev_type = ev["event"]

            # 使用 .get(ev_type, 10000) 来处理未预设上限的事件类型
            if written.get(ev_type, 0) >= cap_per_event.get(ev_type, 10000):
                continue

            seq_id = ev["seq_id"]
            obj_id = ev["obj_id"]
            frames_all = seq2frames.get(seq_id)
            if not frames_all: continue

            # ... (去重和窗口选择逻辑不变)
            center = int(ev["center_idx"])
            key = (seq_id, str(obj_id), ev_type)
            last = last_pick.get(key, -10 ** 9)
            if abs(center - last) < args.min_gap: continue
            win = ev["window"]
            sel = win
            if args.frames < len(win):
                stride = len(win) // args.frames if args.frames > 0 else 1
                sel = win[::stride][:args.frames]

            labels = read_label_frames(Path(ev["label"]))
            imgs, evidence_boxes, categories = [], [], []
            ok = True
            for t in sel:
                if t >= len(frames_all) or t >= len(labels):
                    ok = False;
                    break

                box, category = find_label_for_obj(labels[t], obj_id)
                if box is None or category is None:
                    ok = False;
                    break

                imgs.append(frames_all[t])
                evidence_boxes.append({"frame_idx": int(t), "box2d": [round(v, 2) for v in box]})
                categories.append(category)

            if not ok or len(imgs) < 2: continue

            first_category = categories[0]
            if not all(c == first_category for c in categories):
                continue

            imgs_processed = []
            drawing_ok = True
            for i, original_path_str in enumerate(imgs):
                original_path = Path(original_path_str)
                box_to_draw = evidence_boxes[i]["box2d"]
                new_img_name = f"{original_path.stem}_obj_{obj_id}.jpg"
                save_path = PROCESSED_IMG_DIR / seq_id / new_img_name
                if not draw_bbox_on_image(original_path_str, box_to_draw, str(save_path)):
                    drawing_ok = False;
                    break
                imgs_processed.append(str(save_path))
            if not drawing_ok: continue

            if ev_type == "normal_driving":
                risk, reasons, recommendation, hazards = "low", ["stable_movement"], "continue_driving", []
            else:
                risk, reasons = risk_from_metrics(ev)
                recommendation = recommendation_from_event(ev_type, risk, first_category)
                hazards = [{"obj_id": str(obj_id), "type": first_category, "action": ev_type}]

            resp = {
                "risk_level": risk, "hazards": hazards, "reasoning": reasons,
                "metrics": {k: ev["metrics"].get(k) for k in ["dx_norm", "area_ratio", "cx_mid_norm", "cy_last_norm"]},
                "recommendation": recommendation, "evidence": evidence_boxes
            }

            image_placeholders = "".join(["<image>\n"] * len(imgs_processed))
            prompt_text = (
                "请根据提供的时间连续帧图像，对图中高亮框出的目标行为进行分析，并输出驾驶风险评估JSON。"
                "必须包含字段: risk_level, hazards[], reasoning[], metrics{}, recommendation, evidence[]。"
                "在`evidence`字段中，`box2d`的值是一个包含四个浮点数的数组，分别代表边界框左上角顶点的x和y坐标，以及右下角顶点的x和y坐标 (x1, y1, x2, y2)。"
                "不得编造未在画面中出现的信息。"
            )
            sample = {
                "images": imgs_processed,
                "prompt": image_placeholders + prompt_text,
                "response": json.dumps(resp, ensure_ascii=False)
            }

            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written[ev_type] = written.get(ev_type, 0) + 1
            last_pick[key] = center

    kept = sum(1 for _ in open(out_path, "r", encoding="utf-8")) if out_path.exists() else 0
    print(f"[{split}] total samples in file: {kept} -> {out_path}")
    print(f"  per-event counts: {written}")


def main():
    train_caps = {"cut_in": 10000, "hard_brake": 10000, "normal_driving": 10000}
    val_caps = {"cut_in": 1200, "hard_brake": 1200, "normal_driving": 1200}
    build_samples("train", cap_per_event=train_caps)
    build_samples("val", cap_per_event=val_caps)


if __name__ == "__main__":
    main()