import json, argparse, random
from pathlib import Path

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

def build_samples(split, cap_per_event):
    seq2frames = load_manifest(split)
    events_path = EVT_DIR / f"events_{split}.jsonl"
    out_path = OUT_DIR / f"bdd_multiframe_sft_{split}.jsonl"

    # 去重记忆：同 (seq,obj,event) 最近一次被采纳的 center_idx
    last_pick = {}
    # 计数：每类的已写条数
    written = {"cut_in": 0, "hard_brake": 0}
    kept = 0

    with open(events_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            ev = json.loads(line)
            ev_type = ev["event"]
            if written.get(ev_type, 0) >= cap_per_event:
                continue

            seq_id = ev["seq_id"]
            obj_id = ev["obj_id"]
            frames_all = seq2frames.get(seq_id)
            if not frames_all:
                continue

            center = int(ev["center_idx"])
            key = (seq_id, str(obj_id), ev_type)
            last = last_pick.get(key, -10**9)
            if abs(center - last) < args.min_gap:
                continue  # 相邻窗口去重
            # 选择 frames 个索引（尽量居中）
            win = ev["window"]
            if args.frames < len(win):
                mid = len(win)//2
                half = args.frames//2
                sel = win[max(0, mid-half): max(0, mid-half)+args.frames]
            else:
                sel = win
            # 收集图像路径与证据（该 obj 在每帧的 box2d）
            labels = read_label_frames(Path(ev["label"]))
            imgs, evidence_boxes = [], []
            ok = True
            for t in sel:
                if t >= len(frames_all) or t >= len(labels):
                    ok = False; break
                box = find_box_for_obj(labels[t], obj_id)
                if box is None:
                    ok = False; break
                imgs.append(frames_all[t])
                evidence_boxes.append({"frame_idx": int(t), "box2d": [round(v,2) for v in box]})
            if not ok or len(imgs) < 2:
                continue

            risk, reasons = risk_from_metrics(ev)
            resp = {
                "risk_level": risk,
                "hazards": [{"obj_id": str(obj_id), "type": "vehicle", "action": ev_type}],
                "reasoning": reasons,
                "metrics": {
                    "dx_norm": ev["metrics"].get("dx_norm"),
                    "area_ratio": ev["metrics"].get("area_ratio"),
                    "cx_mid_norm": ev["metrics"].get("cx_mid_norm"),
                    "cy_last_norm": ev["metrics"].get("cy_last_norm")
                },
                "recommendation": recommendation_from_event(ev_type, risk),
                "evidence": evidence_boxes
            }
            sample = {
                "images": imgs,
                "prompt": ("仅依据按时间顺序给出的多帧行车画面，输出驾驶风险评估JSON。"
                           "必须包含字段: risk_level, hazards[], reasoning[], metrics{}, recommendation, evidence[]."
                           "不得编造未在画面中出现的信息。"),
                "response": json.dumps(resp, ensure_ascii=False)
            }
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written[ev_type] = written.get(ev_type, 0) + 1
            last_pick[key] = center
            kept += 1

    print(f"[{split}] wrote {kept} samples -> {out_path}")
    print(f"  per-event counts: {written}")

def main():
    build_samples("train", cap_per_event=args.max_train_per_event)
    build_samples("val",   cap_per_event=args.max_val_per_event)

if __name__ == "__main__":
    main()
