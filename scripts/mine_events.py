import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import random
import sys

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=5, help="frames per window")
parser.add_argument("--stride", type=int, default=1, help="sliding step")
parser.add_argument("--cutin_dx", type=float, default=0.12, help="abs lateral shift (normalized by width)")
parser.add_argument("--brake_area", type=float, default=1.4, help="area growth ratio")
parser.add_argument("--center_tol", type=float, default=0.25, help="|cx_norm-0.5| <= center_tol")
parser.add_argument("--front_y", type=float, default=0.55, help="cy_norm >= front_y")
# 新增：负采样率，例如 0.1 表示对 10% 的平稳窗口进行采样
parser.add_argument("--stable_sample_rate", type=float, default=0.1, help="sampling rate for stable driving windows")
args = parser.parse_args()

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent.parent
MAN_DIR = ROOT / "data/interim/manifests"
OUT_DIR = ROOT / "data/interim/events"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def to_xywh_from_label(lab):
    b2d = lab.get("box2d")
    if isinstance(b2d, dict) and {"x1", "y1", "x2", "y2"} <= b2d.keys():
        x1, y1, x2, y2 = float(b2d["x1"]), float(b2d["y1"]), float(b2d["x2"]), float(b2d["y2"])
        return x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)
    bb = lab.get("bbox") or lab.get("box") or None
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        return float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    return None

def cxcy_area(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0, max(1.0, w * h)


def load_tracks(label_path: Path):
    try:
        data = json.load(open(label_path))
    except (json.JSONDecodeError, FileNotFoundError):
        return {}, 0

    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict):
        frames = data.get("frames") or data.get("video", {}).get("frames") or []
    else:
        frames = []

    tracks = {}
    for fi, fr in enumerate(frames):
        labels = fr.get("labels") if isinstance(fr, dict) else None
        if not labels: continue
        for lab in labels:
            obj_id = lab.get("id", lab.get("track_id", lab.get("index")))
            if obj_id is None: continue
            box = to_xywh_from_label(lab)
            if box is None: continue
            tracks.setdefault(obj_id, {})[fi] = box
    return tracks, len(frames)


def mine_seq(rec, out_f):
    frames = rec["frames"]
    label_path = Path(rec["label"])
    try:
        W, H = Image.open(frames[0]).size
    except Exception:
        W, H = 1280, 720

    tracks, n_frames = load_tracks(label_path)
    win = args.window
    stride = args.stride
    for obj_id, timeline in tracks.items():
        if len(timeline) < 2: continue
        i = 0
        while i + win - 1 < n_frames:
            t0 = i
            t1 = i + win - 1
            if t0 in timeline and t1 in timeline:
                b0 = timeline[t0];
                b1 = timeline[t1]
                cx0, cy0, a0 = cxcy_area(b0);
                cx1, cy1, a1 = cxcy_area(b1)
                dx_norm = (cx1 - cx0) / float(W)
                area_ratio = a1 / a0
                cx_mid_norm = ((cx0 + cx1) / 2.0) / W
                cy_last_norm = cy1 / H
                front = cy_last_norm >= args.front_y
                center_ok = abs(cx_mid_norm - 0.5) <= args.center_tol

                is_cutin = abs(dx_norm) >= args.cutin_dx and front
                is_brake = (area_ratio >= args.brake_area) and front and center_ok

                event_to_write = None

                if is_cutin or is_brake:
                    # 这是一个风险事件，直接记录
                    event_to_write = "cut_in" if is_cutin else "hard_brake"
                else:
                    # 这是一个非风险窗口，我们判断它是否“平稳”
                    is_stable = abs(dx_norm) < 0.05 and abs(area_ratio - 1.0) < 0.1
                    # 如果平稳，我们就按一定概率进行采样，作为负样本
                    if is_stable and random.random() < args.stable_sample_rate:
                        event_to_write = "normal_driving"

                if event_to_write:
                    rec_out = {
                        "split": rec["split"],
                        "seq_id": rec["seq_id"],
                        "center_idx": int((t0 + t1) // 2),
                        "window": list(range(t0, t1 + 1)),
                        "label": rec["label"],
                        "event": event_to_write,
                        "obj_id": obj_id,
                        "metrics": {
                            "dx_norm": round(float(dx_norm), 4),
                            "abs_dx_norm": round(abs(float(dx_norm)), 4),
                            "area_ratio": round(float(area_ratio), 4),
                            "cx_mid_norm": round(float(cx_mid_norm), 4),
                            "cy_last_norm": round(float(cy_last_norm), 4),
                            "W": W, "H": H
                        }
                    }
                    out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

            i += stride


def process_split(split):
    mani = MAN_DIR / f"manifest_{split}.jsonl"
    out_p = OUT_DIR / f"events_{split}.jsonl"
    n_seq = 0
    # 获取总行数用于tqdm
    try:
        total_lines = sum(1 for _ in open(mani, "r", encoding="utf-8"))
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {mani}", file=sys.stderr)
        return

    with open(mani, "r", encoding="utf-8") as f_in, open(out_p, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, total=total_lines, desc=f"Mining events for {split}"):
            rec = json.loads(line)
            n_seq += 1
            mine_seq(rec, f_out)

    n_evt = sum(1 for _ in open(out_p, "r", encoding="utf-8"))
    print(f"[{split}] sequences scanned: {n_seq} ; events found: {n_evt}")
    print(f"[{split}] wrote -> {out_p}")


def main():
    for sp in ["train", "val"]:
        process_split(sp)


if __name__ == "__main__":
    main()