import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import random
import sys

# ---------------- Args ----------------
parser = argparse.ArgumentParser(description="Mine risk events from BDD100K tracks.")
parser.add_argument("--window", type=int, default=15, help="Frames per window for event detection.")
parser.add_argument("--stride", type=int, default=2, help="Sliding step for the window.")
parser.add_argument("--cutin_dx", type=float, default=0.12, help="Threshold for lateral shift (cut-in).")
parser.add_argument("--brake_area", type=float, default=1.4, help="Threshold for area growth (hard-brake).")
parser.add_argument("--center_tol", type=float, default=0.25, help="Tolerance for being in the center.")
parser.add_argument("--front_y", type=float, default=0.55, help="Y-coordinate threshold for being in front.")
parser.add_argument("--stable_sample_rate", type=float, default=0.05,
                    help="Sampling rate for stable (easy negative) windows.")
parser.add_argument("--hard_negative_rate", type=float, default=0.2, help="Sampling rate for hard negative windows.")
parser.add_argument("--width", type=int, default=1280, help="Video width for normalization.")
parser.add_argument("--height", type=int, default=720, help="Video height for normalization.")
args = parser.parse_args()

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent.parent
MAN_DIR = ROOT / "data/interim/manifests"
OUT_DIR = ROOT / "data/interim/events"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper Functions (Corrected and Robust) ---

def to_xywh_from_label(lab):
    b2d = lab.get("box2d")
    if isinstance(b2d, dict) and all(k in b2d for k in ["x1", "y1", "x2", "y2"]):
        x1, y1, x2, y2 = float(b2d["x1"]), float(b2d["y1"]), float(b2d["x2"]), float(b2d["y2"])
        return x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)
    return None


def cxcy_area(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0, max(1.0, w * h)


def load_tracks(label_path: Path):
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Skipping corrupted or missing label file {label_path}: {e}", file=sys.stderr)
        return {}, 0

    # ==========================================================
    #  ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 核心修复：兼容两种JSON格式 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # ==========================================================
    if isinstance(data, dict):
        frames = data.get("frames", [])
    elif isinstance(data, list):
        frames = data
    else:
        frames = []
    # ==========================================================
    #  ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 核心修复：兼容两种JSON格式 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # ==========================================================

    tracks = {}
    for fi, fr in enumerate(frames):
        for lab in fr.get("labels", []):
            obj_id = lab.get("id", lab.get("track_id"))
            box = to_xywh_from_label(lab)
            if obj_id is not None and box is not None:
                tracks.setdefault(str(obj_id), {})[fi] = box
    return tracks, len(frames)


# --- Core Mining Logic (Optimized) ---

def mine_seq(rec, out_f):
    W, H = args.width, args.height
    label_path = Path(rec["label"])
    tracks, n_frames = load_tracks(label_path)

    for obj_id, timeline in tracks.items():
        if len(timeline) < args.window:
            continue

        i = 0
        while i + args.window - 1 < n_frames:
            t0, t1 = i, i + args.window - 1
            if t0 in timeline and t1 in timeline:
                b0, b1 = timeline[t0], timeline[t1]
                cx0, _, a0 = cxcy_area(b0)
                cx1, cy1, a1 = cxcy_area(b1)

                dx_norm = (cx1 - cx0) / float(W)
                area_ratio = a1 / a0 if a0 > 0 else 1.0
                cx_mid_norm = ((cx0 + cx1) / 2.0) / W
                cy_last_norm = cy1 / H

                front = cy_last_norm >= args.front_y
                center_ok = abs(cx_mid_norm - 0.5) <= args.center_tol

                is_cutin = abs(dx_norm) >= args.cutin_dx and front
                is_brake = area_ratio >= args.brake_area and front and center_ok

                event_to_write = None

                if is_cutin or is_brake:
                    event_to_write = "cut_in" if is_cutin else "hard_brake"
                else:
                    is_hard_negative = (abs(dx_norm) >= args.cutin_dx * 0.7 and front) or \
                                       (area_ratio >= args.brake_area * 0.8 and front and center_ok)
                    if is_hard_negative and random.random() < args.hard_negative_rate:
                        event_to_write = "normal_driving"
                    else:
                        is_stable = abs(dx_norm) < 0.05 and abs(area_ratio - 1.0) < 0.1
                        if is_stable and random.random() < args.stable_sample_rate:
                            event_to_write = "normal_driving"

                if event_to_write:
                    rec_out = {
                        "split": rec["split"], "seq_id": rec["seq_id"],
                        "center_idx": (t0 + t1) // 2, "window": list(range(t0, t1 + 1)),
                        "label": rec["label"], "event": event_to_write, "obj_id": obj_id,
                        "metrics": {
                            "dx_norm": round(dx_norm, 4), "area_ratio": round(area_ratio, 4),
                            "cx_mid_norm": round(cx_mid_norm, 4), "cy_last_norm": round(cy_last_norm, 4)
                        }
                    }
                    out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            i += args.stride


def process_split(split):
    mani = MAN_DIR / f"manifest_{split}.jsonl"
    out_p = OUT_DIR / f"events_{split}.jsonl"

    try:
        total_lines = sum(1 for _ in open(mani, "r", encoding="utf-8"))
    except FileNotFoundError:
        print(f"Error: Manifest file '{mani}' not found. Please run 'build_bdd_manifest.py' first.", file=sys.stderr)
        return

    with open(mani, "r", encoding="utf-8") as f_in, open(out_p, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, total=total_lines, desc=f"Mining events for '{split}'"):
            mine_seq(json.loads(line), f_out)

    n_evt = sum(1 for _ in open(out_p, "r", encoding="utf-8"))
    print(f"[{split}] Scanned {total_lines} sequences, found {n_evt} events -> {out_p}")


def main():
    for sp in ["train", "val"]:
        process_split(sp)


if __name__ == "__main__":
    main()