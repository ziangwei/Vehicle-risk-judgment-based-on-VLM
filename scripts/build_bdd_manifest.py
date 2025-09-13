import json, sys
from pathlib import Path

ROOT = Path.cwd()  # 仓库根运行
IMG_TRAIN = ROOT/"data/raw/bdd100k/images/track/train"
IMG_VAL   = ROOT/"data/raw/bdd100k/images/track/val"
LBL_TRAIN = ROOT/"data/raw/bdd100k/labels/box_track_20/train"
LBL_VAL   = ROOT/"data/raw/bdd100k/labels/box_track_20/val"
OUT_DIR   = ROOT/"data/interim/manifests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build(split_img: Path, split_lbl: Path, split_name: str):
    # 1) 列出有图像的序列
    seq_dirs = [d for d in split_img.iterdir() if d.is_dir()]
    print(f"[{split_name}] image seq dirs:", len(seq_dirs))
    # 2) 建立标签映射
    lbl_map = {p.stem: p for p in split_lbl.glob("*.json")}
    print(f"[{split_name}] label json files:", len(lbl_map))

    missing_lbl, empty_seq, ok = 0, 0, 0
    out_path = OUT_DIR/f"manifest_{split_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for seq in sorted(seq_dirs):
            seq_id = seq.name
            frames = sorted(seq.glob("*.jpg"))
            if not frames:
                empty_seq += 1
                continue
            lbl = lbl_map.get(seq_id)
            if lbl is None:
                missing_lbl += 1
                continue
            rec = {
                "split": split_name,
                "seq_id": seq_id,
                "frames": [str(p) for p in frames],
                "label": str(lbl),
                "num_frames": len(frames)
            }
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
            ok += 1
    print(f"[{split_name}] OK: {ok}, empty_seq: {empty_seq}, missing_label: {missing_lbl}")
    print(f"[{split_name}] wrote -> {out_path}")

build(IMG_TRAIN, LBL_TRAIN, "train")
build(IMG_VAL,   LBL_VAL,   "val")