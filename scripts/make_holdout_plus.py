
# =============================
# File: scripts/make_holdout_plus.py
# Desc: Split holdout by seq_id + build conv_eval_* with frame_idx_list
# =============================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, random, re
from collections import defaultdict, Counter
from pathlib import Path
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("--events_val", default="data/interim/events/events_val.jsonl")
ap.add_argument("--manifest",  default="data/interim/manifests/manifest_val.jsonl")
ap.add_argument("--ratio", type=float, default=0.3)
ap.add_argument("--seed",  type=int, default=2024)
ap.add_argument("--prompt", default=(
    "仅依据按时间顺序给出的多帧行车画面，输出驾驶风险评估 JSON。必须包含字段: "
    "risk_level, hazards[], reasoning[], metrics{}, recommendation, evidence[]。不得编造未在画面中出现的信息。"
))
args=ap.parse_args()
random.seed(args.seed)

EVAL_OUT = Path("data/processed"); EVAL_OUT.mkdir(parents=True, exist_ok=True)
INTERIM  = Path("data/interim");   INTERIM.mkdir(parents=True, exist_ok=True)

# manifest: seq -> [abs frames]
seq2frames={}
with open(args.manifest, "r", encoding="utf-8") as f:
    for ln in f:
        rec=json.loads(ln); seq2frames[rec["seq_id"]]=rec["frames"]

seq2events=defaultdict(list)
with open(args.events_val, "r", encoding="utf-8") as f:
    for ln in f:
        ev=json.loads(ln); seq2events[ev["seq_id"]].append(ev)

seqs=list(seq2events.keys()); random.shuffle(seqs)
k=max(1, int(len(seqs)*args.ratio))
holdout_seqs=set(seqs[:k]); dev_seqs=set(seqs[k:])

holdout_seq_txt = INTERIM/"holdout_seq.txt"
holdout_events  = INTERIM/"events_val_holdout.jsonl"
dev_events      = INTERIM/"events_val_dev.jsonl"

with open(holdout_seq_txt, "w", encoding="utf-8") as f:
    for s in sorted(holdout_seqs): f.write(s+"\n")
with open(holdout_events, "w", encoding="utf-8") as fo, open(dev_events, "w", encoding="utf-8") as fd:
    for s in seqs:
        dst=fo if s in holdout_seqs else fd
        for ev in seq2events[s]: dst.write(json.dumps(ev, ensure_ascii=False)+"\n")

# conv_eval_*

def parse_num(p):
    m=re.search(r"/([^/]+)-(\d+)\.jpg$", p); return int(m.group(2)) if m else None

def ev_to_conv(ev_list):
    recs=[]
    for ev in ev_list:
        seq=ev["seq_id"]; window=ev["window"]  # 0-based indices
        frames=seq2frames.get(seq);
        if not frames: continue
        try:
            imgs=[frames[i] for i in window]
        except Exception:
            continue
        idx_list=[parse_num(p) for p in imgs]
        user=("<image>\n"*len(imgs))+args.prompt
        recs.append({"images": imgs, "frame_idx_list": idx_list, "conversations": [{"role":"user","content": user}]})
    return recs

holdout_convs=ev_to_conv([ev for s in holdout_seqs for ev in seq2events[s]])
dev_convs    =ev_to_conv([ev for s in dev_seqs     for ev in seq2events[s]])

conv_hold = EVAL_OUT/"conv_eval_holdout.jsonl"
conv_dev  = EVAL_OUT/"conv_eval_dev.jsonl"
with open(conv_hold, "w", encoding="utf-8") as f:
    for r in holdout_convs: f.write(json.dumps(r, ensure_ascii=False)+"\n")
with open(conv_dev, "w", encoding="utf-8") as f:
    for r in dev_convs: f.write(json.dumps(r, ensure_ascii=False)+"\n")

# stats

def count_by_event(p):
    c=Counter();
    with open(p,"r",encoding="utf-8") as f:
        for ln in f:
            c[json.loads(ln)["event"]]+=1
    return dict(c)

n_hold=sum(1 for _ in open(holdout_events, "r", encoding="utf-8"))
n_dev =sum(1 for _ in open(dev_events, "r", encoding="utf-8"))
print("[make_holdout_plus] total seq:", len(seqs))
print("[make_holdout_plus] holdout seq:", len(holdout_seqs), " events:", n_hold, count_by_event(holdout_events))
print("[make_holdout_plus] dev     seq:", len(dev_seqs), " events:", n_dev,  count_by_event(dev_events))
print("[make_holdout_plus] outputs:\n  -", holdout_seq_txt, "\n  -", holdout_events, "\n  -", dev_events, "\n  -", conv_hold, "\n  -", conv_dev)
