# =============================
# File: scripts/eval_from_conversations_h100.py
# Desc: Evaluation (JSON-Valid / Fields-OK / Fields-Strict / Action-Acc / mIoU / Hit@0.5 / Consistency)
# =============================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, csv, math, argparse
from pathlib import Path
from collections import defaultdict, Counter

REQ_KEYS = {"risk_level","hazards","reasoning","metrics","recommendation","evidence"}

# ----- utils -----

def iou(a,b):
    ax1,ay1,ax2,ay2 = map(float,a); bx1,by1,bx2,by2 = map(float,b)
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1); inter=iw*ih
    if inter<=0: return 0.0
    areaA=(ax2-ax1)*(ay2-ay1); areaB=(bx2-bx1)*(by2-by1)
    u=areaA+areaB-inter
    return inter/u if u>0 else 0.0

def safe_json(s):
    try:
        i,j=s.find("{"), s.rfind("}")
        if i==-1 or j==-1 or j<=i: return None
        return json.loads(s[i:j+1])
    except: return None

def first_action(obj):
    try:
        for h in obj.get("hazards",[]):
            a=h.get("action");
            if isinstance(a,str):
                a=a.lower().replace("cutin","cut_in").replace("hardbrake","hard_brake")
                return a
    except: pass
    return None

def parse_seq_and_idx(path):
    m=re.search(r'/images/track/[^/]+/([^/]+)/\1-(\d+)\.jpg$', path)
    if not m: return None, None
    return m.group(1), int(m.group(2))

# manifest: seq_id -> [abs frame paths]

def load_manifest(p):
    seq2frames={}
    with open(p,"r",encoding="utf-8") as f:
        for ln in f:
            rec=json.loads(ln); seq2frames[rec["seq_id"]]=rec["frames"]
    return seq2frames

# events 索引： seq -> { tuple(window) : event_rec }

def load_events_index(p):
    idx=defaultdict(dict)
    with open(p,"r",encoding="utf-8") as f:
        for ln in f:
            ev=json.loads(ln)
            idx[ev["seq_id"]][tuple(ev["window"])]=ev
    return idx

# 载 BDD100K 帧级标注（一次性缓存）

def load_labels_cache(path):
    data=json.load(open(path,"r"))
    return data["frames"] if isinstance(data,dict) else data

# 字段严格

def fields_strict_ok(obj):
    if not isinstance(obj, dict): return False
    if not REQ_KEYS.issubset(obj.keys()): return False
    ok=True
    ok &= isinstance(obj.get("hazards"), list) and len(obj["hazards"])>=1
    ok &= isinstance(obj.get("recommendation"), str) and len(obj["recommendation"].strip())>0
    ok &= isinstance(obj.get("metrics"), dict) and len(obj["metrics"])>=1
    ok &= isinstance(obj.get("reasoning"), list)
    return ok

# 一致性

def check_consistency(obj, thr_dx=0.12, thr_area=1.4, thr_cx=0.25, thr_cy=0.55):
    if not isinstance(obj, dict): return False, "not_dict"
    a = first_action(obj)
    m = obj.get("metrics", {}) if isinstance(obj.get("metrics"), dict) else {}
    def f(k):
        v=m.get(k);
        try: return float(v)
        except: return None
    dx,ar,cxm,cyl = f("dx_norm"), f("area_ratio"), f("cx_mid_norm"), f("cy_last_norm")
    if a=="cut_in":
        if dx is None or cyl is None: return False, "missing_metrics"
        cond=(abs(dx)>=thr_dx and cyl>=thr_cy)
        return cond, ("ok" if cond else "violate_cut_in")
    if a=="hard_brake":
        if ar is None or cxm is None or cyl is None: return False, "missing_metrics"
        cond=(ar>=thr_area and abs(cxm-0.5)<=thr_cx and cyl>=thr_cy)
        return cond, ("ok" if cond else "violate_hard_brake")
    return False, "unknown_action"

# IoU 对齐所需

def find_gt_box(frames, t0_based, obj_id):
    if t0_based<0 or t0_based>=len(frames): return None
    labs=frames[t0_based].get("labels",[])
    for lab in labs:
        lid=str(lab.get("id", lab.get("track_id", lab.get("index"))))
        if lid==str(obj_id):
            b=lab.get("box2d");
            return [b["x1"],b["y1"],b["x2"],b["y2"]] if b else None
    return None

def clamp_to_allowed(t, allowed, default_by_order_idx):
    if not allowed: return default_by_order_idx
    if isinstance(t,int) and t in set(allowed): return t
    if isinstance(t,int): return min(allowed, key=lambda x: abs(x-t))
    # 没有 t，则按顺序映射
    return allowed[min(default_by_order_idx, len(allowed)-1)]

# ----- main -----

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", help="H100推理脚本输出(jsonl)，含 prediction_json / images / frame_idx_list")
    ap.add_argument("--conv_jsonl", help="备选：直接用 conversations jsonl")
    ap.add_argument("--use_logged_assistant", action="store_true")
    ap.add_argument("--events", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--thr_dx", type=float, default=0.12)
    ap.add_argument("--thr_area", type=float, default=1.4)
    ap.add_argument("--thr_cx", type=float, default=0.25)
    ap.add_argument("--thr_cy", type=float, default=0.55)
    ap.add_argument("--out_dir", default="reports/eval_from_conversations")
    ap.add_argument("--confusion", action="store_true")
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    seq2frames=load_manifest(args.manifest)
    ev_index=load_events_index(args.events)
    label_cache={}

    rows=[]

    # ---- 主路径：评 H100 推理输出 ----
    if args.pred_jsonl:
        with open(args.pred_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                rec=json.loads(ln)
                images=rec.get("images",[])
                allowed = rec.get("frame_idx_list") or [parse_seq_and_idx(p)[1] for p in images if parse_seq_and_idx(p)[1] is not None]
                obj_pred = rec.get("prediction_json") or safe_json(rec.get("prediction_text",""))

                # seq & window from paths（0-based索引）
                seq_set=set(); idxs=[]
                for p in images:
                    s,i=parse_seq_and_idx(p)
                    if s is None: seq_set.add("__unknown__")
                    else: seq_set.add(s); idxs.append(i-1)
                seq = next(iter(seq_set)) if len(seq_set)==1 else "__unknown__"
                window0=tuple(sorted(idxs))
                ev = ev_index.get(seq,{}).get(window0)

                json_valid=int(isinstance(obj_pred, dict))
                fields_ok =int(json_valid and REQ_KEYS.issubset(set(obj_pred.keys())))
                strict_ok =int(fields_strict_ok(obj_pred))

                action_pred = first_action(obj_pred) if isinstance(obj_pred, dict) else None
                action_gt=None; action_correct=None; mean_iou=float("nan"); hit=float("nan")

                if ev is not None:
                    action_gt = ev["event"].lower()
                    action_correct = int((action_pred or "")==action_gt)
                    lab_path=ev["label"]; obj_id=ev["obj_id"]
                    if lab_path not in label_cache:
                        try: label_cache[lab_path]=load_labels_cache(lab_path)
                        except: label_cache[lab_path]=None
                    frames=label_cache[lab_path]
                    ious=[]
                    if frames and isinstance(obj_pred,dict):
                        evid=obj_pred.get("evidence",[])
                        for k, ebox in enumerate(evid):
                            box=ebox.get("box2d"); t=ebox.get("frame_idx", None)
                            if not (isinstance(box, list) and len(box)==4):
                                continue
                            t_eff = clamp_to_allowed(t, allowed, k)
                            gt = find_gt_box(frames, t_eff, obj_id)
                            if gt: ious.append(iou(box, gt))
                    if ious:
                        mean_iou=sum(ious)/len(ious); hit=sum(1 for v in ious if v>=0.5)/len(ious)

                cons, creason = check_consistency(obj_pred or {}, args.thr_dx,args.thr_area,args.thr_cx,args.thr_cy)

                rows.append({
                    "seq_id": seq,
                    "window": list(window0),
                    "json_valid": json_valid,
                    "fields_ok": fields_ok,
                    "fields_strict": strict_ok,
                    "pred_action": action_pred,
                    "gt_action": action_gt,
                    "action_correct": (action_correct if ev is not None else None),
                    "mean_iou": (0.0 if math.isnan(mean_iou) else mean_iou),
                    "hit@0.5": (0.0 if math.isnan(hit) else hit),
                    "consistency": int(cons),
                    "consistency_reason": creason,
                    "n_images": len(images),
                    "matched_gt": bool(ev is not None)
                })
    else:
        # 兼容：从 conv_jsonl 评（不建议，性能差）
        with open(args.conv_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                rec=json.loads(ln)
                images=rec.get("images",[])
                allowed = rec.get("frame_idx_list") or [parse_seq_and_idx(p)[1] for p in images if parse_seq_and_idx(p)[1] is not None]
                # 此分支仅统计 json_valid / fields / 行为；不现场推理（简化）
                obj_pred = safe_json(next((m["content"] for m in rec.get("conversations",[]) if m.get("role")=="assistant"), "")) if args.use_logged_assistant else None
                if obj_pred is None:
                    rows.append({"seq_id":"","window":[],"json_valid":0,"fields_ok":0,"fields_strict":0,
                                 "pred_action":None,"gt_action":None,"action_correct":None,"mean_iou":0.0,"hit@0.5":0.0,
                                 "consistency":0,"consistency_reason":"no_pred","n_images":len(images),"matched_gt":False})
                    continue
                seq_set=set(); idxs=[]
                for p in images:
                    s,i=parse_seq_and_idx(p)
                    if s is None: seq_set.add("__unknown__")
                    else: seq_set.add(s); idxs.append(i-1)
                seq = next(iter(seq_set)) if len(seq_set)==1 else "__unknown__"
                window0=tuple(sorted(idxs))
                ev = ev_index.get(seq,{}).get(window0)

                json_valid=int(isinstance(obj_pred, dict))
                fields_ok =int(json_valid and REQ_KEYS.issubset(set(obj_pred.keys())))
                strict_ok =int(fields_strict_ok(obj_pred))

                action_pred=first_action(obj_pred); action_gt=None; action_correct=None
                if ev is not None:
                    action_gt=ev["event"].lower(); action_correct=int((action_pred or "")==action_gt)
                cons, creason = check_consistency(obj_pred or {}, args.thr_dx,args.thr_area,args.thr_cx,args.thr_cy)

                rows.append({
                    "seq_id": seq,
                    "window": list(window0),
                    "json_valid": json_valid,
                    "fields_ok": fields_ok,
                    "fields_strict": strict_ok,
                    "pred_action": action_pred,
                    "gt_action": action_gt,
                    "action_correct": (action_correct if ev is not None else None),
                    "mean_iou": 0.0,
                    "hit@0.5": 0.0,
                    "consistency": int(cons),
                    "consistency_reason": creason,
                    "n_images": len(images),
                    "matched_gt": bool(ev is not None)
                })

    # ---- 导出 ----
    out_csv = out/"conv_eval.csv"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    def avg(key, filt=None):
        vs=[r[key] for r in rows if (filt(r) if filt else True) and r[key] is not None]
        return (sum(vs)/len(vs)) if vs else 0.0

    matched=[r for r in rows if r["matched_gt"]]
    cm=Counter()
    for r in matched:
        if r["pred_action"] and r["gt_action"]:
            cm[(r["gt_action"], r["pred_action"])]+=1

    summary={
        "n_total": len(rows),
        "n_matched_gt": len(matched),
        "overall": {
            "json_valid": avg("json_valid"),
            "fields_ok": avg("fields_ok"),
            "fields_strict": avg("fields_strict"),
            "action_acc_on_matched": (sum(r["action_correct"] for r in matched)/len(matched) if matched else 0.0),
            "mean_iou_on_matched": avg("mean_iou", lambda r: r["matched_gt"]),
            "hit@0.5_on_matched":  avg("hit@0.5",  lambda r: r["matched_gt"]),
            "consistency_on_matched": avg("consistency", lambda r: r["matched_gt"])
        }
    }
    with open(out/"summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.confusion and cm:
        cm_csv = out/"action_confusion.csv"
        gts=sorted(set(g for g,_ in cm.keys())); prs=sorted(set(p for _,p in cm.keys()))
        with open(cm_csv, "w", newline="", encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["gt/pred"]+prs)
            for g in gts:
                row=[g]+[cm.get((g,p),0) for p in prs]
                w.writerow(row)

    print("[eval_from_conversations_h100] wrote:", out_csv)

if __name__=="__main__":
    main()

