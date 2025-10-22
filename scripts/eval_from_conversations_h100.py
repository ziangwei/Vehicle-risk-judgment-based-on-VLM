#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, csv, math, argparse
from pathlib import Path
from collections import defaultdict, Counter

# --- 核心工具函数 ---

def iou(a, b):
    # (IoU计算函数保持不变)
    ax1, ay1, ax2, ay2 = map(float, a);
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1);
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1);
    inter = iw * ih
    if inter <= 0: return 0.0
    areaA = (ax2 - ax1) * (ay2 - ay1);
    areaB = (bx2 - bx1) * (by2 - by1)
    u = areaA + areaB - inter
    return inter / u if u > 0 else 0.0


def safe_json(s):
    # (安全解析JSON的函数保持不变)
    try:
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j == -1 or j <= i: return None
        return json.loads(s[i:j + 1])
    except:
        return None

def get_first_action(obj):
    """更宽容地提取第一个action"""
    try:
        # 将整个预测的JSON转为小写字符串，用于包含匹配
        pred_text = json.dumps(obj).lower()

        # --- cut_in 的判断 ---
        # 增加对 "cut in" (带空格) 的判断
        if "cut_in" in pred_text or "cutin" in pred_text or "cut in" in pred_text:
            return "cut_in"

        # --- hard_brake 的判断 ---
        # 增加对 "hard brake" (带空格) 的判断
        if "hard_brake" in pred_text or "hardbrake" in pred_text or "hard brake" in pred_text:
            return "hard_brake"

    except:
        pass
    return None

def get_risk_level(obj):
    """提取risk_level"""
    try:
        return obj.get("risk_level", "").lower()
    except:
        return None


def parse_seq_and_idx(path):
    # (解析路径的函数保持不变)
    m = re.search(r'/images/track/[^/]+/([^/]+)/\1-(\d+)\.jpg$', path)
    if not m: return None, None
    return m.group(1), int(m.group(2))


def load_events_index(p):
    # (加载事件索引的函数保持不变)
    idx = defaultdict(dict)
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            ev = json.loads(ln)
            idx[ev["seq_id"]][tuple(ev["window"])] = ev
    return idx


def find_gt_box_robust(bdd_frames_list, frame_idx_pred, obj_id_gt):
    """
    更鲁棒的GT Box查找函数
    - bdd_frames_list: 从BDD100K原始标签JSON读出的帧列表
    - frame_idx_pred: 模型预测的帧序号 (e.g., 110)
    - obj_id_gt: GT事件中的物体ID
    """
    # 遍历BDD标签中的所有帧，寻找与预测的帧序号匹配的帧
    for frame_data in bdd_frames_list:
        # BDD100K的帧名通常是 '序列号-帧序号.jpg'
        frame_name = frame_data.get("name", "")
        if f"-{frame_idx_pred:07d}" in frame_name:  # 格式化为7位数字进行匹配, e.g., -0000110
            # 在找到的帧里，再根据obj_id查找物体
            for label in frame_data.get("labels", []):
                # 强制转换为字符串进行比较，避免类型问题
                label_id = str(label.get("id", label.get("track_id")))
                if label_id == str(obj_id_gt):
                    b = label.get("box2d")
                    if b:
                        return [b["x1"], b["y1"], b["x2"], b["y2"]]
    return None  # 如果找不到，返回None


def main():
    # ... (argparse部分保持不变)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out_dir", default="reports/final_eval")
    ap.add_argument("--confusion", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir);
    out.mkdir(parents=True, exist_ok=True)

    ev_index = load_events_index(args.events)
    label_cache = {}  # 缓存加载过的BDD100K标签文件
    rows = []

    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            rec = json.loads(ln)
            obj_pred = rec.get("prediction_json")
            if not isinstance(obj_pred, dict):
                continue

            # 从图片路径中解析出 seq_id 和 window
            images = rec.get("images", [])
            seq_set = set();
            idxs = []
            for p in images:
                s, i = parse_seq_and_idx(p)
                if s: seq_set.add(s)
                if i: idxs.append(i - 1)  # 0-based index

            if len(seq_set) != 1: continue  # 跳过序列ID不唯一的异常数据
            seq = seq_set.pop()
            window0 = tuple(sorted(idxs))

            # 查找对应的GT事件
            ev = ev_index.get(seq, {}).get(window0)
            if not ev:
                continue

            # --- 开始评估 ---
            action_pred = get_first_action(obj_pred)
            action_gt = ev.get("event", "").lower()
            action_correct = 1 if action_gt and action_pred and action_gt in action_pred else 0

            risk_pred = get_risk_level(obj_pred)
            risk_gt = ev.get("risk_level", "").lower()  # 假设events文件里有risk_level
            risk_correct = 1 if risk_gt and risk_pred and risk_gt == risk_pred else 0

            # --- IoU 计算 ---
            mean_iou = 0.0
            hit_at_0_5 = 0.0
            ious = []

            lab_path = ev.get("label")
            obj_id_gt = ev.get("obj_id")

            if lab_path and obj_id_gt:
                # 加载并缓存BDD标签文件
                if lab_path not in label_cache:
                    try:
                        label_data = json.load(open(lab_path, "r"))
                        label_cache[lab_path] = label_data.get("frames", [])
                    except Exception:
                        label_cache[lab_path] = None

                bdd_frames_list = label_cache[lab_path]
                if bdd_frames_list:
                    for pred_evidence in obj_pred.get("evidence", []):
                        pred_box = pred_evidence.get("box2d")
                        pred_frame_idx = pred_evidence.get("frame_idx")

                        if not (pred_box and isinstance(pred_frame_idx, int)):
                            continue

                        # 使用新的鲁棒查找函数
                        gt_box = find_gt_box_robust(bdd_frames_list, pred_frame_idx, obj_id_gt)

                        if gt_box:
                            ious.append(iou(pred_box, gt_box))

            if ious:
                mean_iou = sum(ious) / len(ious)
                hit_at_0_5 = sum(1 for v in ious if v >= 0.5) / len(ious)

            rows.append({
                "seq_id": seq,
                "pred_action": action_pred,
                "gt_action": action_gt,
                "action_correct": action_correct,
                "pred_risk": risk_pred,
                "gt_risk": risk_gt,
                "risk_correct": risk_correct,
                "mean_iou": mean_iou,
                "hit@0.5": hit_at_0_5,
            })

    # --- 汇总和保存结果 (这部分逻辑保持不变) ---
    out_csv = out / "evaluation_details.csv"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader();
            writer.writerows(rows)

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "total_samples_evaluated": len(rows),
        "action_accuracy": avg("action_correct"),
        "risk_accuracy": avg("risk_correct"),
        "mean_iou": avg("mean_iou"),
        "hit@0.5": avg("hit@0.5"),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.confusion:
        # ... (混淆矩阵逻辑保持不变)
        cm = Counter()
        for r in rows:
            if r["gt_action"] and r["pred_action"]:
                cm[(r["gt_action"], r["pred_action"])] += 1
        if cm:
            cm_csv = out / "action_confusion.csv"
            gts = sorted(set(g for g, _ in cm.keys()));
            prs = sorted(set(p for _, p in cm.keys()))
            with open(cm_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f);
                w.writerow(["gt/pred"] + prs)
                for g in gts:
                    row = [g] + [cm.get((g, p), 0) for p in prs]
                    w.writerow(row)

    print(f"[Evaluation Complete] Reports saved to: {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()