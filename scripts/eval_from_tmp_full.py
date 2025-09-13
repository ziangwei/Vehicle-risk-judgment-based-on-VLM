#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Eval from TMP (完整评测版)
-------------------------
读取 .jsonl.tmp 或 .jsonl 文件，运行完整评测逻辑，输出结果报告。
支持：
- JSON-Valid / Fields-OK
- Action 准确率
- Risk 准确率
- IoU / Hit@0.5
- Consistency
- 混淆矩阵
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import confusion_matrix

# -------------------------
# 工具函数
# -------------------------

def load_jsonl_safe(path):
    """容错读取 jsonl/jsonl.tmp 文件"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, ln in enumerate(f):
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception as e:
                print(f"[WARN] 跳过坏行 {idx}: {e}")
                continue
            records.append(rec)
    return records


def iou(box1, box2):
    """计算两个 box 的 IoU"""
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2
    xi1, yi1 = max(x1, xx1), max(y1, yy1)
    xi2, yi2 = min(x2, xx2), min(y2, yy2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = max(0, (x2 - x1)) * max(0, (y2 - y1))
    area2 = max(0, (xx2 - xx1)) * max(0, (yy2 - yy1))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def check_consistency(pred):
    """一致性检查：metrics 是否满足 action 判据"""
    try:
        hazards = pred.get("hazards", [])
        metrics = pred.get("metrics", {})
        for hz in hazards:
            act = hz.get("action")
            if act == "cut_in":
                if abs(metrics.get("dx_norm", 0)) < 0.12 or metrics.get("cy_last_norm", 0) < 0.55:
                    return False
            if act == "hard_brake":
                if metrics.get("area_ratio", 1.0) < 1.4 or abs(metrics.get("cx_mid_norm", 0) - 0.5) > 0.25:
                    return False
        return True
    except Exception:
        return False


# -------------------------
# 核心评测
# -------------------------

def evaluate(preds, events):
    rows = []
    json_valid, fields_ok = 0, 0
    action_ok, total_action = 0, 0
    risk_ok, total_risk = 0, 0
    iou_scores, hit_scores, consistency_ok = [], [], 0

    y_true, y_pred = [], []

    # 构造 GT 索引
    gt_dict = defaultdict(list)
    for ev in events:
        seq_id = ev.get("seq_id")
        if seq_id:
            gt_dict[seq_id].append(ev)

    for rec in preds:
        seq_id = rec.get("seq_id", "unknown")
        pred = rec.get("prediction_json", {})
        if not isinstance(pred, dict):
            continue

        # JSON valid
        json_valid += 1

        # fields ok
        if all(k in pred for k in ["risk_level","hazards","reasoning","metrics","recommendation","evidence"]):
            fields_ok += 1

        # -------- Action 准确率 --------
        gt_action = None
        if seq_id in gt_dict and gt_dict[seq_id]:
            # 兼容不同字段名
            gt_action = (
                gt_dict[seq_id][0].get("action")
                or gt_dict[seq_id][0].get("label")
                or gt_dict[seq_id][0].get("event_type")
            )

        pred_action = pred.get("hazards", [{}])[0].get("action") if pred.get("hazards") else "none"

        if gt_action:
            total_action += 1
            if pred_action == gt_action:
                action_ok += 1
            y_true.append(gt_action)
            y_pred.append(pred_action)

        # -------- Risk 准确率 --------
        gt_risk = None
        if seq_id in gt_dict and gt_dict[seq_id]:
            gt_risk = gt_dict[seq_id][0].get("risk_level")

        pred_risk = pred.get("risk_level")

        if gt_risk and pred_risk:
            total_risk += 1
            if gt_risk == pred_risk:
                risk_ok += 1

        # -------- IoU / Hit@0.5 --------
        gt_boxes = []
        if seq_id in gt_dict:
            for ev in gt_dict[seq_id]:
                if "box2d" in ev:
                    gt_boxes.append(ev["box2d"])
        pred_boxes = [ev["box2d"] for ev in pred.get("evidence", []) if "box2d" in ev]

        if gt_boxes and pred_boxes:
            ious = [iou(pb, gb) for pb in pred_boxes for gb in gt_boxes]
            if ious:
                iou_scores.append(np.mean(ious))
                hit_scores.append(sum([1 for v in ious if v >= 0.5]) / len(ious))

        # -------- Consistency --------
        if check_consistency(pred):
            consistency_ok += 1

        rows.append({
            "seq_id": seq_id,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "gt_risk": gt_risk,
            "pred_risk": pred_risk,
            "risk_level": pred.get("risk_level"),
            "recommendation": pred.get("recommendation"),
        })

    n = max(1, len(preds))
    summary = {
        "total": len(preds),
        "json_valid": json_valid / n,
        "fields_ok": fields_ok / n,
        "action_acc": action_ok / max(1, total_action),
        "risk_acc": risk_ok / max(1, total_risk),
        "mean_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "hit@0.5": float(np.mean(hit_scores)) if hit_scores else 0.0,
        "consistency": consistency_ok / n,
    }

    return rows, summary, y_true, y_pred


# -------------------------
# 主程序
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_jsonl", type=str, required=True,
                        help="预测结果文件 (.jsonl 或 .jsonl.tmp)")
    parser.add_argument("--events", type=str, required=True,
                        help="事件 ground truth jsonl")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--confusion", action="store_true",
                        help="是否输出混淆矩阵 CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] 读取预测文件: {args.pred_jsonl}")
    preds = load_jsonl_safe(args.pred_jsonl)
    print(f"[INFO] 有效预测条数: {len(preds)}")

    print(f"[INFO] 读取 events: {args.events}")
    events = load_jsonl_safe(args.events)

    # 跑评测
    rows, summary, y_true, y_pred = evaluate(preds, events)

    # 输出 conv_eval.csv
    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "conv_eval.csv")
    df.to_csv(out_csv, index=False)

    # 输出 summary.json
    out_summary = os.path.join(args.out_dir, "summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 输出混淆矩阵
    if args.confusion and len(y_true) > 0:
        labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_path = os.path.join(args.out_dir, "action_confusion.csv")
        cm_df.to_csv(cm_path)
        print(f"[INFO] 混淆矩阵已保存 {cm_path}")

    print(f"[DONE] 已评测 {len(rows)} 条样本，summary: {summary}")
    print(f"[DONE] 报告已保存到 {args.out_dir}")


if __name__ == "__main__":
    main()
