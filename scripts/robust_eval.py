# scripts/robust_eval.py
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm  # <--- 确保 tqdm 被正确引用

# --- 核心功能函数 ---

def iou(boxA, boxB):
    """计算两个边界框的IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


def get_first_action(pred_json):
    """从预测JSON中提取第一个有效的action"""
    if not isinstance(pred_json, dict) or "hazards" not in pred_json:
        return "no_action"
    hazards = pred_json["hazards"]
    if isinstance(hazards, list) and len(hazards) > 0:
        action = hazards[0].get("action")
        return action if action else "no_action"
    return "no_action"


def load_jsonl(path):
    """加载JSONL文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


# --- 主评估逻辑 ---

def main():
    parser = argparse.ArgumentParser(description="A robust evaluation script for VLM risk assessment.")
    parser.add_argument("--pred_jsonl", required=True, help="Path to the prediction JSONL file from inference.")
    parser.add_argument("--gt_jsonl", required=True,
                        help="Path to the ground truth conversation JSONL file (e.g., bdd_multiframe_sft_val_conv_norm.jsonl).")
    parser.add_argument("--out_dir", default="reports/robust_eval", help="Directory to save evaluation reports.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions and ground truth...")
    predictions = load_jsonl(args.pred_jsonl)
    ground_truths = load_jsonl(args.gt_jsonl)

    if len(predictions) != len(ground_truths):
        print(
            f"Warning: Prediction count ({len(predictions)}) does not match GT count ({len(ground_truths)}). Evaluation might be inaccurate.")
        # 截断以较短的为准，以防万一
        min_len = min(len(predictions), len(ground_truths))
        predictions = predictions[:min_len]
        ground_truths = ground_truths[:min_len]

    all_results = []
    y_true = []
    y_pred = []

    for i, (pred_record, gt_record) in enumerate(
            tqdm(zip(predictions, ground_truths), total=len(predictions), desc="Evaluating samples")):

        # --- 1. 提取GT信息 ---
        gt_assistant_json = json.loads(gt_record["conversations"][-1]["content"])
        gt_action = get_first_action(gt_assistant_json)
        gt_evidence_map = {ev["frame_idx"]: ev["box2d"] for ev in gt_assistant_json.get("evidence", [])}

        # --- 2. 提取预测信息 ---
        pred_json = pred_record.get("prediction_json", {})
        pred_action = get_first_action(pred_json)

        y_true.append(gt_action)
        y_pred.append(pred_action)

        # --- 3. 修正frame_idx并计算IoU ---
        # 关键修正：我们不信任模型生成的frame_idx。
        # 我们假设模型输出的evidence顺序与输入图片的顺序是一一对应的。
        # GT的frame_idx_list是我们的“真实坐标系”。
        input_frame_ids = gt_record.get("frame_idx_list", [])

        ious = []
        pred_evidence = pred_json.get("evidence", [])

        if gt_evidence_map and input_frame_ids and pred_evidence:
            for pred_idx, pred_ev in enumerate(pred_evidence):
                if pred_idx < len(input_frame_ids):
                    # 这就是对齐的关键：用输入帧的ID作为真实ID
                    true_frame_id = input_frame_ids[pred_idx]

                    if true_frame_id in gt_evidence_map:
                        gt_box = gt_evidence_map[true_frame_id]
                        pred_box = pred_ev.get("box2d")
                        if gt_box and pred_box and len(gt_box) == 4 and len(pred_box) == 4:
                            ious.append(iou(pred_box, gt_box))

        mean_iou = np.mean(ious) if ious else 0.0
        hit_at_0_5 = np.mean([1 for v in ious if v >= 0.5]) if ious else 0.0

        all_results.append({
            "sample_index": i,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "action_correct": 1 if gt_action == pred_action else 0,
            "mean_iou": mean_iou,
            "hit@0.5": hit_at_0_5,
        })

    # --- 4. 生成报告 ---
    print("\n" + "=" * 20 + " Evaluation Summary " + "=" * 20)

    # 整体准确率
    overall_accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Action Accuracy: {overall_accuracy:.4f}")

    # 分类报告 (精确率, 召回率, F1)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    print("\nClassification Report:")
    print(report)

    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = out_dir / "classification_report.csv"
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")

    # 混淆矩阵
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nConfusion Matrix:")
    print(cm_df)
    cm_path = out_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # 详细结果
    details_df = pd.DataFrame(all_results)
    details_path = out_dir / "evaluation_details.csv"
    details_df.to_csv(details_path, index=False)
    print(f"Detailed results saved to {details_path}")

    # 总体指标JSON
    summary_path = out_dir / "summary.json"
    summary_data = {
        "overall_accuracy": overall_accuracy,
        "macro_avg": report_dict.get("macro avg", {}),
        "weighted_avg": report_dict.get("weighted avg", {}),
        "mean_iou_avg": details_df["mean_iou"].mean(),
        "hit@0.5_avg": details_df["hit@0.5"].mean(),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4)
    print(f"Summary JSON saved to {summary_path}")


if __name__ == "__main__":
    main()