#!/bin/bash
#SBATCH --job-name=BDD_SFT_test_baseline
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G

export HF_HOME=$(pwd)/cache
export TRANSFORMERS_CACHE=$HF_HOME
export WANDB_DISABLED=true

# ==== 1) 推理 (运行基线模型) ====
echo "Running BASELINE inference..."
srun python scripts/infer_baseline.py \
  --input data/processed/bdd_multiframe_sft_val_conv.jsonl \
  --out   outputs/pred_holdout_baseline.jsonl \
  --max_new_tokens 512 \
  --image_longest 448 \
  --device cuda:0 \
  --attn_impl sdpa \
  --compile

# ==== 2) 评测 (评估基线的结果) ====
echo "Evaluating BASELINE results..."
srun python scripts/robust_eval.py \
  --pred_jsonl    outputs/pred_holdout_baseline.jsonl \
  --gt_jsonl      data/processed/bdd_multiframe_sft_val_conv.jsonl \
  --out_dir       reports/robust_eval_baseline