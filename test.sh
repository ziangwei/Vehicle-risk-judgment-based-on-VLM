#!/bin/bash
#SBATCH --job-name=BDD_SFT_test
#SBATCH --partition=lrz-hgx-h100-94x4   # 适配16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00    # 运行时间限制
#SBATCH --mem=64G          # 内存需求

export HF_HOME=$(pwd)/cache
export TRANSFORMERS_CACHE=$HF_HOME
export WANDB_DISABLED=true

# ==== 1) 推理（H100优化版脚本） ====
srun python infer_bdd_multiframe_h100.py \
  --adapter saves/qwen2_5_vl7b/bdd_multiframe_sft/checkpoint-1000 \
  --input data/processed/bdd_multiframe_sft_val_conv.jsonl \
  --out   outputs/pred_holdout.jsonl \
  --json_only_out outputs/pred_holdout.onlyjson.jsonl \
  --greedy --strict_fields \
  --max_new_tokens 512 --image_longest 448 \
  --merge_adapter --device cuda --attn_impl sdpa --compile

# ==== 2) 评测 ====
srun python scripts/eval_from_conversations_h100.py \
  --pred_jsonl outputs/pred_holdout.jsonl \
  --events     data/interim/events_val_holdout.jsonl \
  --manifest   data/interim/manifests/manifest_val.jsonl \
  --out_dir    reports/eval_holdout --confusion