#!/bin/bash
#SBATCH --job-name=BDD_SFT_testfromtmp
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00    # 运行时间限制
#SBATCH --mem=64G          # 内存需求

export HF_HOME=$(pwd)/cache
export TRANSFORMERS_CACHE=$HF_HOME
export WANDB_DISABLED=true

python scripts/eval_from_tmp_full.py \
  --pred_jsonl outputs/pred_holdout.jsonl.tmp \
  --events data/interim/events_val_holdout.jsonl \
  --out_dir reports/eval_partial \
  --confusion