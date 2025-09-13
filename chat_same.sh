#!/bin/bash
#SBATCH --job-name=BDD_SFT_chat
#SBATCH --gres=gpu:2
#SBATCH --partition=lrz-hgx-h100-94x4   # 适配16
#SBATCH --time=1:00:00    # 运行时间限制
#SBATCH --mem=64G          # 内存需求

export HF_HOME=$(pwd)/cache
export TRANSFORMERS_CACHE=$HF_HOME
export WANDB_DISABLED=true

python infer_bdd_multiframe_h100.py \
  --adapter saves/qwen2_5_vl7b/bdd_multiframe_sft/checkpoint-2000 \
  --input one_record_same5.json \
  --single \
  --out outputs/pred_same5.jsonl \
  --json_only_out outputs/pred_same5.onlyjson.jsonl \
  --greedy --strict_fields \
  --append_frame_idx_to_prompt \
  --force_evidence_frames by_order \
  --max_new_tokens 512 --image_longest 384 \
  --merge_adapter --device cuda --attn_impl sdpa --compile
