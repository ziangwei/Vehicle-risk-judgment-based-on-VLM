#!/bin/bash
#SBATCH --job-name=BDD_SFT
#SBATCH --partition=lrz-hgx-h100-94x4   # 或其他GPU分区
#SBATCH --gres=gpu:1       # 请求2个GPUs
#SBATCH --time=15:00:00    # 运行时间限制
#SBATCH --mem=64G          # 内存需求

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama-factory

export HF_HOME=$(pwd)/cache
export TRANSFORMERS_CACHE=$HF_HOME
export WANDB_DISABLED=true

cd LLaMA-Factory
srun llamafactory-cli train examples/train_vl/qwen2vl_bdd_multiframe_sft.yaml
