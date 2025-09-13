import argparse
import json
from datasets import load_dataset
from torch.utils.data import DistributedSampler, DataLoader

def check_images_and_tokens(dataset_file):
    print(f"Checking dataset: {dataset_file}")
    ds = load_dataset("json", data_files=dataset_file, split="train")

    errors = []
    for i, ex in enumerate(ds):
        # 图像数量
        n_img = len(ex["images"]) if "images" in ex else 0
        # prompt 里的 <image> token 数量
        conv = ex.get("conversations", [])
        user_content = ""
        for msg in conv:
            if msg["role"] == "user":
                user_content += msg["content"]
        n_tok = user_content.count("<image>")

        if n_img != n_tok:
            errors.append((i, n_img, n_tok))

    if errors:
        print(f"❌ Found {len(errors)} mismatched samples:")
        for e in errors[:10]:  # 只打印前 10 条
            print(f"  sample {e[0]}: images={e[1]}, tokens={e[2]}")
    else:
        print("✅ All samples have matching <image> tokens and images.")

def simulate_distributed_split(dataset_file, world_size=2):
    print(f"\nSimulating distributed split with world_size={world_size}")
    ds = load_dataset("json", data_files=dataset_file, split="train")

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=0, shuffle=False)
    sampler2 = DistributedSampler(ds, num_replicas=world_size, rank=1, shuffle=False)

    len_rank0 = len(list(iter(sampler)))
    len_rank1 = len(list(iter(sampler2)))

    print(f"Rank0 samples: {len_rank0}")
    print(f"Rank1 samples: {len_rank1}")

    if len_rank0 == len_rank1:
        print("✅ Dataset splits are aligned across ranks.")
    else:
        print("❌ Dataset splits mismatch! Training will hang.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to your dataset jsonl file")
    parser.add_argument("--world_size", type=int, default=2, help="Simulated number of GPUs")
    args = parser.parse_args()

    check_images_and_tokens(args.dataset_file)
    simulate_distributed_split(args.dataset_file, args.world_size)
