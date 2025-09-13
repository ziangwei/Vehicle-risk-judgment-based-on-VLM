import json, sys, os, shutil

def patch(in_path):
    out_path = in_path + ".patched"
    token = "<image>"
    bad = ok = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            n = len(ex.get("images", []))
            # 统计 prompt 中已有的 <image> 数量
            cnt = ex.get("prompt","").count(token)
            if n>0 and cnt != n:
                # 若没有或数量不匹配，就重写为 n 个占位符 + 原提示
                ex["prompt"] = (token + "\n") * n + ex.get("prompt","")
                bad += 1
            else:
                ok += 1
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
    # 覆盖原文件
    backup = in_path + ".bak"
    shutil.move(in_path, backup)
    shutil.move(out_path, in_path)
    print(f"patched: {in_path} (ok {ok}, fixed {bad}, backup -> {backup})")

patch("data/processed/bdd_multiframe_sft_train.jsonl")
patch("data/processed/bdd_multiframe_sft_val.jsonl")