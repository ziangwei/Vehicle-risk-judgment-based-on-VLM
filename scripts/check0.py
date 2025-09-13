import json

fn = "processed/bdd_multiframe_sft_train_conv.jsonl"
with open(fn, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        imgs = data.get("images", [])
        convs = data.get("conversations", [])
        text = "".join([m.get("content", "") for m in convs])
        n_placeholders = text.count("<image>")
        if n_placeholders != len(imgs):
            print(f"❌ 行 {i}: images={len(imgs)}, <image> tokens={n_placeholders}")
