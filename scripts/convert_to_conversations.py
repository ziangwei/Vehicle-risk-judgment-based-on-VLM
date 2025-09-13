import json
import os

def convert_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            images = data.get("images", [])
            prompt = data.get("prompt", "")
            response = data.get("response", "")

            new_item = {
                "images": images,
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            }

            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    print(f"✅ 转换完成: {input_file} → {output_file}")


if __name__ == "__main__":
    # 修改成你自己的路径
    input_path = "data/processed/bdd_multiframe_sft_train.jsonl"
    output_path = input_path.replace(".jsonl", "_conv.jsonl")

    convert_file(input_path, output_path)

    # 如果还有 val 数据
    input_val = "data/processed/bdd_multiframe_sft_val.jsonl"
    if os.path.exists(input_val):
        output_val = input_val.replace(".jsonl", "_conv.jsonl")
        convert_file(input_val, output_val)
