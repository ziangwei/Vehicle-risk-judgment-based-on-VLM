# scripts/convert_coords_to_int.py
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# --- 核心功能：读取图片尺寸和转换坐标 ---

def get_image_size(image_path, cache):
    """获取图片尺寸，利用传入的字典进行缓存"""
    if image_path not in cache:
        try:
            with Image.open(image_path) as img:
                cache[image_path] = img.size
        except Exception as e:
            print(f"Warning: Cannot open image {image_path}. Error: {e}")
            cache[image_path] = None
    return cache[image_path]


def scale_box_to_int(box, orig_w, orig_h, target_longest_edge=448):
    """将原始像素坐标的bbox，缩放到目标尺寸并转换为整数"""
    # 1. 计算缩放比例
    if orig_w > orig_h:
        scale = target_longest_edge / orig_w
    else:
        scale = target_longest_edge / orig_h

    # 2. 缩放并四舍五入到整数
    # 使用列表推导式，代码更简洁
    int_box = [int(round(coord * scale)) for coord in box]

    return int_box


# --- 主处理逻辑 ---

def process_conv_file(input_path: Path, output_path: Path):
    """
    处理 *_conv.jsonl 文件，只转换坐标数字。
    """
    print(f"Reading from: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 尺寸缓存仅在单次文件处理中有效
    image_size_cache = {}

    with open(input_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:

        for line in tqdm(fin, desc=f"Converting {input_path.name}"):
            try:
                record = json.loads(line)

                # 获取图像尺寸
                first_image_path = record["images"][0]
                orig_size = get_image_size(first_image_path, image_size_cache)
                if orig_size is None:
                    continue  # 跳过无法读取图片的记录
                orig_w, orig_h = orig_size

                # 找到并修改 assistant 的回复
                for turn in record["conversations"]:
                    if turn.get("role") == "assistant":
                        assistant_json = json.loads(turn["content"])

                        evidence = assistant_json.get("evidence", [])
                        for ev in evidence:
                            box = ev.get("box2d")
                            if box and len(box) == 4:
                                # 原地修改 box
                                ev["box2d"] = scale_box_to_int(box, orig_w, orig_h)

                        # 将修改后的JSON转回字符串
                        turn["content"] = json.dumps(assistant_json, ensure_ascii=False)
                        break

                fout.write(json.dumps(record, ensure_ascii=False) + '\n')

            except Exception as e:
                # 简化错误处理
                pass

    print(f"Successfully created: {output_path}")


def main():
    """
    自动查找并处理训练集和验证集。
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"

    files_to_process = {
        "train": data_dir / "bdd_multiframe_sft_train_conv.jsonl",
        "val": data_dir / "bdd_multiframe_sft_val_conv.jsonl"
    }

    for split, path in files_to_process.items():
        if path.exists():
            output_path = path.with_name(path.stem + "_norm.jsonl")
            process_conv_file(path, output_path)
        else:
            print(f"File not found for '{split}' split, skipping: {path}")


if __name__ == "__main__":
    main()