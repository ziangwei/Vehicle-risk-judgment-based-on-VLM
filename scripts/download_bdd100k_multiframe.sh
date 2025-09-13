#!/usr/bin/env bash
set -euo pipefail

# === 配置 ===
DL_ROOT="${1:-data/raw/bdd100k}"   # 可传入自定义目标目录
mkdir -p "$DL_ROOT"/{tmp,images/track,labels}

# ETH Zürich 镜像直链（含 md5）
BASE="https://dl.cv.ethz.ch/bdd100k/data"
URL_TRACK_TRAIN="$BASE/track_images_train.zip"
URL_TRACK_TRAIN_MD5="$BASE/track_images_train.zip.md5"
URL_TRACK_VAL="$BASE/track_images_val.zip"
URL_TRACK_VAL_MD5="$BASE/track_images_val.zip.md5"
URL_BOX_TRACK_LABELS="$BASE/box_track_labels_trainval.zip"

# 下载器：优先 aria2c，没有就用 wget
down() {
  local url="$1" out="$2"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x16 -s16 -c -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
  else
    wget -c -O "$out" "$url"
  fi
}

echo ">>> 下载到: $DL_ROOT/tmp"
down "$URL_TRACK_TRAIN"      "$DL_ROOT/tmp/track_images_train.zip"
down "$URL_TRACK_TRAIN_MD5"  "$DL_ROOT/tmp/track_images_train.zip.md5"
down "$URL_TRACK_VAL"        "$DL_ROOT/tmp/track_images_val.zip"
down "$URL_TRACK_VAL_MD5"    "$DL_ROOT/tmp/track_images_val.zip.md5"
down "$URL_BOX_TRACK_LABELS" "$DL_ROOT/tmp/box_track_labels_trainval.zip"

# === 校验（有 md5 的就校验）===
echo ">>> 校验 MD5"
( cd "$DL_ROOT/tmp" && md5sum -c track_images_train.zip.md5 )
( cd "$DL_ROOT/tmp" && md5sum -c track_images_val.zip.md5 )

# === 解压 ===
echo ">>> 解压图像（可能较久）"
unzip -q -o "$DL_ROOT/tmp/track_images_train.zip" -d "$DL_ROOT"
unzip -q -o "$DL_ROOT/tmp/track_images_val.zip"   -d "$DL_ROOT"

# 解压后通常得到：$DL_ROOT/images/track/{train,val}/...
# 标签
echo ">>> 解压轨迹标签"
unzip -q -o "$DL_ROOT/tmp/box_track_labels_trainval.zip" -d "$DL_ROOT/labels"

# 可选：标准化一下标签目录名（便于之后脚本书写）
if [ -d "$DL_ROOT/labels/box_track_labels" ]; then
  mv "$DL_ROOT/labels/box_track_labels" "$DL_ROOT/labels/box_track_20"
fi

echo ">>> 完成。示例路径："
echo "  图像:  $DL_ROOT/images/track/train/0000/00000001.jpg"
echo "  标签:  $DL_ROOT/labels/box_track_20/box_track_train_coco.json 或同名结构文件"
