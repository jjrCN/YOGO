#!/usr/bin/env bash
set -e

### 自行设置
# 本次实验的备注
exp_name="${EXP_NAME:-YOGO}"
model_base_path="${MODEL_BASE_PATH:-/XXX/XXX/XXX/iteration_XXX}"
images_txt_bin_path="${IMAGES_TXT_BIN_PATH:-/Path/to/images.txt}"
cameras_txt_bin_path="${CAMERAS_TXT_BIN_PATH:-/Path/to/cameras.txt}"
sample_stride="${SAMPLE_STRIDE:-1}"

######################################
######################################

# 下面无需改动
# 渲染参数设置
image_downsample="${IMAGE_DOWNSAMPLE:-1}" #渲染分辨率下采样倍数
# 开始渲染
python render_single.py \
    --ply_path "$model_base_path/point_cloud.ply" \
    --exp_name "$exp_name" \
    --images_txt_bin_path "$images_txt_bin_path" \
    --cameras_txt_bin_path "$cameras_txt_bin_path" \
    --sample_stride "$sample_stride" \
    --image_downsample "$image_downsample"
