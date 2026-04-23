#!/usr/bin/env bash
set -e

get_free_gpus() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1, NR-1}' | sort -nr | head -n ${#data_path_name[@]} | awk '{print $2}'
}

data_dir="${DATA_DIR:-/Your/path/to/data}"
exp_name=YOGO
exp_name_base=$exp_name"_base"
exp_name_expo=$exp_name"_expo"
exp_name_fusion=$exp_name"_fusion"
dir_name=(
    'scene00001'
    'scene00002'
    'scene00003'
    'scene00004'
)
data_path_name=("${dir_name[@]}")
cuda_device=($(get_free_gpus))

echo "即将启动训练: $exp_name_base"
for i in ${!dir_name[@]}
do
    echo "实验${i}_${dir_name[$i]} 显卡${cuda_device[$i]}"
    output_dir_base[$i]="./output_${exp_name_base}/${dir_name[$i]}"
    output_dir_expo[$i]="./output_${exp_name_expo}/${dir_name[$i]}"
    output_dir_fusion[$i]="./output_${exp_name_fusion}/${dir_name[$i]}"
done

ulimit -c 0
cp -r train.py "train_${exp_name_base}.py"

for i in ${!dir_name[@]}
do
    CUDA_VISIBLE_DEVICES="${cuda_device[$i]}" python "train_${exp_name_base}.py" \
        -s "$data_dir/${data_path_name[$i]}" \
        -d "$data_dir/${data_path_name[$i]}/depths" \
        --masks "$data_dir/${data_path_name[$i]}/masks" \
        -m "${output_dir_base[$i]}" \
        --port 359$((5+$i)) \
        --optimizer_type sparse_adam \
        --train_test_exp  \
        --sparse_dir "$data_dir/${data_path_name[$i]}/sparse/" \
        --points3D_dir "$data_dir/${data_path_name[$i]}/sparse/0/points3D.ply" \
        --sensor_mod only_x5 &
done
wait
echo "训练完成: $exp_name_base"
