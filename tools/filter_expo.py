import json
import numpy as np
import argparse
import os

def analyze_exposure_matrix(matrix):
    """分析3DGS曝光矩阵（3x4），输出综合偏差评分。"""
    mat = np.array(matrix, dtype=float)
    if mat.shape != (3, 4):
        raise ValueError("矩阵必须是 3x4 维度")
    
    gain_matrix = mat[:, :3]
    bias_vector = mat[:, 3]
    
    diag = np.diag(gain_matrix)
    gain_diff = diag - 1.0
    
    off_diag_mask = ~np.eye(3, dtype=bool)
    off_diag_values = gain_matrix[off_diag_mask]
    off_diag_mean = np.mean(np.abs(off_diag_values))
    
    bias_mean = np.mean(np.abs(bias_vector))
    
    exposure_bias_score = np.max(np.abs(gain_diff)) + off_diag_mean + bias_mean
    
    return exposure_bias_score


def process_exposure_json(json_path, threshold=0.15):
    """处理曝光 JSON，返回 bad_images 列表"""
    with open(json_path, 'r', encoding='utf-8') as f:
        exposure_data = json.load(f)
    
    bad_images = []
    for key, matrix in exposure_data.items():
        try:
            score = analyze_exposure_matrix(matrix)
            if score > threshold:  # 超过阈值就判定为 bad
                bad_images.append(key)
        except Exception as e:
            print(f"⚠️ 处理 {key} 出错: {e}")
    
    print(f"✅ 筛选完成，共 {len(bad_images)} 张 bad images")
    return bad_images


def save_list_as_txt(lines, txt_path, replace=True):
    """把 bad images 列表保存成 TXT 文件"""
    save_path = txt_path if replace else txt_path + ".bk"

    with open(save_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + "\n")

    print(f"✅ 已生成 TXT 文件: {save_path}，共 {len(lines)} 个条目")


def main():
    parser = argparse.ArgumentParser(description="分析 exposure.json 并生成 bad images TXT 文件")
    parser.add_argument("--json", required=True, help="输入 exposure.json 文件路径")
    parser.add_argument("--out", required=True, help="输出 TXT 文件路径")
    parser.add_argument("--threshold", type=float, default=0.15, help="判定 bad image 的阈值 (默认: 0.15)")
    parser.add_argument("--replace", action="store_true", help="是否替换已有文件，默认 False 生成 .bk 文件")

    args = parser.parse_args()

    if not os.path.exists(args.json):
        raise FileNotFoundError(f"找不到 JSON 文件: {args.json}")

    bad_images = process_exposure_json(args.json, threshold=args.threshold)
    save_list_as_txt(bad_images, args.out, replace=args.replace)


if __name__ == "__main__":
    main()
