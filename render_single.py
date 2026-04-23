import numpy as np
import os
import math
import torch
from torch import nn
import torchvision
import argparse
from tqdm import tqdm

from gaussian_renderer import GaussianModel, render
from arguments import PipelineParams, ModelParams
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(1, -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

class Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, image_width, image_height,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"):
        super(Camera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_width = image_width
        self.image_height = image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="示例程序")
    parser.add_argument('--ply_path', type=str, help='训练好的高斯模型ply地址')
    parser.add_argument('--exp_name', type=str, help='本次推理的实验名称')
    parser.add_argument('--images_txt_bin_path', type=str, help='需要渲染的images.txt或images.bin文件路径')
    parser.add_argument('--cameras_txt_bin_path', type=str, help='需要渲染的cameras.txt或cameras.bin文件路径')
    parser.add_argument('--sample_stride', type=int, help='从images位姿中采样的间隔')
    parser.add_argument('--image_downsample', type=int, help='渲染图像下采样倍数, 1为不下采样')

    pipeline = PipelineParams(parser)
    modelparam = ModelParams(parser)
    args = parser.parse_args()
    pipeline_args = pipeline.extract(args)
    modelparams_args = modelparam.extract(args)

    output_dir = args.ply_path.split(".")[0] + "_" + args.exp_name
    cmd = f'rm -rf {output_dir}'
    os.system(cmd)
    cmd = f'mkdir -p {output_dir}'
    os.system(cmd)

    sh_max = 3
    if not os.path.exists(args.ply_path):
        print("model path not exists: ", args.ply_path)
        exit(1)

    if not os.path.exists(args.images_txt_bin_path):
        print("images.txt/bin path not exists: ", args.images_txt_bin_path)
        exit(1)

    if not os.path.exists(args.cameras_txt_bin_path):
        print("cameras.txt/bin path not exists: ", args.cameras_txt_bin_path)
        exit(1)
    
    if args.images_txt_bin_path.split(".")[-1] == "txt":
        cam_extrinsics = read_extrinsics_text(args.images_txt_bin_path)
    else:
        cam_extrinsics = read_extrinsics_binary(args.images_txt_bin_path)
    print("Read cam_extrinsics from : ", args.images_txt_bin_path)
 
    if args.cameras_txt_bin_path.split(".")[-1] == "txt":
        cam_intrinsics = read_intrinsics_text(args.cameras_txt_bin_path)
    else:
        cam_intrinsics = read_intrinsics_binary(args.cameras_txt_bin_path)
    print("Read cam_intrinsics from : ", args.cameras_txt_bin_path)
    
    with torch.no_grad():
        gaussians = GaussianModel(modelparams_args)
        gaussians.load_ply(args.ply_path)
        print("已加载高斯模型: ", args.ply_path)
        print("正在渲染图像...")
        
        count_cam = -1
        render_num = 0
        for idx, key in tqdm(enumerate(cam_extrinsics), total=len(cam_extrinsics), desc="Processing Cameras"):
            count_cam += 1
            if count_cam % args.sample_stride != 0:
                continue
            # 准备相机参数
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = int(intr.height // args.image_downsample)
            width = int(intr.width // args.image_downsample)

            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0] / args.image_downsample
                focal_length_y = focal_length_x
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0] / args.image_downsample
                focal_length_y = intr.params[1] / args.image_downsample
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            image_name = str(extr.id) + ".png"
            fx = focal_length_x
            fy = focal_length_y
            rotation_matrix = R
            trans_matrix = T

            viewpoint_camera = Camera(rotation_matrix, trans_matrix, FovX, FovY, width, height)
            bg_color = [0,0,0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            # 渲染
            rendering = render(viewpoint_camera, gaussians, pipeline_args, background, use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE, infer_mode=True)
            rendering_img = rendering["render"]
            
            torchvision.utils.save_image(rendering_img, os.path.join(output_dir, image_name))
            render_num += 1
        print(f"共渲染了{render_num}个相机位姿")
        print("Render output img dir : ", output_dir)
