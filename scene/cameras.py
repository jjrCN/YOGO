import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
from utils.image_utils import high_frequency_heatmap_fourier
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, mask, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False, cam_info=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.is_test_view = is_test_view

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        resized_mask = None
        if mask is not None:
            resized_mask = PILtoTorch(mask, resolution)
            if resized_mask.shape[0] != 1:
                resized_mask = resized_mask[:1, ...]
            resized_mask[resized_mask > 0] = 1.

            # masked_preview = torch.clone(resized_image_rgb)
            # masked_preview[0, resized_mask[0] == 0.] /= 4.
            # masked_preview[0, resized_mask[0] == 0.] += 0.75
            # preview_save_path = os.path.join(args.model_path, "mask_preview", cam_info.image_name.replace("/", "_"))
            # os.makedirs(os.path.dirname(preview_save_path), exist_ok=True)
            # torchvision.utils.save_image(masked_preview, preview_save_path)

        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        # if train_test_exp and is_test_view:
        #     if is_test_dataset:
        #         self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
        #     else:
        #         self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        #frequency_heatmap gen
        self.freq_hm = None
        gt_image_np = gt_image.permute(1, 2, 0).cpu().numpy()
        gt_image_gray = cv2.cvtColor(gt_image_np, cv2.COLOR_RGB2GRAY)
        freq_hm = high_frequency_heatmap_fourier(gt_image_gray)
        self.freq_hm = freq_hm.to(self.data_device)

        self.raw_mask = mask
        self.is_masked = None
        if resized_mask is not None:
            # print("resized_mask is not None:")
            self.is_masked = (resized_mask == 0).expand(*resized_image_rgb.shape)  # True represent masked pixel

        self.invdepthmap = None
        self.depth_reliable = False
        self.depth_mask = torch.ones_like(self.alpha_mask)
        
        if invdepthmap is not None:
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] <= 0.0 or depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                else:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

            if self.depth_reliable:
                self.depth_mask += torch.clamp(self.invdepthmap, min=0.1, max=10.0)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        fx = cam_info.fx // (cam_info.width / self.image_width)
        fy = cam_info.fy // (cam_info.width / self.image_width)
        cx = cam_info.cx // (cam_info.width / self.image_width)
        cy = cam_info.cy // (cam_info.width / self.image_width)
        # self.projection_matrix = getProjectionMatrix_cxcy(znear=self.znear, zfar=self.zfar, fx=fx, fy=fy, cx=cx, cy=cy, W=self.image_width, H=self.image_height).transpose(0,1).cuda()
        
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
