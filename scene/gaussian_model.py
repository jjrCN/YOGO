import torch
import numpy as np
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from scene.appearance import AppearanceMLP, AppearanceEmbedding
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, strip_symmetric, build_scaling_rotation, get_polygons, compute_points_in_each_polygon_torch

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except:
    OPEN3D_AVAILABLE = False



class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, args, optimizer_type="default", use_sh_embedding=False):
        self.dense_stage_time = 0
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = args.sh_degree  
        self._xyz = torch.empty(0)
        self._lidar_init_mask = torch.empty(0)
        self._min_xyz_m = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.abs_xyz_gradient_accum = torch.empty(0)
        self.effect_opacity = torch.empty(0)
        self.pivot = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.effective_flag = False
        self.effect_opacity_prune_num_per_itv = 0
        self.save_densify_result = False

        self.polygons = []
        self.objects = []
        self.house_polygon = []
        if os.path.exists(args.polygon_3d_list_path):
            self.polygons, self.objects, self.house_polygon = get_polygons(args.polygon_3d_list_path) 
            # polygons: list, 每个元素是一个polygon, 每个polygon也是一个list，其中每个元素是一个面的4个顶点，顶点数序支持乱序，当前仅支持立方体3dbox
            # objects: list，每个元素是一个字典，用于储存每个polygon的原始信息
            # 给objects增加一个空字典，用于储存不在任何polygon的点的信息
        print("house_polygon : ", len(self.house_polygon))
        self.objects.append({"name": "background"})
        self.effective_increase_num_per_densify = [0 for i in range(len(self.polygons))]

        self.use_sh_embedding = use_sh_embedding

    def apply_lr_mult(self, set_grad_zero=False):
        for tensor in [self._xyz, self._scaling, self._rotation, self._opacity, self._features_dc, self._features_rest]:
            if tensor.grad is None: 
                continue
            # 每行对应一个高斯
            for i in range(len(self.points_in_each_polygon)):
                if self.objects[i]["freeze_gs"] or set_grad_zero:
                    tensor.grad[self.points_index_in_each_polygon[i]] *= 0.0 
                    # if len(tensor.grad.shape) == 2:
                    #     tensor.grad[self.points_index_in_each_polygon[i]] *= 0.0  
                    # else:
                    #     tensor.grad[self.points_index_in_each_polygon[i]] *= 0.0  

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_num_points(self):
        return self.get_xyz.shape[0]
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : list, spatial_lr_scale : float, init_clamp_scale_rate : float, init_delete_scale_rate : float, pretrained_exposure_path : str):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # compute thred
        dist2_temp = dist2.clone()
        sort_dist2_temp, _ = torch.sort(dist2_temp)
        clamp_thred = sort_dist2_temp[int((sort_dist2_temp.shape[0] - 1) * init_clamp_scale_rate)]
        dist2 = torch.clamp_max(dist2, clamp_thred)

        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        ## add lidar init mask
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        if os.path.exists(pretrained_exposure_path):
            with open(pretrained_exposure_path, "r") as f:
                exposures = json.load(f)
            self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
            print(f"Pretrained exposures loaded : ", pretrained_exposure_path)
        else:
            print(f"No exposure to be loaded at {pretrained_exposure_path}")
            self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        self.num_points = self.get_xyz.shape[0]
        self.dense_stage = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.prune_points_xyz = None
        self.prune_points_dense_stage = None

        self.color_predictor = None
        self.appearance_app = None
        if self.use_sh_embedding:
            self.color_predictor  = AppearanceMLP(sh_dim=16, app_dim=32).to("cuda")
            self.appearance_app = AppearanceEmbedding(num_images=len(cam_infos), app_dim=32).to("cuda")
        
        self.points_in_house_polygon, self.points_index_in_house_polygon = compute_points_in_each_polygon_torch(self.house_polygon, self.get_xyz.detach())
        temp = 0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        # temp[self.points_index_in_house_polygon[-1]] *= 9 
        self._opacity = nn.Parameter(self.inverse_opacity_activation(temp).requires_grad_(True))

    def save_pcd_result(self, path):
        if OPEN3D_AVAILABLE:
            mkdir_p(os.path.dirname(path))
            result_xyz = self._xyz.detach().cpu().numpy()
            result_dense_stage = self.dense_stage.detach().cpu().numpy()
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(result_xyz)
            intensity_normalized = result_dense_stage / (self.dense_stage_time * 1.0 + 1.0)
            intensity_normalized = intensity_normalized.astype(np.float64)
            point_cloud.colors = o3d.utility.Vector3dVector(np.tile(intensity_normalized, (1, 3)))
            o3d.io.write_point_cloud(path, point_cloud)

    
    def save_pcd_prune(self, path):
        if OPEN3D_AVAILABLE and self.prune_points_xyz != None:
            mkdir_p(os.path.dirname(path))
            prune_xyz = self.prune_points_xyz.detach().cpu().numpy()
            prune_dense_stage = self.prune_points_dense_stage.detach().cpu().numpy()
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(prune_xyz)
            intensity_normalized = prune_dense_stage / (self.dense_stage_time * 1.0 + 1.0)
            intensity_normalized = intensity_normalized.astype(np.float64)
            point_cloud.colors = o3d.utility.Vector3dVector(np.tile(intensity_normalized, (1, 3)))
            o3d.io.write_point_cloud(path, point_cloud)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.abs_xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.effect_opacity = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.pivot = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.relative_fineness = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.training_args = training_args

        base_lr = 1.0
        if training_args.only_train_expo:
            print("使用仅训练曝光系数模式, gs属性学习率调整为0.0")
            base_lr = 0.0

        l = [
            {'params': [self._xyz], 'lr': base_lr * training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': base_lr * training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': base_lr * training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': base_lr * training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': base_lr * training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': base_lr * training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, cam_infos = None, pretrained_exposure_path = "exposure.json"):
        plydata = PlyData.read(path)
        if cam_infos is not None:
            self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}

            if os.path.exists(pretrained_exposure_path):
                with open(pretrained_exposure_path, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded : ", pretrained_exposure_path)
            else:
                print(f"No exposure to be loaded at {pretrained_exposure_path}")
                self.pretrained_exposures = None
                exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
                self._exposure = nn.Parameter(exposure.requires_grad_(True))
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.num_points = self.get_xyz.shape[0]
        self.dense_stage = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.prune_points_xyz = None
        self.prune_points_dense_stage = None

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, prune_flag):
        valid_points_mask = ~mask

        if prune_flag == True:
            num_prune_points_xyz = 0
            if self.prune_points_xyz == None:
                self.prune_points_xyz = self._xyz[mask]
                self.prune_points_dense_stage = self.dense_stage[mask]
            else:
                num_prune_points_xyz = self.prune_points_xyz.shape[0]
                self.prune_points_xyz = torch.cat((self.prune_points_xyz, self._xyz[mask]), dim=0)
                self.prune_points_dense_stage = torch.cat((self.prune_points_dense_stage, self.dense_stage[mask]), dim=0)
            
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.abs_xyz_gradient_accum = self.abs_xyz_gradient_accum[valid_points_mask]
        self.effect_opacity = self.effect_opacity[valid_points_mask]
        self.pivot = self.pivot[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.relative_fineness = self.relative_fineness[valid_points_mask]
        self.dense_stage = self.dense_stage[valid_points_mask]
        self.num_points = self.get_xyz.shape[0]

        # 每次稠密化或剪枝操作后都会执行此函数，因此在这更新一下polygon索引，保证成员变量的值一直是最新的
        self.points_in_each_polygon, self.points_index_in_each_polygon = compute_points_in_each_polygon_torch(self.polygons, self.get_xyz.detach())
        self.points_in_house_polygon, self.points_index_in_house_polygon = compute_points_in_each_polygon_torch(self.house_polygon, self.get_xyz.detach())

    def prune_points_init(self):
        valid_points_mask = torch.ones(self._xyz.shape[0], device="cuda", dtype=bool)
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.abs_xyz_gradient_accum = self.abs_xyz_gradient_accum[valid_points_mask]
        self.effect_opacity = self.effect_opacity[valid_points_mask]
        self.pivot = self.pivot[valid_points_mask]
        self.relative_fineness = self.relative_fineness[valid_points_mask]
        self.dense_stage = self.dense_stage[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.num_points = self.get_xyz.shape[0]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, \
                    new_scaling, new_rotation, new_relative_fineness, new_effect_opacity, new_pivot, new_dense_stage):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.relative_fineness = torch.cat((self.relative_fineness, new_relative_fineness))
        self.effect_opacity = torch.cat((self.effect_opacity, new_effect_opacity))
        self.dense_stage = torch.cat((self.dense_stage, new_dense_stage))
        self.pivot = torch.cat((self.pivot, new_pivot))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.abs_xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads: torch.Tensor):
        """
        """
        import numpy as np
        device = grads.device
        if grads.ndim > 1 and grads.shape[1] == 1:
            grads = grads.view(-1)

        num_polygons = len(self.points_in_each_polygon)

        # === Step 1: 统一把 points_index_in_each_polygon 转成 per-polygon 索引（torch.LongTensor）列表 ===
        perpoly_idx = []
        total_points = grads.shape[0]
        for i, arr in enumerate(self.points_index_in_each_polygon):
            idx = arr.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                idx = idx.new_empty((0,), dtype=torch.long)
            perpoly_idx.append(idx)

        # 如果所有 polygon 都为空，直接返回
        total_selected_points = sum([x.numel() for x in perpoly_idx])
        print(f"[densify] 所有 polygon 总包含点数 (sum of per-polygon sizes): {int(total_selected_points)}")
        if total_selected_points == 0:
            print("[densify] 没有任何点属于任何 polygon, 跳过 densify")
            return

        # 拼接全局索引与 polygon id（长度 = sum perpoly sizes）
        all_indices = torch.cat(perpoly_idx)  # LongTensor
        polygon_ids = torch.cat([
            torch.full((idx.numel(),), i, dtype=torch.long, device=device)
            for i, idx in enumerate(perpoly_idx)
        ])

        # sanity check
        if all_indices.numel() != polygon_ids.numel():
            raise RuntimeError("internal error: all_indices and polygon_ids length mismatch")

        # === Step 2: 使用 all_indices 对 grads 索引 ===
        grads_all = grads[all_indices]  # 现在长度正确 = sum perpoly sizes

        # === Step 3: 计算每个 polygon 的 inc_num 等 ===
        cur_counts = torch.tensor(self.points_in_each_polygon, device=device)
        targets = torch.tensor([obj.get("target_gaussian_num", 0) for obj in self.objects], device=device)
        abs_increase_flags = torch.tensor(
            [obj.get("absolutely_increase_num_control", False) for obj in self.objects],
            device=device, dtype=torch.bool
        )
        increase_each_time = torch.tensor(
            [obj.get("increase_gaussian_num_each_time", 0) for obj in self.objects],
            device=device
        )
        effective_increase = torch.tensor(self.effective_increase_num_per_densify, device=device)
        inc_num = torch.where(
            abs_increase_flags,
            (targets - cur_counts) // (self.densification_times + 1 - self.dense_stage_time),
            effective_increase + increase_each_time
        ).clamp(min=0)

        extra_cond = abs_increase_flags & (targets > cur_counts) & (inc_num <= 1)
        inc_num = torch.where(extra_cond, torch.ones_like(inc_num) * 2, inc_num)

        # === Step 4: 计算 per-polygon threshold===
        k_ks = (inc_num // 2).clamp(min=0)
        thresholds = torch.full((num_polygons,), float("-inf"), device=device)
        for i in range(num_polygons):
            if self.objects[i].get("freeze_densify", False):
                print("[densify] polygon {}, polygon name: {}, freeze_densify=True, 跳过 densify".format(i, self.objects[i]["name"]))
                continue
            # 从 all_indices 的 concat 结构中筛出这个 polygon 的项
            cur_mask = (polygon_ids == i)
            if cur_mask.sum() == 0:
                # 该 polygon 没点，跳过
                continue
            g_sub = grads_all[cur_mask]
            k_k = int(k_ks[i].item())
            if g_sub.numel() == 0:
                continue
            if k_k <= 0: 
                clamp_thred = g_sub.max() + 0.1 # 设置成比最大值大0.1，表示无高斯需要分裂
            elif k_k <= g_sub.numel():
                clamp_thred = torch.topk(g_sub, k_k, largest=True, sorted=True).values[-1]
            else:
                clamp_thred = g_sub.min()
            thresholds[i] = clamp_thred
            print("[densify] polygon {}, polygon name: {}, 被选中分裂的高斯数={}, 当前点数={}, 目标点数={}".format(i, self.objects[i]["name"], k_k, g_sub.numel(), self.objects[i]["target_gaussian_num"]))

        # === Step 5: 批量选点并回到全局 mask ===
        clamp = thresholds[polygon_ids]           # 每个位置对应其 polygon 的阈值
        selected_pts_mask_local = (grads_all >= clamp)  # 相对于 grads_all 的局部 mask
        global_selected_mask = torch.zeros_like(grads, dtype=torch.bool)
        global_selected_mask[all_indices[selected_pts_mask_local]] = True

        num_selected = int(global_selected_mask.sum().item())
        print(f"[densify] 本轮选中点数 (global): {num_selected}")
        if num_selected == 0:
            print("[densify] 无选中点，跳过 densify")
            return

        # === Step 6: 构造新高斯参数等 ===
        idx = global_selected_mask.nonzero(as_tuple=False).squeeze(-1)
        stds = self.get_scaling[idx]
        rots = build_rotation(self._rotation[idx])
        _, max_dim_idx = stds.max(dim=1)
        main_axis_local = torch.zeros_like(stds)
        main_axis_local[torch.arange(stds.size(0), device=device), max_dim_idx] = 1.0
        std_for_main_axis = stds[torch.arange(stds.size(0), device=device), max_dim_idx] * 0.3
        delta_length = torch.normal(0, std_for_main_axis).unsqueeze(-1)
        delta_local = main_axis_local * delta_length
        delta_xyz = torch.bmm(rots, delta_local.unsqueeze(-1)).squeeze(-1)

        get_xyz_sel = self.get_xyz[idx]
        new_xyz1 = delta_xyz + get_xyz_sel
        new_xyz0 = -delta_xyz + get_xyz_sel

        new_scaling1 = self.scaling_inverse_activation(self.get_scaling[idx] / 1.6)
        new_opacity1 = self.inverse_opacity_activation(self.get_opacity[idx] * 0.3)
        new_relative_fineness1 = (self.relative_fineness[idx] + 1.0)
        new_dense_stage1 = torch.ones_like(self.dense_stage[idx]) * self.dense_stage_time
        new_effect_opacity1 = self.effect_opacity[idx] * 0.3
        new_pivot1 = self.pivot[idx]

        new_xyz2 = self._xyz[idx]
        new_scaling2 = self.scaling_inverse_activation(self.get_scaling[idx])
        new_opacity2 = self.inverse_opacity_activation(self.get_opacity[idx] * 0.3)
        new_relative_fineness2 = self.relative_fineness[idx]
        new_dense_stage2 = self.dense_stage[idx]
        new_effect_opacity2 = self.effect_opacity[idx] * 0.3
        new_pivot2 = self.pivot[idx]

        new_xyz = torch.cat([new_xyz1, new_xyz2, new_xyz0], dim=0)
        new_scaling = torch.cat([new_scaling1, new_scaling2, new_scaling1], dim=0)
        new_opacity = torch.cat([new_opacity1, new_opacity2, new_opacity1], dim=0)
        new_relative_fineness = torch.cat([new_relative_fineness1, new_relative_fineness2, new_relative_fineness1], dim=0)
        new_dense_stage = torch.cat([new_dense_stage1, new_dense_stage2, new_dense_stage1], dim=0)
        new_features_dc = self._features_dc[idx].repeat(3, 1, 1)
        new_features_rest = self._features_rest[idx].repeat(3, 1, 1)
        new_rotation = self._rotation[idx].repeat(3, 1)
        new_effect_opacity = torch.cat([new_effect_opacity1, new_effect_opacity2, new_effect_opacity1], dim=0)
        new_pivot = torch.cat([new_pivot1, new_pivot2, new_pivot1], dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                new_relative_fineness, new_effect_opacity, new_pivot, new_dense_stage)

        prune_filter = torch.cat([
            global_selected_mask,
            torch.zeros(3 * num_selected, device=device, dtype=torch.bool)
        ])
        self.prune_points(prune_filter, False)
        self.num_points = self.get_xyz.shape[0]
        print(f"[densify] 新增高斯数量: {2 * num_selected}, densify 后总点数: {self.num_points}")


    def effective_prune(self, densification_times_per_epoch):
        min_effect_opacity = 0.01
        min_effect_opacity_ratio = 0.02
        total_points = self.get_xyz.shape[0]
        candidate_prune_mask = (self.effect_opacity < min_effect_opacity).squeeze()
        candidate_prune_count = torch.sum(candidate_prune_mask)
        print("candidate_prune_count: ", candidate_prune_count)
        max_prune_by_ratio = int(total_points * min_effect_opacity_ratio)
        if candidate_prune_count > max_prune_by_ratio:
            sorted_opacities, _ = torch.sort(self.effect_opacity.squeeze())
            threshold_opacity = sorted_opacities[max_prune_by_ratio - 1]
            print("max_prune_by_ratio", max_prune_by_ratio, "use threshold_opacity: ", threshold_opacity)
            prune_mask = (self.effect_opacity <= threshold_opacity).squeeze()
        else:
            prune_mask = candidate_prune_mask
        self.effective_increase_num_per_densify = [prune_mask[points_index].sum() // densification_times_per_epoch for points_index in self.points_index_in_each_polygon]
        self.effect_opacity = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        pre_point = self.get_num_points
        self.prune_points(prune_mask, True)
        print('Prune points by effective opacity: prune point {}, now point {}.'.format(pre_point - self.get_num_points, self.get_num_points))
        torch.cuda.empty_cache()

    def opa_and_scale_prune(self, min_opacity, extent):
        total_points = self.get_xyz.shape[0]
        candidate_prune_mask = (self.get_opacity < min_opacity).squeeze()
        big_points_ws = self.get_scaling.max(dim=1).values > 10 * extent
        if len(self.house_polygon) == 1:
            big_points_ws[self.points_index_in_house_polygon[-1]] = False

        big_points_ws2 = self.get_scaling.max(dim=1).values > 15
        big_points_ws = torch.logical_or(big_points_ws, big_points_ws2)


        prune_mask = torch.logical_or(candidate_prune_mask, big_points_ws)
        pre_point = self.get_num_points
        self.prune_points(prune_mask, True)
        print('Prune points by normal opacity and big points: global prune point {}.'.format(pre_point - self.get_num_points, self.get_num_points))
        torch.cuda.empty_cache()

    def densify(self, substage_k):
        self.dense_stage_time += 1
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        abs_grads = self.abs_xyz_gradient_accum / self.denom
        abs_grads[abs_grads.isnan()] = 0.0

        # 新版稠密化，反算不同阶段的梯度翻倍
        abs_grads_ResGS = torch.where(self.relative_fineness >= substage_k, abs_grads, abs_grads * torch.pow(2.0, (substage_k - self.relative_fineness) / 3.0))
        grad_temp = torch.norm(abs_grads_ResGS, dim=-1).clone()
        pre_point = self.get_num_points
        self.densify_and_split(abs_grads_ResGS)
        print('Densify res split: split increase point {}, now point {}.'.format(self.get_num_points-pre_point, self.get_num_points))
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, pixels, view_alpha):
        view_alpha = view_alpha.view(-1, 1)
        pixels = pixels.view(-1, 1)
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.abs_xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.denom[update_filter] += pixels[update_filter]

        safe_div = view_alpha[update_filter] / torch.where(pixels[update_filter] != 0, pixels[update_filter], torch.ones_like(pixels[update_filter]))
        safe_div = torch.where(pixels[update_filter] != 0, safe_div, torch.zeros_like(safe_div))
        self.effect_opacity[update_filter] = torch.max(self.effect_opacity[update_filter], safe_div)
        self.pivot[update_filter] = torch.where(pixels[update_filter] != 0, torch.ones_like(pixels[update_filter]), torch.zeros_like(pixels[update_filter])).float()
