import torch
import sys
from datetime import datetime
import numpy as np
import random
import json

def get_polygons(polygon_path):
    polygons = []
    polygon_house = []
    with open(polygon_path, "r", encoding="utf-8") as f:
        polygon_json = json.load(f)
        objects = polygon_json["objects"]
        objects_result = []
        for object_item in objects:
            vertices = object_item["vertices"]
            p0 = vertices[0]
            p1 = vertices[1]
            p2 = vertices[2]
            p3 = vertices[3]
            p4 = vertices[4]
            p5 = vertices[5]
            p6 = vertices[6]
            p7 = vertices[7]
            cube_faces = torch.tensor([
                [[p0["x"],p0["y"],p0["z"]],[p1["x"],p1["y"],p1["z"]],[p2["x"],p2["y"],p2["z"]],[p3["x"],p3["y"],p3["z"]]],  # bottom
                [[p4["x"],p4["y"],p4["z"]],[p5["x"],p5["y"],p5["z"]],[p6["x"],p6["y"],p6["z"]],[p7["x"],p7["y"],p7["z"]]],  # top
                [[p4["x"],p4["y"],p4["z"]],[p5["x"],p5["y"],p5["z"]],[p1["x"],p1["y"],p1["z"]],[p0["x"],p0["y"],p0["z"]]],  # front
                [[p7["x"],p7["y"],p7["z"]],[p6["x"],p6["y"],p6["z"]],[p2["x"],p2["y"],p2["z"]],[p3["x"],p3["y"],p3["z"]]],  # back
                [[p7["x"],p7["y"],p7["z"]],[p4["x"],p4["y"],p4["z"]],[p0["x"],p0["y"],p0["z"]],[p3["x"],p3["y"],p3["z"]]],  # left
                [[p6["x"],p6["y"],p6["z"]],[p5["x"],p5["y"],p5["z"]],[p1["x"],p1["y"],p1["z"]],[p2["x"],p2["y"],p2["z"]]],  # right
            ], dtype=torch.float32, device="cuda")
            if object_item["name"] == "house":
                polygon_house.append(cube_faces)
            else:
                polygons.append(cube_faces)
                objects_result.append(object_item)
    return polygons, objects_result, polygon_house

def compute_points_in_each_polygon_torch(polygons, points, eps=1e-9):
    """
    输入:
      polygons: list of torch.Tensor, 每个 (6,4,3)
      points: torch.Tensor (P,3)
    返回:
      points_in_each_polygon: list[int], 最后一项是 outside_count
      points_index_in_each_polygon: list[torch.BoolTensor (P,)]
    说明:
      完全等价于 numpy 版本的 compute_points_in_each_polygon
      - 使用 torch 实现
      - 支持 GPU、梯度
    """
    device = points.device
    dtype = points.dtype
    P = points.shape[0]

    points_in_each_polygon = []
    points_index_in_each_polygon = []
    inside_total = torch.zeros(P, dtype=torch.bool, device=device)

    # 遍历每个 poly (6,4,3)
    for poly in polygons:
        # 保证为 tensor
        if not isinstance(poly, torch.Tensor):
            poly = torch.tensor(poly, dtype=dtype, device=device)
        else:
            poly = poly.to(device=device, dtype=dtype)

        all_vertices = poly.reshape(-1, 3)  # (24,3)
        centroid = all_vertices.mean(dim=0)  # (3,)

        # 用前三个点计算法向量
        v0 = poly[:, 0, :]  # (6,3)
        v1 = poly[:, 1, :]
        v2 = poly[:, 2, :]
        normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (6,3)
        norms = torch.linalg.norm(normals, dim=1, keepdim=True)
        normals = normals / (norms + 1e-12)

        # 质心校正
        dots_centroid = torch.sum(normals * (centroid[None, :] - v0), dim=1)  # (6,)
        flip_mask = dots_centroid > 0
        normals[flip_mask] *= -1.0

        # 计算点对面的符号
        normals_T = normals.T  # (3,6)
        pts_dot = torch.matmul(points, normals_T)  # (P,6)
        v0_dot = torch.sum(v0 * normals, dim=1)  # (6,)
        dots = pts_dot - v0_dot[None, :]  # (P,6)

        # 点在所有面的内侧
        inside_mask = torch.all(dots <= eps, dim=1)  # (P,)
        inside_total |= inside_mask
        points_in_each_polygon.append(int(inside_mask.sum().item()))
        points_index_in_each_polygon.append(inside_mask.detach())

    # outside
    outside_mask = ~inside_total
    points_in_each_polygon.append(int(outside_mask.sum().item()))
    points_index_in_each_polygon.append(outside_mask.detach())

    return points_in_each_polygon, points_index_in_each_polygon

# def compute_points_in_each_polygon_torch(polygons, points):
#     """
#     输入:
#       polygons: list of np.ndarray (6,4,3) from get_polygons
#       points: np.ndarray (P,3)
#     返回:
#       points_in_each_polygon: list of counts, 最后一项是 outside_count
#       points_index_in_each_polygon: list of boolean masks (P,) 对每个 polygon 一个 mask，最后一项为 outside mask
#     说明:
#       - 使用向量化矩阵运算替代每个面逐点循环
#       - 精度与原逻辑一致（用前三点计算法向并质心修正）
#     """
#     P = points.shape[0]
#     points_in_each_polygon = []
#     points_index_in_each_polygon = []
#     inside_total = np.zeros(P, dtype=bool)

#     # 保证数据连续性（提高 numpy 运算性能）
#     points = np.ascontiguousarray(points, dtype=np.float64)

#     for poly in polygons:
#         # poly: (6,4,3)
#         # 所有面顶点展开 (6,4,3) already
#         # 保证 poly 是 numpy array
#         poly = np.array(poly, dtype=np.float32)
#         all_vertices = poly.reshape(-1, 3)  # (24,3)
#         centroid = all_vertices.mean(axis=0)  # (3,)

#         # 使用前三个顶点计算法向量 (6,3)
#         v0 = poly[:, 0, :]  # (6,3)
#         v1 = poly[:, 1, :]
#         v2 = poly[:, 2, :]
#         normals = np.cross(v1 - v0, v2 - v0)  # (6,3)
#         norms = np.linalg.norm(normals, axis=1, keepdims=True)
#         normals = normals / (norms + 1e-12)

#         # 质心校正：如果 normal 与 (centroid - v0) 点乘 >0，则翻转
#         dots_centroid = np.sum(normals * (centroid[None, :] - v0), axis=1)  # (6,)
#         flip_mask = dots_centroid > 0
#         normals[flip_mask] *= -1.0

#         # 现在我们需要对每个点计算 (points - v0)·normal  对所有面
#         # 点对面的点积矩阵: D[i,j] = (points[i] - v0[j])·normals[j]
#         # = points @ normals.T  - v0 @ normals.T
#         normals_T = normals.T  # (3,6)
#         pts_dot = points.dot(normals_T)  # (P,6)
#         # v0_dot = v0.dot(normals_T)      # (6,) -> broadcast
#         v0_dot = np.sum(v0 * normals, axis=1) 
#         dots = pts_dot - v0_dot[None, :]  # (P,6)

#         # inside: 点在所有 6 个面的内侧：dots <= eps 对所有面都成立
#         eps = 1e-9
#         inside_mask = np.all(dots <= eps, axis=1)  # (P,)
#         inside_total |= inside_mask
#         points_in_each_polygon.append(int(inside_mask.sum()))
#         points_index_in_each_polygon.append(inside_mask)

#     # outside
#     outside_mask = ~inside_total
#     points_in_each_polygon.append(int(outside_mask.sum()))
#     points_index_in_each_polygon.append(outside_mask)

#     return points_in_each_polygon, points_index_in_each_polygon

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
