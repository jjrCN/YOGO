import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


WARNED_Mask = False


class CameraInfo(NamedTuple):
    uid: int
    image_id_mark: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    mask_path: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    fx: np.array
    fy: np.array
    cx: np.array
    cy: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, masks_folder, depths_folder, test_cam_names_list, sensor_mod, bad_image, refined_pose):
    cam_infos = []
    mask_count = 0
    image_count = 0

    if refined_pose and os.path.exists(refined_pose):
        split_name = refined_pose.split("/")[-3]
        print("images_folder:",images_folder)
        print("################# use adjusted_cameras.json {}".format(split_name))
        with open(refined_pose, 'r') as file:
            adjusted_cams = json.load(file)
    else:
        adjusted_cams = None
        print("################# use cam_extrinsics.txt")

    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

         
        if sensor_mod == 'only_osmo':
            if 'cam' not in extr.name or 'pano' in extr.name:
                continue
        elif sensor_mod == 'only_x5':
            if 'pano' not in extr.name:
                continue
        elif sensor_mod == "only_s20":
            if 'pano' in extr.name or 'cam' in extr.name:
                continue
        elif sensor_mod == "fusion":
            if extr.name in bad_image:
                continue
        else:
            pass

        image_id_mark = int(extr.id)

        uid = intr.id

        if adjusted_cams is not None:
            for cam_data in adjusted_cams:
                if cam_data['img_name'] == extr.name:
                    position = np.array(cam_data['position'], dtype=np.float64)
                    rotation = np.array(cam_data['rotation'], dtype=np.float64)

                    W2C = np.eye(4, dtype=np.float64)
                    W2C[:3, :3] = rotation
                    W2C[:3, 3] = position

                    Rt = np.linalg.inv(W2C)
                    R = Rt[:3, :3].T
                    T = Rt[:3, 3]

                    break
        else:
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = focal_length_x
            cx = intr.params[1]
            cy = intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        image_count += 1
        mask = None
        mask_path = None
        if masks_folder is not None and masks_folder != "":
            possible_mask_path = os.path.join(masks_folder, "{}.png".format(os.path.splitext(extr.name)[0]))
            
            if os.path.exists(possible_mask_path):
                mask_path = possible_mask_path
                mask_count += 1
            else:
                global WARNED_Mask
                if not WARNED_Mask:
                    print("mask_path: {} is not found".format(possible_mask_path))
                    print("This message will print only once")
                    WARNED_Mask = True
                
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, image_id_mark=image_id_mark, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, mask_path=mask_path, depth_path=depth_path,
                              fx=focal_length_x, fy=focal_length_y, cx=cx, cy=cy,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    sys.stdout.write("Read {} images".format(image_count))
    if masks_folder != "":
        sys.stdout.write('\n')
        sys.stdout.write("Read {} masks".format(mask_count))
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, split_val_data, llffhold=8, masks=None, sparse_dir="",points3D_dir="", sensor_mod="", bad_image_list="", refined_pose=""):

    if sparse_dir != "":
        print("use sparse dir:",sparse_dir)
        try:
            cameras_extrinsic_file = os.path.join(sparse_dir, "0", "images.bin")
            cameras_intrinsic_file = os.path.join(sparse_dir, "0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(sparse_dir, "0", "images.txt")
            cameras_intrinsic_file = os.path.join(sparse_dir, "0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    else:
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    bad_image = {}
    if bad_image_list != "":
        with open(bad_image_list, 'r', encoding='utf-8') as f:
            bad_image = {line.strip() for line in f if line.strip()}  # 用 set 推导式去重

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    val_list_path =  os.path.join(path, "val_list.txt")
    if os.path.exists(val_list_path):
        print("Load val image from : ", val_list_path)
        with open(val_list_path, 'r') as file:
            test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        masks_folder=masks,
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list, sensor_mod=sensor_mod, bad_image=bad_image, refined_pose=refined_pose)
    
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    if len(test_cam_names_list) > 0:
        train_cam_infos = [c for c in cam_infos]
        test_cam_infos = [c for c in cam_infos if c.image_name in test_cam_names_list]
    else:
        val_list = []
        if split_val_data:
            val_list = [idx for idx in range(10, len(cam_infos), 100)]

        print(f"val_list_path : {val_list_path} not exist. Train all images.")
        train_cam_infos = [c for c in cam_infos]
        test_cam_infos = [c for c in cam_infos if c.image_id_mark in val_list]
        

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    if points3D_dir!="":
        print("use points3D dir:",points3D_dir)
        ply_path = points3D_dir
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    try:
        positions2, colors2, normals2 = fetchPly(ply_path)
        print("Load lidar points : ", positions2.shape[0])
        pcd = BasicPointCloud(points=positions2, colors=colors2, normals=normals2)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            fx = fov2focal(FovX, image.size[0])
            fy = fov2focal(FovY, image.size[1])
            cam_infos.append(CameraInfo(uid=idx, image_id_mark=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            depth_params=None, image_path=image_path, image_name=image_name,
                            mask_path=None, depth_path=depth_path, width=image.size[0],
                            height=image.size[1], is_test=is_test, fx=fx, fy=fy,
                            cx=image.size[0] / 2.0, cy=image.size[1] / 2.0))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
