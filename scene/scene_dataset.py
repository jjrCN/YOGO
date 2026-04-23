import os
import torch
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import camera_to_JSON
import time
import threading
import queue
from torch.utils.data import Dataset
from utils.camera_utils import loadCam
from utils.general_utils import compute_points_in_each_polygon_torch
from collections import deque

class Scene_dataset(Dataset):

    gaussians : GaussianModel
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, init_clamp_scale_rate=1.0, init_delete_scale_rate=1.0, val_set = False, train_scene=None, stride_L=10000):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.global_data_iteration = 0
        self.stride_L = stride_L
        self.model_path = args.model_path
        self.loaded_iter = None
        if args.base_gaussian_dir:
            load_iteration = 10000000000
        self.gaussians = gaussians
        self.save_densify_result = gaussians.save_densify_result
        self.args = args
        if val_set:
            self.scene_info_cameras = train_scene.scene_info_test_cameras
            self.scene_info_is_nerf_synthetic = train_scene.scene_info_is_nerf_synthetic
            self.mode = "test"

        else:
            if load_iteration:
                if load_iteration == -1:
                    self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                else:
                    self.loaded_iter = load_iteration
                # print("Loading trained model at iteration {}".format(self.loaded_iter))

            self.train_cameras = {}
            self.test_cameras = {}

            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, \
                args.train_test_exp, args.split_val_data, masks=args.masks, sparse_dir=args.sparse_dir, points3D_dir=args.points3D_dir, sensor_mod=args.sensor_mod, bad_image_list=args.bad_image_list, refined_pose=args.refined_pose)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
            else:
                assert False, "Could not recognize scene type!"

        
            if not self.loaded_iter:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                    json.dump(json_cams, file)

            print("scene_info.train_cameras : ", len(scene_info.train_cameras))
            print("scene_info.test_cameras : ", len(scene_info.test_cameras))
            self.scene_info_cameras = scene_info.train_cameras
            self.scene_info_test_cameras = scene_info.test_cameras
            if len(scene_info.test_cameras) == 0:
                self.scene_info_test_cameras = scene_info.train_cameras[:10]

            if shuffle:
                random.shuffle(self.scene_info_cameras)
            self.cur_train_index = 0
            self.cur_test_index = 0
            self.scene_info_is_nerf_synthetic = scene_info.is_nerf_synthetic

            self.cameras_extent = 5 #scene_info.nerf_normalization["radius"]
            self.mode = "train"

            new_pcd = scene_info.point_cloud
            if self.loaded_iter:
                self.gaussians.load_ply(args.base_gaussian_dir, scene_info.train_cameras, args.pretrained_exposure_path)
                print("Loading trained model from path: {}".format(args.base_gaussian_dir))
            else:
                self.gaussians.create_from_pcd(new_pcd, scene_info.train_cameras, self.cameras_extent, init_clamp_scale_rate, init_delete_scale_rate, args.pretrained_exposure_path)
            self.gaussians.points_in_each_polygon, self.gaussians.points_index_in_each_polygon = compute_points_in_each_polygon_torch(self.gaussians.polygons, torch.from_numpy(new_pcd.points).cuda())
            self.gaussians.points_in_house_polygon, self.gaussians.points_index_in_house_polygon = compute_points_in_each_polygon_torch(self.gaussians.house_polygon, torch.from_numpy(new_pcd.points).cuda())
            
            print("init_points_in_each_polygon : ", self.gaussians.points_in_each_polygon)
            for i in range(len(self.gaussians.points_in_each_polygon)):
                print("Polygon {}, name: {},  points_in_polygon: {}".format(i, self.gaussians.objects[i]["name"], self.gaussians.points_in_each_polygon[i]))
            if len(self.gaussians.house_polygon) == 1:
                print("points in house: ", self.gaussians.points_in_house_polygon[0])
                print("points out of house: ", self.gaussians.points_in_house_polygon[-1])
            else:
                print("Not house polygon")

    def set_mode (self, mode): 
        self.mode = mode
    
    def __len__(self):
        return len(self.scene_info_cameras)
    
    def set_stride_L(self, stride_L):
        self.stride_L = stride_L

    def __getitem__(self, index):
                
        if self.mode=="train":
            is_test_dataset = False
        else:
            is_test_dataset = True
        cam_infos = self.scene_info_cameras
        
        cam_info = cam_infos[index]
        resolution_scale = 1.0
        is_nerf_synthetic = self.scene_info_is_nerf_synthetic
        cameras = loadCam(self.args, index, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset)
        return cameras    
                  

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_ply(os.path.join(self.model_path, "point_cloud/point_cloud.ply"))
        if self.save_densify_result:
            self.gaussians.save_pcd_result(os.path.join(point_cloud_path, "point_cloud_result.pcd"))
            self.gaussians.save_pcd_prune(os.path.join(point_cloud_path, "point_cloud_prune.pcd"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]






class BackgroundLoader:
    def __init__(self, dataset, buffer_size=100, num_workers=4, is_val_set=False):
        self.dataset = dataset
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.num_workers = num_workers
        self._stop_event = threading.Event()
        self.lock = threading.Lock()
        self.is_val_set = is_val_set
        
        # 启动后台线程
        self.workers = []
        len_dataset = self.dataset.__len__()
        self.worker_idxs = deque(range(len_dataset))
        for worker_id in range(num_workers):
            t = threading.Thread(target=self._preload ,daemon=True)
            t.start()
            self.workers.append(t)

    def reset_worker_idxs(self):
        with self.lock:
            len_dataset = self.dataset.__len__()
            print("reload workers")
            self.worker_idxs = list(range(len_dataset))
   
    def _preload(self):
        # print(self.worker_idxs)
        while not self._stop_event.is_set():
            try:
                with self.lock :
                    if not self.worker_idxs:
                        self.worker_idxs = deque(range(len(self.dataset)))
                    idx = self.worker_idxs.popleft()
                item = self.dataset[idx]
                self.buffer.put(item)
            except Exception as e:
                print(f"Error in preload: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer.empty():
            print('waiting...')
            time.sleep(30)
        elif self._stop_event.is_set():
            raise StopIteration
        return self.buffer.get(timeout=1)
    
    # def __iter__(self):
    #     while not self._stop_event.is_set() or not self.buffer.empty():
    #         try:
    #             yield self.buffer.get(timeout=1)
    #             self.buffer.task_done()
    #         except queue.Empty:
    #             break

    def shutdown(self):
        self._stop_event.set()
        for t in self.workers:
            t.join(timeout=5)
        self.buffer.queue.clear()

