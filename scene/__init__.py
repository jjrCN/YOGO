import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], max_data_num=300, init_clamp_scale_rate=1.0, init_delete_scale_rate=1.0):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.max_data_num = max_data_num
        self.max_data_num_test = max_data_num

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.split_val_data, masks=args.masks)
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

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        print("scene_info.train_cameras : ", len(scene_info.train_cameras))
        print("scene_info.test_cameras : ", len(scene_info.test_cameras))
        self.scene_info_train_cameras = scene_info.train_cameras
        self.scene_info_test_cameras = scene_info.test_cameras
        self.cur_train_index = 0
        self.cur_test_index = 0
        self.scene_info_is_nerf_synthetic = scene_info.is_nerf_synthetic

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            before_index = self.cur_train_index
            if self.max_data_num == None or self.max_data_num > len(self.scene_info_train_cameras):
                self.max_data_num = len(self.scene_info_train_cameras)
            # self.max_data_num = 10
            self.train_cameras[resolution_scale], self.cur_train_index = cameraList_from_camInfos(self.scene_info_train_cameras, resolution_scale, self.max_data_num, self.cur_train_index, args, scene_info.is_nerf_synthetic, False)
            print(" ")
            print("Load Training Cameras from index {}-{}".format(before_index, self.cur_train_index))

            if self.max_data_num_test == None or self.max_data_num_test > len(self.scene_info_test_cameras):
                self.max_data_num_test = len(self.scene_info_test_cameras)
            before_index = self.cur_test_index
            self.test_cameras[resolution_scale], self.cur_test_index = cameraList_from_camInfos(self.scene_info_test_cameras, resolution_scale, self.max_data_num_test, self.cur_test_index, args, scene_info.is_nerf_synthetic, True)
            print("Load Test Cameras from index {}-{}".format(before_index, self.cur_test_index))

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.pretrained_exposure_path)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent, init_clamp_scale_rate, init_delete_scale_rate, args.pretrained_exposure_path)
            

    def reload(self, args : ModelParams, resolution_scales=[1.0]):
        for resolution_scale in resolution_scales:
            before_index = self.cur_train_index
            self.train_cameras[resolution_scale], self.cur_train_index = cameraList_from_camInfos(self.scene_info_train_cameras, resolution_scale, self.max_data_num, self.cur_train_index, args, self.scene_info_is_nerf_synthetic, False)
            print("ReLoad Training Cameras from index {}-{}, train cameras len : {}".format(before_index, self.cur_train_index, len(self.train_cameras[resolution_scale])))

            before_index = self.cur_test_index
            self.test_cameras[resolution_scale], self.cur_test_index = cameraList_from_camInfos(self.scene_info_test_cameras, resolution_scale, self.max_data_num_test, self.cur_test_index, args, self.scene_info_is_nerf_synthetic, True)
            print("ReLoad Test Cameras from index {}-{}, test cameras len : {}".format(before_index, self.cur_test_index, len(self.test_cameras[resolution_scale])))


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
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
