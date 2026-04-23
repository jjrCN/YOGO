from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = Image.open(cam_info.image_path)

    # Optional input sharpening before converting the image to a training tensor.
    if args.sharpened:
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        blurred = cv2.GaussianBlur(image_bgr, (0, 0), 1.0)
        sharpened_bgr = cv2.addWeighted(image_bgr, 1 + args.sharpened, blurred, -args.sharpened, 0)
        sharpened_bgr = np.clip(sharpened_bgr, 0, 255).astype(np.uint8)
        sharpened_rgb = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(sharpened_rgb)

    if cam_info.mask_path:
        mask = Image.open(cam_info.mask_path)
        # print("loaded mask:", mask)
    else:
        mask = None

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, mask=mask, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test, cam_info=cam_info)

# def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
#     camera_list = []

#     for id, c in enumerate(tqdm.tqdm(cam_infos)):
#         camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

#     return camera_list

def cameraList_from_camInfos(cam_infos, resolution_scale, max_data_num, cur_index, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []
    count = 0
    for id, c in enumerate(cam_infos):
        if id >= cur_index and id < cur_index + max_data_num:
            camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))
            count += 1
            if count == max_data_num:
                break
        if count % 1000 == 0:
            print('Already loaded cameras {0}/{1}'.format(count, len(cam_infos)))

    if count == max_data_num:
        cur_index = cur_index + max_data_num
    else:
        rest_count = max_data_num - count
        for id, c in enumerate(cam_infos):
            if id >= 0 and id < rest_count:
                camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))
                count += 1
                if count == max_data_num:
                    cur_index = rest_count
                    break

    return camera_list, cur_index

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
