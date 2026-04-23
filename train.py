import os
import shutil
import torch
from utils.loss_utils import l1_loss, ssim, l1_loss_mask
from gaussian_renderer import render, network_gui
import sys
from pathlib import Path
from scene import GaussianModel
from scene.scene_dataset import BackgroundLoader, Scene_dataset
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import tools.logging as log
from scene.BilateraIGrid.BilateralGrid import BilateralGridOptimizer
from lpipsPyTorch import lpips

TRAIN_CODE_ITEMS = (
    "train.py",
    "train_base.sh",
    "train_expo.sh",
    "train_fusion.sh",
    "render_single.py",
    "render.sh",
    "arguments",
    "gaussian_renderer",
    "scene",
    "utils",
    "tools",
    "lpipsPyTorch",
)


def backup_training_code(model_path):
    backup_dir = Path(model_path) / "train_code"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for item in TRAIN_CODE_ITEMS:
        src = Path(item)
        dst = backup_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy2(src, dst)


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")
    

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # backup main code
    backup_training_code(dataset.model_path)
    log_ = log.LoggerWriter(out_dir=dataset.model_path)
    log_.stdout_to_log_file()

    gaussians = GaussianModel(dataset, opt.optimizer_type, dataset.use_sh_embedding)
    scene = Scene_dataset(dataset, gaussians, init_clamp_scale_rate=opt.init_clamp_scale_rate, init_delete_scale_rate=opt.init_delete_scale_rate)
    points_in_each_polygon_init = gaussians.points_in_each_polygon
    dataset_len = scene.__len__()
    scene.set_stride_L(dataset_len)
    loader = BackgroundLoader(scene, buffer_size=100, num_workers=12)
    # 自适应参数设置
    opt.iterations = opt.epochs * dataset_len 
    stride_iterations = int(opt.iterations / 3)
    saving_iterations.append(dataset_len//10)
    testing_iterations.append(dataset_len)
    saving_iterations.append(dataset_len)
    for i in range(0, opt.iterations, stride_iterations):
        if i > 10:
            saving_iterations.append(i)
            testing_iterations.append(i)
    saving_iterations.append(opt.iterations)
    testing_iterations.append(opt.iterations)
    opt.position_lr_max_steps = opt.iterations
    opt.densify_from_iter = dataset_len # 第1个epoch后开始稠密化和删点
    opt.densification_interval = 100  
    densification_times_per_epoch = dataset_len // opt.densification_interval
    opt.shdgree_interval = dataset_len  # 每隔1个epoch增加一层sh系数，直到max结束
    opt.densify_until_iter = int(opt.densify_until_rate * opt.epochs) * dataset_len # 训练到densify_until_rate=65%以后时不再稠密化和删点
    opt.densification_times = int((opt.densify_until_iter - opt.densify_from_iter) // opt.densification_interval) + 1 # 预计执行的稠密化次数
    opt.exposure_lr_delay_steps = dataset_len ## 在1个epoch时进行warm up
    opt.bilateral_grid_warmup = dataset_len   ## 在1个epoch时进行warm up
    # 给objects设置默认训练超参
    for i in range(len(gaussians.objects)):
        # 统一原则：json里配置优先，如果json里没有配置，使用init.py中的设置
        if "densify_points_num_from_init" not in gaussians.objects[i]:
            if i == (len(gaussians.objects) - 1):
                gaussians.objects[i]["densify_points_num_from_init"] = opt.densify_points_num_from_init_env
            else:
                gaussians.objects[i]["densify_points_num_from_init"] = opt.densify_points_num_from_init
        print("Polygon {}, name: {}, densify_points_num_from_init: {}".format(i, gaussians.objects[i]["name"], gaussians.objects[i]["densify_points_num_from_init"]))
        
        if "target_gaussian_num" not in gaussians.objects[i]:
            gaussians.objects[i]["target_gaussian_num"] = int(points_in_each_polygon_init[i] * gaussians.objects[i]["densify_points_num_from_init"])
        print("Polygon {}, name: {}, target_gaussian_num: {}".format(i, gaussians.objects[i]["name"], gaussians.objects[i]["target_gaussian_num"]))

        if "increase_gaussian_num_each_time" not in gaussians.objects[i]:
            gaussians.objects[i]["increase_gaussian_num_each_time"] = int((gaussians.objects[i]["target_gaussian_num"] - points_in_each_polygon_init[i]) // opt.densification_times)
            if gaussians.objects[i]["increase_gaussian_num_each_time"] < 0:
                gaussians.objects[i]["increase_gaussian_num_each_time"] = 0

        # freeze_gs为True里的高斯不会被更新梯度
        if "freeze_gs" not in gaussians.objects[i]:
            gaussians.objects[i]["freeze_gs"] = opt.freeze_gs
        print("Polygon {}, name: {}, freeze_gs: {}".format(i, gaussians.objects[i]["name"], gaussians.objects[i]["freeze_gs"]))

        # freeze_densify为True里的高斯不会执行稠密化，但仍然会参与梯度排序（为了避免非freeze部分稠密过多）
        if "freeze_densify" not in gaussians.objects[i]:
            gaussians.objects[i]["freeze_densify"] = opt.freeze_gs
        print("Polygon {}, name: {}, freeze_densify: {}".format(i, gaussians.objects[i]["name"], gaussians.objects[i]["freeze_densify"]))

        if "absolutely_increase_num_control" not in gaussians.objects[i]:
            gaussians.objects[i]["absolutely_increase_num_control"] = opt.absolutely_increase_num_control
            if i == (len(gaussians.objects) - 1): # 当前策略: 环境polygon强制设置成软上限稠密化，只设置最大值，不要求一定达到
                gaussians.objects[i]["absolutely_increase_num_control"] = False
        print("Polygon {}, name: {}, absolutely_increase_num_control: {}".format(i, gaussians.objects[i]["name"], gaussians.objects[i]["absolutely_increase_num_control"]))

    gaussians.densification_times = opt.densification_times
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    stage1_start_iteration = opt.densify_from_iter # 第一子阶段从第一个epoch结束开始
    stage2_start_iteration = stage1_start_iteration + int((opt.densify_until_rate * opt.epochs - 1) * 0.1) * dataset_len # 第一子阶段持续总体稠密化阶段的10%
    stage3_start_iteration = stage2_start_iteration + int((opt.densify_until_rate * opt.epochs - 1) * 0.2) * dataset_len # 第二子阶段持续总体稠密化阶段的20%，剩下是第三子阶段

    print("opt.iterations : {}".format(opt.iterations))
    print("opt.densify_from_iter : {}".format(opt.densify_from_iter))
    print("opt.shdgree_interval : {}".format(opt.shdgree_interval))
    print("opt.densification_interval : {}".format(opt.densification_interval))
    print("opt.densify_until_iter : {}".format(opt.densify_until_iter))
    print("opt.densification_times : {}".format(opt.densification_times))
    print("testing_iterations : {}".format(testing_iterations))
    print("saving_iterations : {}".format(saving_iterations))
    print("stage1_start_iteration : {}".format(stage1_start_iteration))
    print("stage2_start_iteration : {}".format(stage2_start_iteration))
    print("stage3_start_iteration : {}".format(stage3_start_iteration))
    
    print("optimizer_type : {}".format(opt.optimizer_type))
    print("SPARSE_ADAM_AVAILABLE : {}".format(SPARSE_ADAM_AVAILABLE))
    print("use_sparse_adam : {}".format(use_sparse_adam))
    print("use_sh_embedding : {}".format(dataset.use_sh_embedding))
    print("use_bilateral_grid : {}".format(dataset.use_bilateral_grid))
    print("train_test_exp : {}".format(dataset.train_test_exp))
    
    val_scene = Scene_dataset(dataset, gaussians, init_clamp_scale_rate=opt.init_clamp_scale_rate, init_delete_scale_rate=opt.init_delete_scale_rate,val_set=True, train_scene=scene)
    val_dataset_len = val_scene.__len__()
    val_loader = BackgroundLoader(val_scene, buffer_size=val_dataset_len, num_workers=1, is_val_set =True)
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("random_background : {}".format(opt.random_background))
    print("background : {}".format(background))

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    first_iter += 1
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    global_iteration = 0
    substage_k = 1
    first_flag = True
    loss_presison = 2 / opt.iterations
    freq_max = -1.0
    freq_min = float('inf')
    assert dataset.use_bilateral_grid * dataset.train_test_exp == False, "use_bilateral_grid and train_test_exp cannot be both True"

    if dataset.use_sh_embedding:
        optimizer_app = None
        if gaussians.appearance_app is not None:
            optimizer_app = torch.optim.Adam(
                gaussians.appearance_app.parameters(), lr=1e-5
            )
        
        optimizer_mlp = None
        if gaussians.color_predictor is not None:
            optimizer_mlp = torch.optim.Adam(
                gaussians.color_predictor.parameters(), lr=0.2*1e-4
            )

    while global_iteration <= opt.iterations:
        global_iteration +=1
        iter_start.record()
        gaussians.update_learning_rate(global_iteration)
        if global_iteration % opt.shdgree_interval == 0:
            gaussians.oneupSHdegree()
        viewpoint_cam = next(loader)
        # Render
        if (global_iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, use_sh_embedding=dataset.use_sh_embedding)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if dataset.use_bilateral_grid and global_iteration == opt.bilateral_grid_warmup:
            print("初始化双边网格...")
            bilateral_grid_optimizer = BilateralGridOptimizer(
                image_num=dataset_len,  # 使用数据集中的图像数量
                grid_shape=opt.bilateral_grid_shape,  # 可配置的网格大小
                initial_lr=opt.bilateral_grid_lr,  # 可配置的学习率
                warmup_iters=opt.bilateral_grid_warmup,
                end_iteration=opt.iterations,
                device="cuda"
            )
            print(f"双边网格初始化完成，网格形状: {bilateral_grid_optimizer.bilateral_grid_shape}")
        
        if  dataset.use_bilateral_grid and global_iteration >= opt.bilateral_grid_warmup:
            enhanced_image = bilateral_grid_optimizer.process_image(image, viewpoint_cam)
            image = enhanced_image.squeeze(0).permute(2, 0, 1)    # (H,W,3)转换为(3,H,W)
            
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.is_masked

        if mask is not None:
            mask = mask.cuda().bool()
            gt_image[mask] = image.detach()[mask]
        else:
            mask = torch.zeros_like(gt_image,  device="cuda", dtype=torch.bool)

        depth_mask = viewpoint_cam.depth_mask.cuda()
        freq_mask = viewpoint_cam.freq_hm.cuda()

        if global_iteration < opt.densify_from_iter:
            freq_mask_min_val = freq_mask.min()
            freq_mask_max_val = freq_mask.max()
            freq_min = min(freq_mask_min_val.item(), freq_min)
            freq_max = max(freq_mask_max_val.item(), freq_max)
            freq_mask = torch.zeros_like(freq_mask)
        else:
            freq_mask = (freq_mask - freq_min) / (freq_max - freq_min)
        high_freq_weight = 1.0 + global_iteration * loss_presison * 0.5
        low_freq_weight = 1.0 - global_iteration * loss_presison * 0.5
        Ll1 = l1_loss_mask(image, gt_image, depth_mask, low_freq_weight * (1 - freq_mask) + high_freq_weight * (freq_mask))

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        if dataset.use_bilateral_grid and global_iteration >= opt.bilateral_grid_warmup:
            tv_loss = bilateral_grid_optimizer.bil_grids.tv_loss()
            loss += opt.bilateral_grid_tv_weight * tv_loss

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(global_iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * (~mask[:1])).mean()
            Ll1depth = depth_l1_weight(global_iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        
        scaling_loss = torch.max(gaussians.get_scaling) 
        if scaling_loss > 10:
            loss += scaling_loss
        

        # 跳过前两轮，为了解决xyz不优化的问题
        if global_iteration <= 2:
            loss *= 0.
        
        if torch.isnan(loss) or torch.isinf(loss) or loss > 10000:
            print("\n[ITER {}] loss == {} error, skip ".format(global_iteration, loss.item()))
            continue

        if opt.only_train_expo:
            loss = 5 * Ll1

        loss.backward()
        
        # 更新各个polygon的高斯梯度学习率
        gaussians.apply_lr_mult(set_grad_zero=viewpoint_cam.is_test_view)

        iter_end.record()
        with torch.no_grad():
            # compute substage
            if global_iteration == int(stage1_start_iteration + (stage2_start_iteration - stage1_start_iteration) * 1.0 / 3.0):
                substage_k += 1 # 2
                print("\n[ITER {}] substage_k == {}".format(global_iteration, substage_k))
            elif global_iteration == int(stage1_start_iteration + (stage2_start_iteration - stage1_start_iteration) * 2.0 / 3.0):
                substage_k += 1 # 3
                print("\n[ITER {}] substage_k == {}".format(global_iteration, substage_k))
            elif global_iteration == int(stage2_start_iteration):
                substage_k += 1 # 4
                print("\n[ITER {}] substage_k == {}".format(global_iteration, substage_k))
            elif global_iteration == int(stage2_start_iteration + (stage3_start_iteration - stage2_start_iteration) * 1.0 / 3.0):
                substage_k += 1 # 5
                print("\n[ITER {}] substage_k == {}".format(global_iteration, substage_k))
            elif global_iteration == int(stage2_start_iteration + (stage3_start_iteration - stage2_start_iteration) * 2.0 / 3.0):
                substage_k += 1 # 6
                print("\n[ITER {}] substage_k == {}".format(global_iteration, substage_k))
            elif global_iteration == int(stage3_start_iteration):
                substage_k += 1 # 7
                print("\n[ITER {}] substage_k == {}".format(global_iteration, substage_k))
            elif global_iteration == int(stage3_start_iteration + (opt.densify_until_iter - stage3_start_iteration) * 1.0 / 3.0):
                substage_k += 1 # 8
                print("\n[ITER {}] substage_k == {}".format(global_iteration, substage_k))
            elif global_iteration == int(stage3_start_iteration + (opt.densify_until_iter - stage3_start_iteration) * 2.0 / 3.0):
                substage_k += 1 # 9
                print("\n[ITER {}] substage_k == {}".format(global_iteration, substage_k))

            # Progress bar
            psnr_for_log = (1.0 - opt.lambda_dssim) * Ll1.item()
            ssim_for_log = opt.lambda_dssim * (1.0 - ssim_value.item())

            loss_for_log = loss.item()
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if global_iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss_for_log:.{4}f}","01loss": f"{0:.{4}f}", "Ll1depth": f"{Ll1depth:.{4}f}","PSNR Loss": f"{psnr_for_log:.{4}f}","SSIM Loss": f"{ssim_for_log:.{4}f}", "Ema Loss": f"{ema_loss_for_log:.{7}f}", "cloud xyz": f"{len(gaussians.get_xyz):.{1}f}"})
                progress_bar.update(10)
            if global_iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, global_iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, loader, val_loader, gaussians, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp))
            if (global_iteration in saving_iterations):
                print("\n[ITER {}] points_in_each_polygon : {}".format(global_iteration, gaussians.points_in_each_polygon))
                for i in range(len(gaussians.points_in_each_polygon)):
                    print("Polygon {}, name: {},  points_in_polygon: {}".format(i, gaussians.objects[i]["name"], gaussians.points_in_each_polygon[i]))
                
                print("\n[ITER {}] Saving Gaussians".format(global_iteration))
                scene.save(global_iteration)
                
            # Densification
            if global_iteration < opt.densify_until_iter and opt.only_train_expo == False:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["view_times"], render_pkg["view_alpha"])
                if global_iteration >= opt.densify_from_iter and global_iteration % opt.densification_interval == 0:
                    effect_opacity_prune_flag = False
                    if global_iteration % (densification_times_per_epoch * opt.densification_interval) == 0:
                        effect_opacity_prune_flag = True
                    if first_flag:
                        effect_opacity_prune_flag = True
                        first_flag = False
                    if effect_opacity_prune_flag:
                        gaussians.effective_prune(densification_times_per_epoch)
                    
                    if not effect_opacity_prune_flag:
                        gaussians.opa_and_scale_prune(opt.opactiy_thred, scene.cameras_extent)
                    gaussians.densify(substage_k)

                    for i in range(len(gaussians.objects)):
                        if gaussians.objects[i]["increase_gaussian_num_each_time"] == 0:
                            gaussians.objects[i]["increase_gaussian_num_each_time"] = int((gaussians.objects[i]["target_gaussian_num"] - points_in_each_polygon_init[i]) // (opt.densification_times + 1 - gaussians.dense_stage_time))
                            if gaussians.objects[i]["increase_gaussian_num_each_time"] < 0:
                                gaussians.objects[i]["increase_gaussian_num_each_time"] = 0
                
                # 在第二轮时初始化一下，解决xyz不优化的问题
                if global_iteration == 2:
                    gaussians.prune_points_init()
            
            if global_iteration > opt.densify_until_iter and global_iteration % opt.densification_interval == 0:
                gaussians.opa_and_scale_prune(opt.opactiy_thred, scene.cameras_extent)

            # Optimizer step
            if global_iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                
                # 双边网格优化步骤
                if  dataset.use_bilateral_grid and global_iteration >= opt.bilateral_grid_warmup:
                    bilateral_grid_optimizer.optimization_step()
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if dataset.use_sh_embedding:
                    optimizer_app.step()
                    optimizer_mlp.step()
                    optimizer_app.zero_grad()
                    optimizer_mlp.zero_grad()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, train_scene_loader, val_scene_loader, gaussians, renderFunc, renderArgs, renderArgs_for_val):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'train', 'cameras' : train_scene_loader, 'index': range(10, train_scene_loader.dataset.__len__(), 100)}, 
                              {'name': 'test', 'cameras' : val_scene_loader, 'index': range(val_scene_loader.dataset.__len__())})
        for config in validation_configs:
            if config['cameras'] and len(config['index']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_value_test = 0.0
                lpips_test = 0.0
                for idx in config['index']:
                    if config['name'] == "train":
                        viewpoint = config['cameras'].dataset.__getitem__(idx)
                        image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
                        
                    else:
                        viewpoint = next(val_scene_loader)
                        image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs_for_val)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    
                    mask = viewpoint.is_masked
                    if mask is not None:
                        mask = mask.cuda()
                        gt_image[mask] = image.detach()[mask]
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if FUSED_SSIM_AVAILABLE:
                        ssim_value_test += fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                    else:
                        ssim_value_test += ssim(image, gt_image)
                    lpips_test += lpips(image, gt_image, net_type='vgg').item()
                    del image,gt_image
                    torch.cuda.empty_cache()
                psnr_test /= len(list(config['index']))
                l1_test /= len(list(config['index']))  
                ssim_value_test /= len(list(config['index']))
                lpips_test /= len(list(config['index']))
                print("\n[ITER {}] Evaluating {}: L1 {} SSIM {} PSNR {} LPIPS {} points {} test images {}".format(iteration, config['name'], l1_test, ssim_value_test, psnr_test, lpips_test, gaussians.get_xyz.shape[0], len(list(config['index'])) ))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_value_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", train_scene_loader.dataset.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', train_scene_loader.dataset.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=True)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
