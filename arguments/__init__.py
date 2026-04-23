from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.masks = ""
        self._depths = ""
        self._resolution = 1
        self.sharpened = None
        self._white_background = False
        self.train_test_exp = False
        self.use_bilateral_grid = False
        self.use_sh_embedding = False
        self.pretrained_exposure_path = "exposure.json"
        self.split_val_data = True
        self.data_device = "cpu"
        self.eval = False
        self.sparse_dir = ""
        self.points3D_dir = ""
        self.base_gaussian_dir = ""
        self.bad_image_list = ""
        self.refined_pose = ""
        self.sensor_mod = "only_x5" # only_x5 only_s20 only_osmo fusion
        self.polygon_3d_list_path = ""

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")



# 210000
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 210_000
        self.epochs = 15
        self.position_lr_init = 0.000016
        self.position_lr_final = 0.000004
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 210_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001 #调小
        self.exposure_lr_init = 0.002
        self.exposure_lr_final = 0.00003
        self.exposure_lr_delay_mult = 0.1
        self.percent_dense = 0.001
        self.lambda_dssim = 0.8

        self.densification_interval = 100
        self.opacity_reset_interval = 21000
        self.densify_from_iter = 9984
        self.densify_until_iter = 105_000
        self.densify_until_rate = 0.96
        self.densify_grad_threshold = 0.0002
        self.densify_points_num_from_init = 1.0
        self.densify_points_num_from_init_env = 1.0
        
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        self.shdgree_interval = 9984
        self.opactiy_thred = 0.005
        self.init_clamp_scale_rate = 1.0
        self.init_delete_scale_rate = 1.0 # do not use, not write yet

        self.lambda_zero_one = 1e-2
        self.lambda_datten = 0.0

        self.bilateral_grid_shape = (16, 16, 1)
        self.bilateral_grid_lr = 2e-3
        self.bilateral_grid_tv_weight = 10.0

        self.only_train_expo = False
        self.absolutely_increase_num_control = False
        self.freeze_gs = False
        self.freeze_densify = False

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
