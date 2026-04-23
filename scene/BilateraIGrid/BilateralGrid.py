import torch
import math
from torch.optim import Adam
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, ExponentialLR

from scene.BilateraIGrid.lib_bilagrid import (
    BilateralGrid,
    slice,
)


class BilateralGridOptimizer:
    def __init__(self, image_num=8000, grid_shape=(4, 4, 4), initial_lr=2e-3, 
                 warmup_iters=1000, end_iteration=350000, device="cuda"):
        """
        Initialize the bilateral grid optimizer with configurable parameters.
        
        Args:
            image_num: Number of images in the dataset (default: 8000)
            grid_shape: Tuple of (grid_X, grid_Y, grid_W) dimensions (default: (4, 4, 4))
            initial_lr: Initial learning rate (default: 2e-3)
            warmup_iters: Number of warmup iterations for linear LR (default: 1000)
            end_iteration: Total iterations until LR decays to 1% (default: 350000)
            device: Device to run on ('cuda' or 'cpu') (default: 'cuda')
        """
        self.device = device
        self.bilateral_grid_shape = grid_shape
        
        # Initialize bilateral grid with configurable image_num
        self.bil_grids = BilateralGrid(
            image_num,
            grid_X=self.bilateral_grid_shape[0],
            grid_Y=self.bilateral_grid_shape[1],
            grid_W=self.bilateral_grid_shape[2],
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = Adam(
            self.bil_grids.parameters(),
            lr=initial_lr * math.sqrt(1),
            eps=1e-15,
        )
        
        # Initialize scheduler with configurable parameters
        self.scheduler = ChainedScheduler(
            [
                LinearLR(
                    self.optimizer,
                    start_factor=0.01,
                    total_iters=warmup_iters,
                ),
                ExponentialLR(
                    self.optimizer, 
                    gamma=0.01 ** (1.0 / (end_iteration - warmup_iters))  # Now decays over (end_iteration - warmup_iters) steps
                ),
            ]
        )
        self.current_iteration = 0
        self.end_iteration = end_iteration
    
    def process_image(self, image, viewpoint_cam):
        """
        Process an image through the bilateral grid.
        
        Args:
            image: Input image tensor (C,H,W format)
            viewpoint_cam: Camera/viewpoint information containing:
                - image_height: height of the image
                - image_width: width of the image
                - uid: unique identifier for the image
                
        Returns:
            Processed RGB values (H,W,3 format)
        """
        # Create normalized grid coordinates [0,1] range
        grid_y, grid_x = torch.meshgrid(
            (torch.arange(viewpoint_cam.image_height, device=self.device) + 0.5) / viewpoint_cam.image_height,
            (torch.arange(viewpoint_cam.image_width, device=self.device) + 0.5) / viewpoint_cam.image_width,
            indexing="ij",
        )
        
        # Prepare inputs for bilateral grid processing
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).to(self.device)
        colors = image.permute(1, 2, 0).unsqueeze(0).to(self.device)
        image_idxs = torch.ones((grid_xy.size(0), 1), dtype=torch.long).to(self.device) * viewpoint_cam.uid
        
        return slice(self.bil_grids, grid_xy, colors, image_idxs)["rgb"]
    
    def optimization_step(self):
        """
        Perform one optimization step:
        1. Update parameters based on computed gradients
        2. Zero out the gradients
        3. Update learning rate according to scheduler
        4. Increment iteration counter
        """
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.current_iteration += 1
    
    def get_current_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_progress(self):
        """Get training progress as a percentage (0-100)."""
        if hasattr(self.scheduler, '_schedulers') and len(self.scheduler._schedulers) > 1:
            end_iter = self.scheduler._schedulers[1].total_iters if hasattr(self.scheduler._schedulers[1], 'total_iters') else None
            if end_iter:
                return min(100.0, 100.0 * self.current_iteration / end_iter)
        # if hasattr(self, 'end_iteration'):
        #     return min(100.0, 100.0 * self.current_iteration / self.end_iteration)
        # return 0.0
