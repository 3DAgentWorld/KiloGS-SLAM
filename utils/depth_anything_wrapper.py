import torch
import torch.nn.functional as F
import numpy as np
from depth_anything_3.api import DepthAnything3

class DepthAnythingWrapper:
    def __init__(self, config):
        """
        Initialize DA3 from the DepthModel.DepthAnything3 configuration.
        """
        self.config = config
        model_path = config.get('model_path')
        self.process_res = config.get('process_res', 504)
        self.scale_factor = config.get('depth_scale_factor', 1.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[DepthAnythingWrapper] Loading model from: {model_path} ...")
        self.model = DepthAnything3.from_pretrained(model_path).to(self.device)
        
        # Warm up the depth model before tracking begins.
        print("[DepthAnythingWrapper] Warming up...")
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        self.model.inference([dummy], process_res=self.process_res)

    @torch.no_grad()
    def infer(self, image_tensor, intrinsics=None):
        """
        Return depth and validity in the UniDepth-compatible format:
        Input: 
            image_tensor: [3, H, W] uint8 tensor, on CUDA 
            intrinsics: [3, 3] tensor or numpy 
        Output:
            dict {"depth": [1, 1, H, W] float tensor, "mask": [1, 1, H, W] float tensor}
        """

        
        _, H_in, W_in = image_tensor.shape

        
        if isinstance(image_tensor, torch.Tensor):
            img_np = image_tensor.permute(1, 2, 0).cpu().numpy() # (H, W, 3)
        else:
            img_np = image_tensor

        # Optionally pass the original camera intrinsics.
        intrinsics_batch = None
        if self.config.get('use_intrinsics', False) and intrinsics is not None:
            if isinstance(intrinsics, torch.Tensor):
                K = intrinsics.cpu().numpy()
            else:
                K = intrinsics
            intrinsics_batch = K[None, ...]

        # DA3 may return depth at a different resolution from the input.
        prediction = self.model.inference(
            [img_np], 
            intrinsics=intrinsics_batch,
            process_res=self.process_res
        )

        
        raw_depth = prediction.depth[0] # Numpy, e.g., (336, 504)
        
        # Convert the sky prediction to a valid-object mask.
        if hasattr(prediction, 'sky') and prediction.sky is not None:
            sky_raw = prediction.sky[0]
            valid_mask_raw = 1.0 - sky_raw 
        else:
            # Fallback
            valid_mask_raw = (raw_depth < 100.0).astype(np.float32)


        depth_tensor = torch.from_numpy(raw_depth).to(self.device).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.from_numpy(valid_mask_raw).to(self.device).float().unsqueeze(0).unsqueeze(0)

        # Align depth and validity mask to the input image resolution.
        
        if depth_tensor.shape[-2:] != (H_in, W_in):
            # Use bilinear interpolation for depth and nearest-neighbor for the mask.
            depth_tensor = F.interpolate(depth_tensor, size=(H_in, W_in), mode='bilinear', align_corners=False)
            
            
            mask_tensor = F.interpolate(mask_tensor, size=(H_in, W_in), mode='nearest')

        # Apply the configured dataset-specific scale correction.
        depth_tensor = depth_tensor * self.scale_factor

        return {
            "depth": depth_tensor, # shape: [1, 1, H_in, W_in]
            "mask": mask_tensor,   # shape: [1, 1, H_in, W_in]
            "confidence": None
        }