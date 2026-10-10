import csv
import glob
import os
import time

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from gaussian_splatting.utils.graphics_utils import focal2fov

try:
    import pyrealsense2 as rs
except Exception:
    pass

class KittiParser:
    def __init__(self, color_folder, index_path=None):
        self.input_folder = color_folder
        
        # Use the supplied frame selection when available.
        if index_path and os.path.exists(index_path):
            self.color_paths = []
            print(f"Loading KITTI-360 frames from index file: {index_path}")
            
            with open(index_path, 'r') as f:
                lines = f.readlines()
                # Skip the frame-index header.
                for line in lines[1:]:
                    if not line.strip(): 
                        continue
                    parts = line.strip().split()
                    
                    # The first column is the source frame ID; list positions are zero-based.
                    
                    
                    frame_id = int(parts[0])
                    
                    
                    img_name = f"{frame_id:010d}.png"
                    img_path = os.path.join(color_folder, img_name)
                    
                    
                    if os.path.exists(img_path):
                        self.color_paths.append(img_path)
                    else:
                        print(f"Warning: Image listed in index but not found: {img_path}")
                        
        else:
            
            print(f"Loading all images from folder (Glob mode): {color_folder}")
            self.color_paths = sorted(glob.glob(f"{color_folder}/*.[jp][pn]g"))

        self.n_img = len(self.color_paths)
        print(f"Total images loaded: {self.n_img}")

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        print(calibration)

        if True:
            self.fx_ori = calibration["fx"]
            self.fy_ori = calibration["fy"]
            self.cx_ori = calibration["cx"]
            self.cy_ori = calibration["cy"]
            self.width_ori = calibration["width"]
            self.height_ori = calibration["height"]

            w_set = config['Hierarchical']['rendering_width']
            h0, w0 =  self.height, self.width
            ratio = w_set / w0
            h_set = ratio * h0
            h1 = int(h0 * np.sqrt((h_set * w_set) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((h_set * w_set) / (h0 * w0)))

            self.height, self.width = h1, w1
            self.fx *= ratio
            self.fy *= ratio
            self.cx *= ratio
            self.cy *= ratio
            self.ratio = ratio


        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        pose = None
        depth_ori = None
        image_ori = None

        # Fall back to the nearest earlier readable frame for corrupt images.
        search_idx = idx
        while search_idx >= 0:
            color_path = self.color_paths[search_idx]
            try:
                
                image_ori = np.array(Image.open(color_path))
                
                
                if self.has_depth:
                    depth_path = self.depth_paths[search_idx]
                    depth_ori = np.array(Image.open(depth_path)) / self.depth_scale
                
                
                break
            
            except Exception as e:
                # Record the originally requested corrupt frame once.
                if search_idx == idx:
                    print(f"\n[FATAL WARNING] Failed to read image: {color_path}")
                    print(f"Error Message: {e}")
                    
                    try:
                        
                        broken_log_path = os.path.join(self.config["Results"]["save_dir"], "broken_png_log.txt")
                        with open(broken_log_path, "a") as f:
                            f.write(f"Index {idx}: {color_path}\n")
                    except Exception as log_e:
                        print(f"Failed to write log: {log_e}")
                
                
                search_idx -= 1

        if image_ori is None:
            
            raise RuntimeError(f"[ERROR] No readable image between indices 0 and {idx}.")
        
        
        if search_idx != idx:
            print(f"[INFO] Using earlier valid frame {search_idx} in place of corrupt frame {idx}.")
        # =========================================================================

        
        if self.disorted:
            image_ori = cv2.remap(image_ori, self.map1x, self.map1y, cv2.INTER_LINEAR)
        if len(image_ori.shape) == 2:
            # gray to rgb
            image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)
            
        image = cv2.resize(image_ori, (self.width, self.height))
        depth = None

        if self.has_depth and depth_ori is not None:
            depth = cv2.resize(depth_ori, (self.width, self.height))

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        return image, depth, image_ori.transpose(2, 0, 1), depth_ori, pose

class KITTIDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        color_folder = config["Dataset"]["color_path"]
        index_path = config["Dataset"].get("index_path", None)

        parser = KittiParser(color_folder, index_path=index_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths

def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "kitti":
        return KITTIDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
