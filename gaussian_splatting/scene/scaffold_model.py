#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from torch_scatter import scatter_max
from gaussian_splatting.utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from gaussian_splatting.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussian_splatting.scene.embedding import Embedding
import math

import torch.nn.functional as F
from argparse import ArgumentParser, Namespace


def generate_mask(depth, point_ratio = 26):
    """
    Generates a mask tensor of the same shape as the input depth tensor, 
    where approximately 1/point_ratio of the elements are selected.

    Args:
        depth (torch.Tensor([h, w])): A 2D tensor representing the depth values.

    Returns:
        torch.Tensor: A boolean mask tensor of the same shape as the input, 
                      with floor(h * w / point_ratio) randomly selected elements.
    """
    h, w = depth.shape
    num_points = h * w
    num_true = num_points // point_ratio

    # Create a mask with all False values
    mask = torch.zeros(h, w, dtype=torch.bool)

    # Randomly select num_true positions to set to True
    indices = torch.randperm(num_points)[:num_true]
    mask.view(-1)[indices] = True

    return mask



class AnchorDict:
    '''
        Performance Metrics (Scale set to 10,000,000):
        
        - Time taken to create hash table: 0.0008 seconds
        - Time taken for hash detection: 0.0470 seconds 
        - (for 10,000,000 size of the hash table & 1,000,000 query)
        
        - Time taken by hash_table_add: 0.0138 seconds 
        - (for adding 10,000,000 size of a hash table to the 10,000,000 size of hash table)
        - Size of the hash table after removing duplicates: torch.Size([17,022,822])

        - Time taken by hash_table_add: 0.0037 seconds 
        - (for adding 4,000,000 size of a hash table to the 10,000,000 size of hash table)
        - Size of the hash table after removing duplicates: torch.Size([13,278,442])

        Note: All tests were performed on single RTX 4090.
    '''
    
    MULTIPLIER = torch.tensor([1, 2654435761, 805459861]) 

    INCREMENT = 1
    MODULUS = 2**63 - 1 # data type long is 64-bit

    def __init__(self) -> None:
        self._hash_table = None
    
        
    def _hash_func(self, in_tensor):
        """
        Returns hash tensor using method described in Instant-NGP
        https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/field_components/encodings.py#L401

        Args:
            in_tensor: Tensor to be hashed
        """

        in_tensor = in_tensor * self.MULTIPLIER.to(in_tensor.device)
        in_tensor = in_tensor.long()
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.MODULUS
        x += self.INCREMENT

        return x
    
    def set_hash_table(self, x:torch.tensor):
        # assert self._hash_table is None, 'A hash table has been created!'

        hashed_tensor = self._hash_func(x)
        self._hash_table = torch.unique(hashed_tensor)
    
    def hash_detect(self, x:torch.tensor):
        if self._hash_table is None:
            # self.set_hash_table(x)
            return torch.zeros((x.shape[0]), device=x.device, dtype=torch.bool)
            
        # assert self._hash_table is not None, 'Hash table is not existed!'

        x = self._hash_func(x)

        result = torch.isin(x, self._hash_table)
        return result

    def hash_table_add(self, x: torch.Tensor):
        if self._hash_table is None:
            self.set_hash_table(x)
            return
        
        x = self._hash_func(x)
        
        combined_tensor = torch.cat((self._hash_table, x))
        self._hash_table = torch.unique(combined_tensor)


class GaussiansOptParams:
    def __init__(self) -> None:
        self.iterations = 50       # optimization per frame
        self.position_lr_init = 0.0016
        self.position_lr_final = 0.000016
        self.position_lr_delay_mult = 0.0001
        self.position_lr_max_steps = 30_00000
        self.feature_lr = 0.025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.15
        self.lambda_dssim = 0.2
        self.densification_interval = 75
        self.opacity_reset_interval = 300
        self.densify_from_iter = 5
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.002
        self.random_background = False

        self.slide_window = 12
        self.upsample_scale = 3.16 # this parameter is constraint in [1, 8]
        self.saving_frame = 1
        self.frame_computed = 0

        # Scaffold-GS parameters
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 10 # 1500
        self.update_interval = 100
        self.update_until = 1500_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        self.init_proba_scaler = 2.0
        self.penalty_scaler = 2.0  # Scale down penalty to reduce its effect

class GaussianModel:

    # Serilised function
    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def setup_functions(self):

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.anchor_dict = {}

        for i in range(len(self.voxel_size_lis)):
            self.anchor_dict[i] = AnchorDict()

    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size_lis: dict = None, # float=0.01,
                 distance_lis: dict = None,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 intrinsics = None,
                 config = None
                 ):
        
        # intrinsics: torch.tensor([fx, fy, cx, cy], dtype=torch.float)
        self.intrinsics = intrinsics
        self.config = config
        self.opt = GaussiansOptParams()

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size_lis = voxel_size_lis
        self.distance_lis = distance_lis
        self.max_level = len(voxel_size_lis) - 1
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.level_dim = 1

        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0).cuda()
        self._anchor_index = torch.empty(0, dtype=torch.int32, device="cuda")
        self._offset = torch.empty(0).cuda()
        self._anchor_feat = torch.empty(0).cuda()
        
        self.opacity_accum = torch.empty(0).cuda()

        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0).cuda()
        self._opacity = torch.empty(0).cuda()
        self._level = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0).cuda()
        
        self.offset_gradient_accum = torch.empty(0).cuda()
        self.offset_denom = torch.empty(0).cuda()

        self.anchor_demon = torch.empty(0).cuda()

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.height = None
        self.width = None
        
        self.save_dir = None

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim+self.level_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim+self.level_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.level_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        radius = 3
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1)
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.disc_kernel[0, 0, torch.sqrt(x**2 + y**2) <= radius + 0.5] = 1
        self.disc_kernel = self.disc_kernel.cuda() / self.disc_kernel.sum()
        lap = torch.tensor(
            [[0, 1, 0],
             [1,-4, 1],
             [0, 1, 0]], 
            device="cuda", dtype=torch.float32
        )
        lap = lap.view(1, 1, 3, 3)
        # Apply the same depthwise convolution kernel to each channel.
        self.laplacian_kernel = lap


        #print(f"[depth debug] saved to: {out_path}")
    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor
    @property
    def get_anchor_index(self):
        return self._anchor_index


    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_level(self):
        return self._level

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        if isinstance(data, np.ndarray):
            np.random.shuffle(data)
            data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
        elif isinstance(data, torch.Tensor):

            data = torch.unique(torch.round(data / voxel_size), dim=0) * voxel_size
        else:
            raise TypeError("Unsupported data type. Please provide a numpy array or a torch tensor.")
        return data
    
    def pcd_from_depth(self, depth, w2c, mask=None, rgb=None,
                    compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
        '''
            get the point clouds from depth map, the fun is used
                before generating the voxels.

            code is adapted from SplaTAM.

            input:
                depth:  torch.tensor([h, w], device = 'cuda')
                
                w2c:    torch.tensor([4, 4])
                mask:   torch.tensor([h, w])
            return:
                pcd:    torch.tensor([n, 3], device = 'cuda')
        '''
        

        width, height = depth.shape[1], depth.shape[0]
        FX, FY, CX, CY = self.intrinsics

        x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                        torch.arange(height).cuda().float(),
                                        indexing='xy')
        xx = (x_grid - CX)/FX
        yy = (y_grid - CY)/FY
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth.reshape(-1)

        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
        

        if compute_mean_sq_dist:
            if mean_sq_dist_method == "projective":
                scale_gaussian = depth_z / ((FX + FY)/2)
                mean3_sq_dist = scale_gaussian**2
            else:
                raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        
        if mask is not None:
            pts = pts[mask.reshape(-1)]
            if compute_mean_sq_dist:
                mean3_sq_dist = mean3_sq_dist[mask]

        if compute_mean_sq_dist:
            return pts, mean3_sq_dist
        else:
            # If an RGB image is provided, extract corresponding colors
            if rgb is not None:
                # Adjust RGB shape if necessary (from (3, h, w) to (h, w, 3))
                rgb = rgb.permute(1, 2, 0)  # Change from (3, h, w) to (h, w, 3)
                rgb = rgb.reshape(-1, 3)
                if mask is not None:
                    rgb = rgb[mask.reshape(-1)]
                return pts, rgb
            else:
                return pts

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def create_from_pcd(self, pcd, cam_pos, anchor_index:int, init_data=None):

        points = pcd
        self.set_appearance(15120)
        
        for i in range(self.max_level+1):

            if self.voxel_size_lis[i] <= 0:
                init_points = torch.tensor(points).float().cuda()
                init_dist = distCUDA2(init_points).float().cuda()
                median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
                self.voxel_size_lis[i] = median_dist.item()
                del init_dist
                del init_points
                torch.cuda.empty_cache()

            fused_point_cloud = self.voxelize_sample(points, voxel_size=self.voxel_size_lis[i])

            point_dist = torch.norm(fused_point_cloud - cam_pos, dim=-1)
            if i == 0:
                fused_point_cloud = fused_point_cloud[point_dist < self.distance_lis[0]]
            elif i == self.max_level:
                fused_point_cloud = fused_point_cloud[point_dist >= self.distance_lis[i-1]]
            else:
                fused_point_cloud = fused_point_cloud[(self.distance_lis[i-1] <= point_dist) & (point_dist < self.distance_lis[i])]

            if fused_point_cloud.shape[0] > 0:
                #print("The number of new inserted point is:", fused_point_cloud.shape[0])
                offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
                anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
                
                dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
                dist_scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)
                if init_data is not None:
                    init_scales, init_opacity = self._init_scaling_opacity_from_proba(
                    fused_point_cloud, cam_pos, init_data
                )
                    if init_scales is not None:
                        scales = torch.min(init_scales, dist_scales)
                    if init_opacity is not None:
                        opacities = init_opacity
                else:
                    scales = dist_scales
                    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
                
                rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots[:, 0] = 1

                level_anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
                level_offset = nn.Parameter(offsets.requires_grad_(True))
                level_anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
                level_scaling = nn.Parameter(scales.requires_grad_(True))
                level_rotation = nn.Parameter(rots.requires_grad_(False))
                level_opacity = nn.Parameter(opacities.requires_grad_(False))
                level_level = torch.full((fused_point_cloud.shape[0],), i, device='cuda', dtype=torch.int)
                
                self._anchor = torch.cat([self._anchor.detach(), level_anchor.requires_grad_(True)], dim=0)
                self._offset = torch.cat([self._offset.detach(), level_offset.requires_grad_(True)], dim=0)
                self._anchor_feat = torch.cat([self._anchor_feat.detach(), level_anchor_feat.requires_grad_(True)], dim=0)
                self._scaling = torch.cat([self._scaling.detach(), level_scaling.requires_grad_(True)], dim=0)
                self._rotation = torch.cat([self._rotation.detach(), level_rotation.requires_grad_(False)], dim=0)
                self._opacity = torch.cat([self._opacity.detach(), level_opacity.requires_grad_(False)], dim=0)
                self._level = torch.cat([self._level.detach(), level_level.requires_grad_(False)], dim=0)

                self._anchor = nn.Parameter(self._anchor.requires_grad_(True))
                self._offset = nn.Parameter(self._offset.requires_grad_(True))
                self._anchor_feat = nn.Parameter(self._anchor_feat.requires_grad_(True))
                self._scaling = nn.Parameter(self._scaling.requires_grad_(True))
                self._rotation = nn.Parameter(self._rotation.requires_grad_(False))
                self._opacity = nn.Parameter(self._opacity.requires_grad_(False))

                self.anchor_dict[i].set_hash_table(fused_point_cloud)
                # self.anchor_dict[i].set_kdtree(fused_point_cloud)
    
                add_anchor_index = torch.full((fused_point_cloud.shape[0],), anchor_index, dtype=torch.int).cuda()
                self._anchor_index = torch.cat([self._anchor_index, add_anchor_index])

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
    
        self.training_setup(self.opt)


    def add_pcd(self, pcd, cam_pos, anchor_index:int, init_data=None):
        points = pcd

        for i in range(self.max_level+1):
            fused_point_cloud = self.voxelize_sample(points, voxel_size=self.voxel_size_lis[i])

            result = self.anchor_dict[i].hash_detect(fused_point_cloud)

            fused_point_cloud = fused_point_cloud[~result]

            point_dist = torch.norm(fused_point_cloud - cam_pos, dim=-1)
            if i == 0:
                fused_point_cloud = fused_point_cloud[point_dist < self.distance_lis[0] * 1.05]
            elif i == self.max_level:
                fused_point_cloud = fused_point_cloud[point_dist >= self.distance_lis[i-1]]
            else:
                fused_point_cloud = fused_point_cloud[(self.distance_lis[i-1] <= point_dist * 1.05) & (point_dist < self.distance_lis[i] * 1.05)]

            if fused_point_cloud.shape[0] > 0:
                #print("Level--", i, "The number of new inserted point is:", fused_point_cloud.shape[0])
                new_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim), dtype=torch.float, device=fused_point_cloud.device)
                new_scaling = torch.ones_like(fused_point_cloud).repeat([1,2]).float().cuda()*self.voxel_size_lis[i] # *0.05
                new_scaling = torch.log(new_scaling)

                init_scales, init_opacity = self._init_scaling_opacity_from_proba(
                    fused_point_cloud, cam_pos, init_data
                ) if init_data is not None else (None, None)
                if init_scales is not None:
                    new_scaling = init_scales 

                new_rotation = torch.zeros([fused_point_cloud.shape[0], 4], device=fused_point_cloud.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
                if init_opacity is not None:
                    new_opacities = init_opacity

                new_offsets = torch.zeros_like(fused_point_cloud, dtype=torch.float, device=fused_point_cloud.device).unsqueeze(dim=1).repeat([1,self.n_offsets,1])

                d = {
                    "anchor": fused_point_cloud,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                add_level = torch.full((fused_point_cloud.shape[0],), i, device='cuda', dtype=torch.int)
                self._level = torch.cat([self._level.detach(), add_level.requires_grad_(False)], dim=0)
                
                self.anchor_dict[i].hash_table_add(fused_point_cloud)
                # self.anchor_dict[i].kdtree_add(fused_point_cloud)
                
                add_anchor_index = torch.full((fused_point_cloud.shape[0],), anchor_index, dtype=torch.int).cuda()
                self._anchor_index = torch.cat([self._anchor_index, add_anchor_index])

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)


    def get_dog_norm(self, img, sigma1=1.0, sigma2=2.0):
        """
        Compute DoG (Difference of Gaussian) strength map.
        Input:
            img:  (C, H, W) torch tensor
        Output:
            2D tensor (H, W) with normalized responses in [0, 1].
        """
    
        import torch
        import torch.nn.functional as F
    
        C, H, W = img.shape
    
        # ---- helper: generate gaussian kernel ----
        def gaussian_kernel(sigma):
            # kernel size ≈ 6σ+1
            k = int(6 * sigma + 1)
            if k % 2 == 0:
                k += 1
            x = torch.arange(k) - k // 2
            g = torch.exp(-(x**2) / (2 * sigma * sigma))
            g = g / g.sum()
            return g
    
        # ---- create 1D kernels ----
        g1 = gaussian_kernel(sigma1).to(img.device, img.dtype)
        g2 = gaussian_kernel(sigma2).to(img.device, img.dtype)
    
        # reshape for depthwise conv
        g1 = g1.view(1, 1, -1)               # (1,1,k)
        g2 = g2.view(1, 1, -1)
    
        # ---- apply separable Gaussian blur ----
    
        # blur along width
        img_x1 = F.conv2d(img.unsqueeze(0),
                          weight=g1.unsqueeze(1).repeat(C,1,1,1),
                          padding=(0, g1.shape[-1]//2),
                          groups=C)
    
        img_x2 = F.conv2d(img.unsqueeze(0),
                          weight=g2.unsqueeze(1).repeat(C,1,1,1),
                          padding=(0, g2.shape[-1]//2),
                          groups=C)
    
        # blur along height
        img_g1 = F.conv2d(img_x1,
                          weight=g1.view(1,1,-1,1).repeat(C,1,1,1),
                          padding=(g1.shape[-1]//2, 0),
                          groups=C)[0]
    
        img_g2 = F.conv2d(img_x2,
                          weight=g2.view(1,1,-1,1).repeat(C,1,1,1),
                          padding=(g2.shape[-1]//2, 0),
                          groups=C)[0]
    
        # ---- DoG (Difference of Gaussian) ----
        dog = img_g1 - img_g2    # shape: (C,H,W)
    
        # ---- Compute norm per pixel ----
        dog_norm = dog.abs().sum(dim=0)   # (H,W)
    
        # ---- normalize to [0,1] ----
        #dog_norm = dog_norm / (dog_norm.max().clamp_min(1e-6))
        
        quantile_val = torch.quantile(dog_norm, 0.99)
        dog_norm = torch.clamp(dog_norm, max=quantile_val)
        dog_norm = dog_norm / (quantile_val + 1e-6)

        return dog_norm.clamp(0, 1)


    def _init_scaling_opacity_from_proba(self, points, cam_pos, init_data):
        prob_map = init_data.get("prob_map") if init_data is not None else None
        if prob_map is None or points.shape[0] == 0:
            return None, None

        T_w2c = init_data["T_w2c"]
        width = float(init_data["width"])
        height = float(init_data["height"])
        focal = torch.tensor(init_data["focal"], device=points.device, dtype=points.dtype)

        ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
        pts_h = torch.cat([points, ones], dim=1)
        pts_cam = (T_w2c @ pts_h.T).T[:, :3]
        z = pts_cam[:, 2].clamp_min(1e-6)

        fx, fy, cx, cy = self.intrinsics
        fx = fx.to(dtype=points.dtype)
        fy = fy.to(dtype=points.dtype)
        cx = cx.to(dtype=points.dtype)
        cy = cy.to(dtype=points.dtype)

        u = (pts_cam[:, 0] * fx) / z + cx
        v = (pts_cam[:, 1] * fy) / z + cy

        width_denom = max(width - 1.0, 1.0)
        height_denom = max(height - 1.0, 1.0)

        grid_x = (u / width_denom) * 2.0 - 1.0
        grid_y = (v / height_denom) * 2.0 - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)

        prob_map = prob_map.to(points.device, dtype=points.dtype)
        sampled_proba = F.grid_sample(
            prob_map[None, None],
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        ).view(-1)
        sampled_proba = sampled_proba.clamp_min(1e-6)

        max_scale = 8
        scales = 1.0 / torch.sqrt(sampled_proba)
        scales = scales.clamp_(1.0, max_scale)

        dist = torch.linalg.vector_norm(points - cam_pos[None], dim=-1)
        scales = scales * dist / focal.clamp_min(1e-6)
        scales = scales.clamp_(1e-6, 1e6)

        scale_logs = torch.log(scales).unsqueeze(-1).repeat(1, 2).repeat(1, 3)

        max_proba = sampled_proba.max().clamp_min(1e-6)
        proba_norm = sampled_proba / max_proba
        base_opacity = (0.02 + 0.05 * proba_norm).clamp(0.02, 0.2)
        opacity_logits = inverse_sigmoid(base_opacity.unsqueeze(-1))

        return scale_logs, opacity_logits


    def extend_gaussian(self, camera, depthmap, anchor_index:int, rgb = None, init = False, render_pkg=None, frame_idx=0):
        '''
            get the keyframe and extend gaussians

        '''
        use_nms = self.config['Hierarchical']['use_nms']
        T_w2c = torch.eye(4, device=camera.R.device)
        T_w2c[0:3, 0:3] = camera.R
        T_w2c[0:3, 3] = camera.T

        cam_center = camera.camera_center

        self.height = depthmap.shape[0]
        self.width = depthmap.shape[1]
        #print(self.height, self.width)      
        img = camera.original_image
        img = F.avg_pool2d(img, 2)
        img = F.interpolate(
           img[None], (self.height, self.width), mode="bilinear", align_corners=True
        )[0]

        init_proba = self.get_dog_norm(img) # try dog

        penalty = torch.zeros_like(init_proba)
        if self.get_anchor.shape[0] > 0:
            render = render_pkg["render"]
            rendered_depth = render_pkg["depth"]

            penalty = self.get_dog_norm(render)
        else :
            render = torch.zeros_like(init_proba)
            rendered_depth = torch.zeros_like(init_proba)

        init_proba *= self.opt.init_proba_scaler
        penalty *=  self.opt.penalty_scaler
        sampling_scores = (init_proba - penalty).clamp_min(0.0)
        sample_mask = torch.rand_like(init_proba) < sampling_scores# eq. 3

        if sample_mask.sum() == 0:
            print("Warning: no points sampled, relax the sampling criteria.")
            flat_scores = sampling_scores.view(-1)
            k_select = min(flat_scores.numel(), max(flat_scores.numel() // 64, 1))
            if k_select > 0:
                topk_idx = torch.topk(flat_scores, k=k_select, sorted=False).indices
                sample_mask = torch.zeros_like(sample_mask, dtype=torch.bool)
                sample_mask.view(-1)[topk_idx] = True
                
        # Retry sampling when fewer than 50 candidates survive.
        if sample_mask.sum() < 50:
            
            print("No enough points, relax the sampling criteria.")
            sampling_scores = sampling_scores / 2.0
            sample_mask = torch.rand_like(init_proba) < sampling_scores
        
        ## NMS(Non-Maximum Suppression)
        if use_nms:
            nms_kernel = 3 
            padding = nms_kernel // 2
            max_map = F.max_pool2d(
                init_proba.unsqueeze(0).unsqueeze(0), 
                kernel_size=nms_kernel, 
                stride=1, 
                padding=padding
            ).squeeze()
        # Keep local maxima with a numerical tolerance and minimum response.
        if use_nms:
            nms_mask = (init_proba >= (max_map - 1e-6)) & (init_proba > 1e-3)
            downsample_mask = nms_mask
        else:
            downsample_mask = generate_mask(depthmap, point_ratio=self.config['Hierarchical']['point_ratio']).to(sample_mask.device)
        sample_mask &= downsample_mask
        
        # visualization to save 
        
        point_cloud, _ = self.pcd_from_depth(
            depthmap,
            T_w2c,
            rgb=rgb,
            mask=sample_mask,
        )

        focal = float((self.intrinsics[0] + self.intrinsics[1]) * 0.5)
        init_data = {
            "prob_map": init_proba.detach(),
            "T_w2c": T_w2c,
            "width": self.width,
            "height": self.height,
            "focal": focal,
        }

        if init:
            sample_mask = torch.rand_like(init_proba) < init_proba
            point_cloud, color = self.pcd_from_depth(depthmap, T_w2c, rgb=rgb, mask=sample_mask)
            self.create_from_pcd(point_cloud, cam_center, anchor_index, init_data=init_data)
        else:
            self.add_pcd(point_cloud, cam_center, anchor_index, init_data=init_data)


    def update_anchor_loop(self, update_point_cloud):
        
        updated_anchor = self._anchor.clone().detach()
        # Update anchors at each voxel level after loop correction.
        for level in range(self.max_level + 1):
            voxel_size = self.voxel_size_lis[level]
            level_mask = (self._level == level)     
            level_indices = torch.nonzero(level_mask, as_tuple=False).squeeze(-1)

            if level_indices.numel() == 0:
                continue
            
            level_points = update_point_cloud[level_indices]
            # Snap corrected points to voxel centers at this level.
            voxelized_points = torch.round(level_points / voxel_size) * voxel_size

            if voxelized_points.shape[0] == 0:
                continue
            
            min_len = min(len(level_indices), len(voxelized_points))
            selected_indices = level_indices[:min_len]
            selected_voxelized = voxelized_points[:min_len]
            
            updated_anchor[selected_indices] = selected_voxelized
            # Rebuild spatial hashes alongside anchor positions to avoid stale lookups.
            
            
            self.anchor_dict[level].set_hash_table(selected_voxelized)
        # Replace the anchor tensor in the optimizer.
        self._anchor = nn.Parameter(updated_anchor.requires_grad_(True))

        self.replace_tensor_to_optimizer(self._anchor, 'anchor')


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        level_mask = (self._level == 0)

        anchor = self._anchor[level_mask].detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat[level_mask].detach().cpu().numpy()
        offset = self._offset[level_mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity[level_mask].detach().cpu().numpy()
        scale = self._scaling[level_mask].detach().cpu().numpy()
        rotation = self._rotation[level_mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1
    
    @torch.no_grad()
    def _prune_anchor_optimizer(self, mask):
        mask = mask.view(-1)
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors

    def prune_anchor(self, mask):
        mask = mask.view(-1)
        valid_points_mask = ~mask

        if valid_points_mask.shape[0] != self._anchor.shape[0]:
            raise ValueError(f"Mask length {valid_points_mask.shape[0]} does not match anchor length {self._anchor.shape[0]}")

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._level = self._level[valid_points_mask]
        self._anchor_index = self._anchor_index[valid_points_mask]

        for level in range(self.max_level+1):
            self.anchor_dict[level].set_hash_table(self._anchor[self._level == level])

        
    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size_lis[0]*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size           
            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                # New anchors inherit their source frame indices and LoD levels.
                anchor_index_expanded = self._anchor_index.unsqueeze(1).repeat([1, self.n_offsets]).reshape(-1)
                anchor_level_expanded = self._level.unsqueeze(1).repeat(1, self.n_offsets).view(-1) # [N*k]

                
                candidate_anchor_index = anchor_index_expanded[candidate_mask]
                new_anchor_index = scatter_max(candidate_anchor_index.unsqueeze(1), inverse_indices.unsqueeze(1), dim=0)[0].squeeze(1)[remove_duplicates]

                
                # Use the coarsest contributing level when offsets share a voxel.
                candidate_levels = anchor_level_expanded[candidate_mask]
                new_levels = scatter_max(candidate_levels.unsqueeze(1), inverse_indices.unsqueeze(1), dim=0)[0].squeeze(1)[remove_duplicates]
                new_levels = new_levels.int() 

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()


                self._level = torch.cat([self._level.detach(), new_levels.requires_grad_(False)], dim=0)
                self._anchor_index = torch.cat([self._anchor_index, new_anchor_index], dim=0)

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]

                # Insert each new anchor into the hash table for its inherited level.
                
                unique_levels = torch.unique(new_levels)
                for lvl in unique_levels:
                    lvl_mask = (new_levels == lvl)
                    
                    if lvl.item() < len(self.anchor_dict):
                        self.anchor_dict[lvl.item()].hash_table_add(candidate_anchor[lvl_mask])

    def adjust_anchor_hybrid(self, check_interval=100, success_threshold=0.8, 
                             grad_threshold=0.0002, min_opacity=0.005, 
                             fastgs_densify_mask=None, pruning_score=None, 
                             enable_densify=True, enable_prune=True):
        
        
        vis_dir = os.path.join(self.save_dir, "debug_vis")
        os.makedirs(vis_dir, exist_ok=True)
        
        iter_idx = len(os.listdir(vis_dir)) // 3 

        
        grads = self.offset_gradient_accum / self.offset_denom 
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(input=grads, dim=-1)
        
        
        n_anchors_before = self.get_anchor.shape[0]
        
        # Require more than 20 observations before structural changes.
        OBS_THRESHOLD_FIXED = 20.0 
        obs_count_mask = (self.offset_denom > OBS_THRESHOLD_FIXED).squeeze(dim=1)

        # -----------------------------------------------------------------------------------
        
        # -----------------------------------------------------------------------------------
        final_densify_mask = torch.zeros_like(grad_mask) if 'grad_mask' in locals() else torch.zeros(grads_norm.shape[0], dtype=torch.bool, device='cuda')
        
        if enable_densify:
            
            grad_mask = grads_norm >= grad_threshold
            
            
            # force_mask = torch.zeros_like(grad_mask)
            # if grads_norm.numel() > 0:
            #     if grads_norm.numel() > 10000000:
            #         step = 100
            #         sampled = grads_norm[::step]
            #         top1_thr = torch.quantile(sampled, 0.99)
            #     else:
            #         top1_thr = torch.quantile(grads_norm, 0.99)
                
            
            #     top1_thr = torch.maximum(top1_thr, torch.tensor(grad_threshold, device=grads_norm.device))
            #     force_mask = grads_norm >= top1_thr

            # Densification requires both gradient and FastGS support.
            
            densify_mask = torch.logical_and(grad_mask, fastgs_densify_mask)

            final_densify_mask = torch.logical_and(densify_mask, obs_count_mask)
            
            
            n_grad = grad_mask.sum().item()

            n_fastgs = fastgs_densify_mask.sum().item() if fastgs_densify_mask is not None else 0
            n_obs = obs_count_mask.sum().item()
            print(f"[DENSIFY] TotalCandidates: {grads_norm.shape[0]}")
            # print(f"  > GradMatch: {n_grad} | Top1%Force: {n_force} | FastGS: {n_fastgs}")
            print(f"  > GradMatch: {n_grad}  | FastGS: {n_fastgs}")
            print(f"  > ObsQualified: {n_obs} ==> FINAL ADD: {final_densify_mask.sum().item()}")

            self.anchor_growing(grads_norm, grad_threshold, final_densify_mask)

        # -----------------------------------------------------------------------------------
        
        # -----------------------------------------------------------------------------------
        
        reset_mask = obs_count_mask 
        current_num_anchors = self.get_anchor.shape[0]
        if reset_mask.shape[0] < current_num_anchors:
             padding_len = current_num_anchors - reset_mask.shape[0]
             reset_mask = torch.cat([reset_mask, torch.zeros(padding_len, dtype=torch.bool, device=reset_mask.device)])
        self.offset_denom[reset_mask] = 0
        padding_len = current_num_anchors*self.n_offsets - self.offset_denom.shape[0]
        if padding_len > 0:
            padding = torch.zeros([padding_len, 1], dtype=torch.int32, device=self.offset_denom.device)
            self.offset_denom = torch.cat([self.offset_denom, padding], dim=0)
        self.offset_gradient_accum[reset_mask] = 0
        padding_len = current_num_anchors*self.n_offsets - self.offset_gradient_accum.shape[0]
        if padding_len > 0:
            padding = torch.zeros([padding_len, 1], dtype=torch.int32, device=self.offset_gradient_accum.device)
            self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding], dim=0)

        # -----------------------------------------------------------------------------------
        
        # -----------------------------------------------------------------------------------
        final_prune_mask = torch.zeros(current_num_anchors, dtype=torch.bool, device='cuda')
        
        if enable_prune:
            avg_opacity = self.opacity_accum / (self.anchor_demon + 1e-6)
            low_opacity_mask = (avg_opacity < min_opacity).squeeze(dim=1)
            
            fastgs_prune_mask = torch.zeros_like(low_opacity_mask)
            if pruning_score is not None:
                score_len = pruning_score.shape[0]
                if score_len < current_num_anchors:
                    padding_size = current_num_anchors - score_len
                    padding = torch.zeros(padding_size, device=pruning_score.device, dtype=pruning_score.dtype)
                    pruning_score_padded = torch.cat([pruning_score, padding], dim=0)
                    pruning_score = pruning_score_padded 
                    fastgs_prune_mask = (pruning_score > self.config["Training"]["prune_thresh"])
                else:
                    pruning_score = pruning_score[:current_num_anchors]
                    fastgs_prune_mask = (pruning_score > self.config["Training"]["prune_thresh"])

            candidate_prune_mask = torch.logical_or(low_opacity_mask, fastgs_prune_mask)
            safe_mask = (self.anchor_demon > OBS_THRESHOLD_FIXED).squeeze(dim=1)
            final_prune_mask = torch.logical_and(candidate_prune_mask, safe_mask)

            
            print(f"[PRUNE] LowOpacity: {low_opacity_mask.sum().item()} | FastGS>prune_threshold: {fastgs_prune_mask.sum().item()}")
            print(f"  ==> FINAL REMOVE: {final_prune_mask.sum().item()}")

        # ================== [VISUALIZATION START (CPU Optimized)] ==================
        import numpy as np
        
        # 1. Structure Change
        # Export diagnostic clouds on CPU to limit GPU memory use.
        xyz_all_cpu = self.get_anchor.detach().cpu().numpy()
        n_current = xyz_all_cpu.shape[0]
        
        # Slice newly added anchors after transferring to CPU.
        if n_current > n_anchors_before:
            xyz_added = xyz_all_cpu[n_anchors_before:]
        else:
            xyz_added = np.empty((0, 3))
        
        
        rgb_added = np.tile(np.array([0, 255, 0], dtype=np.uint8), (xyz_added.shape[0], 1))
        
        # Collect pruned anchors for the diagnostic point cloud.
        prune_mask_cpu = final_prune_mask.cpu().numpy()
        xyz_removed = xyz_all_cpu[prune_mask_cpu]
        
        
        rgb_removed = np.tile(np.array([255, 0, 0], dtype=np.uint8), (xyz_removed.shape[0], 1))
        
        
        if xyz_added.shape[0] > 0 or xyz_removed.shape[0] > 0:
            xyz_vis = np.concatenate([xyz_added, xyz_removed], axis=0)
            rgb_vis = np.concatenate([rgb_added, rgb_removed], axis=0)
            self.save_ply_numpy(f"{vis_dir}/structure_iter{iter_idx}.ply", xyz_vis, rgb_vis)
            
        
        del xyz_added, rgb_added, xyz_removed, rgb_removed, xyz_vis, rgb_vis

        # 2. Gradient Heatmap
        if grads_norm.numel() > 0:
            
            grad_anchor = grads_norm.view(-1, self.n_offsets).max(dim=1)[0]
            
            
            if grad_anchor.shape[0] < n_current:
                padding = torch.zeros(n_current - grad_anchor.shape[0], device='cuda')
                grad_anchor = torch.cat([grad_anchor, padding])
            
            
            if grad_anchor.numel() > 10000000:
                 g_max = torch.quantile(grad_anchor[::100], 0.99)
            else:
                 g_max = torch.quantile(grad_anchor, 0.99)
            
            
            g_norm = (grad_anchor / (g_max + 1e-6)).cpu()
            
            
            rgb_grad = self.scalar_to_colormap(g_norm)
            self.save_ply_numpy(f"{vis_dir}/gradient_iter{iter_idx}.ply", xyz_all_cpu, rgb_grad)
            del grad_anchor, g_norm, rgb_grad

        # 3. FastGS Score Heatmap
        if pruning_score is not None:
            
            score_cpu = pruning_score.detach().cpu()
            rgb_score = self.scalar_to_colormap(score_cpu)
            self.save_ply_numpy(f"{vis_dir}/fastgs_score_iter{iter_idx}.ply", xyz_all_cpu, rgb_score)
            del score_cpu, rgb_score
        
        
        del xyz_all_cpu
        # ================== [VISUALIZATION END] ==================

        
        if enable_prune and final_prune_mask.sum() > 0:
            self.prune_anchor(final_prune_mask)

        # Resize statistics to match the surviving anchors and offsets.
        if enable_prune and final_prune_mask.sum() > 0:
            valid_mask = ~final_prune_mask
            self.opacity_accum = self.opacity_accum[valid_mask]
            self.anchor_demon = self.anchor_demon[valid_mask]
            self.offset_denom = self.offset_denom.view([-1, self.n_offsets])[valid_mask].view([-1, 1])
            self.offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[valid_mask].view([-1, 1])

        if enable_prune:
            self.opacity_accum.fill_(0)
            self.anchor_demon.fill_(0)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
    
        
    def save_ply_numpy(self, path, xyz, rgb):
        """Write a PLY with NumPy without allocating GPU memory."""
        import numpy as np
        num_points = xyz.shape[0]
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(num_points):
                f.write(f"{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f} {int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n")

    def scalar_to_colormap(self, values, cmap='jet'):
        """Map tensor or NumPy scalar values to uint8 RGB colors."""
        import numpy as np
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        values = np.clip(values, 0, 1)
        
        n = len(values)
        colors = np.zeros((n, 3), dtype=np.uint8)
        
        # Vectorized blue-to-red colormap.
        
        idx0 = values < 0.25
        idx1 = (values >= 0.25) & (values < 0.5)
        idx2 = (values >= 0.5) & (values < 0.75)
        idx3 = values >= 0.75
        
        # 0-0.25: Blue to Cyan
        colors[idx0, 1] = (255 * (values[idx0] / 0.25)).astype(np.uint8)
        colors[idx0, 2] = 255
        
        # 0.25-0.5: Cyan to Green
        colors[idx1, 1] = 255
        colors[idx1, 2] = (255 * (1 - (values[idx1] - 0.25) / 0.25)).astype(np.uint8)
        
        # 0.5-0.75: Green to Yellow
        colors[idx2, 0] = (255 * ((values[idx2] - 0.5) / 0.25)).astype(np.uint8)
        colors[idx2, 1] = 255
        
        # 0.75-1.0: Yellow to Red
        colors[idx3, 0] = 255
        colors[idx3, 1] = (255 * (1 - (values[idx3] - 0.75) / 0.25)).astype(np.uint8)
        
        return colors


    def save_ckpt(self, path):
        path = os.path.join(path, 'gaussian_save')
        '''
        Namespace(add_color_dist=False, add_cov_dist=False, add_opacity_dist=False, appearance_dim=0, data_device='cuda', ds=1, eval=False, feat_dim=32, images='images', lod=0, lowpoly=False, model_path='outputs/mipnerf360/kitchen/baseline/2024-01-19_14:00:04', n_offsets=10, ratio=1, resolution=-1, sh_degree=3, source_path='/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/renkerui/data/mipnerf360/kitchen', undistorted=False, update_depth=3, update_hierachy_factor=4, update_init_factor=16, use_feat_bank=False, voxel_size=0.001, white_background=True)
        '''
        params = {
            "add_color_dist": self.add_color_dist,
            "add_cov_dist": self.add_cov_dist,
            "add_opacity_dist": self.add_opacity_dist,
            "appearance_dim": self.appearance_dim,
            "data_device": "cuda",
            "ds": 1,
            "eval": False,
            "feat_dim": self.feat_dim,
            "images": "images",
            "lod": 0,
            "lowpoly": False,
            "model_path": "./", 
            "n_offsets": self.n_offsets,
            "ratio": self.ratio,
            "resolution": -1,
            "sh_degree": 3,
            "source_path": "./", 
            "undistorted": False,
            "update_depth": 3,
            "update_hierachy_factor": 4,
            "update_init_factor": 16,
            "use_feat_bank": False,
            "voxel_size": self.voxel_size_lis[0],
            "white_background": True
        }

        class MockArgs:
            def __init__(self, params):
                for k, v in params.items():
                    setattr(self, k, v)

        args = MockArgs(params)
        print("Output folder: {}".format(path))
        os.makedirs(path, exist_ok = True)
        with open(os.path.join(path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))

        point_cloud_path = os.path.join(path, "point_cloud")
        self.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.save_mlp_checkpoints(point_cloud_path)
        
        
        total_size = 0
        
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
        
        
        size_in_mb = total_size / (1024 * 1024)
        # print(f"[INFO] Gaussian Scene Saved to: {path}")
        print(f"[INFO] Total Storage Size: {size_in_mb:.2f} MB")
        # ======================================================

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim+self.level_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim+self.level_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim+self.level_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
