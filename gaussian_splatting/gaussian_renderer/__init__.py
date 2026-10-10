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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.scene.scaffold_model import GaussianModel

from einops import repeat

def generate_neural_gaussians(viewpoint_camera, pc: GaussianModel, visible_mask=None, is_training=False, mask_opacity: bool = True):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    level = pc.get_level[visible_mask].view(-1, 1)

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        if pc.level_dim == 0:
            cat_view = torch.cat([ob_view], dim=1)
        else:
            cat_view = torch.cat([ob_view, level], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    if pc.level_dim == 0:
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    else:
        cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1) # [N, c+3]
        
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid

        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    if mask_opacity:
        mask = (neural_opacity > 0.0).view(-1)
    else:
        # Keep all offsets when computing metrics so counts align with total anchors.
        mask = torch.ones_like(neural_opacity, dtype=torch.bool).view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot
    
def _ensure_metric_map(metric_map, height, width, device):
    """Return a contiguous int32 map of shape (H*W,) for the rasterizer."""
    if metric_map is None:
        metric_map = torch.zeros((height * width,), dtype=torch.int32, device=device)
    else:
        metric_map = metric_map.to(device=device, dtype=torch.int32)
        if metric_map.dim() > 1:
            metric_map = metric_map.reshape(-1)
        metric_map = metric_map.contiguous()
        if metric_map.numel() != height * width:
            raise ValueError(f"metric_map expects {height * width} values, got {metric_map.numel()}")
    return metric_map


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, visible_mask=None, get_flag=None, metric_map=None, mask_opacity: bool = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(
            viewpoint_camera, pc, visible_mask, is_training=is_training, mask_opacity=mask_opacity
        )
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(
            viewpoint_camera, pc, visible_mask, is_training=is_training, mask_opacity=mask_opacity
        )
    
    # xyz_dist = torch.norm(xyz - viewpoint_camera.camera_center, dim=-1, keepdims=True)
    # dis = xyz_dist.repeat([1, 3])
        
    # cam_center = viewpoint_camera.camera_center.detach()
    # xyz_dist = torch.norm(xyz - cam_center, dim=-1)
    # xyz_dist = xyz_dist.unsqueeze(-1) 
    # dis = xyz_dist.repeat([1, 3])
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points.requires_grad_(True)
    try:
        screenspace_points.retain_grad()
    except:
        pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    metric_map_tensor = _ensure_metric_map(metric_map,
                                           int(viewpoint_camera.image_height),
                                           int(viewpoint_camera.image_width),
                                           bg_color.device)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        get_flag=get_flag,
        metric_map = metric_map_tensor
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # if mask is not None:
    #     rendered_image, radii, depth, opacity = rasterizer(
    #         means3D=xyz[mask],
    #         means2D=screenspace_points[mask],
    #         shs=None,
    #         colors_precomp=color[mask] if color is not None else None,
    #         opacities=opacity[mask],
    #         scales=scaling[mask],
    #         rotations=rot[mask],
    #         # cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
    #         theta=viewpoint_camera.cam_rot_delta,
    #         rho=viewpoint_camera.cam_trans_delta,
    #     )
    # else:
    
    rendered_image, radii, depth, opacity, n_touched, accum_metric_counts = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,

        theta=viewpoint_camera.cam_rot_delta,
        rho=viewpoint_camera.cam_trans_delta,
    )

    # rendered_depth, _ = rasterizer(
    # means3D = xyz,
    # means2D = screenspace_points,
    # shs = None,
    # colors_precomp = dis,
    # opacities = opacity,
    # scales = scaling,
    # rotations = rot,
    # cov3D_precomp = None)

    # shape of depth: torch.tensor([3, h, w])
    # rendered_depth = rendered_depth.mean(dim=0)
    rendered_depth = depth.squeeze()

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "depth": rendered_depth,
                "opacity": opacity,
                "n_touched": n_touched,
                "accum_metric_counts" : accum_metric_counts
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth": rendered_depth,
                "opacity": opacity,
                "n_touched": n_touched,
                "accum_metric_counts" : accum_metric_counts
                }


def render_fastgs(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, visible_mask=None, get_flag=None, metric_map=None):
    """Render without opacity masking so all anchors/offsets participate (used for Gaussian scoring)."""
    return render(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier=scaling_modifier,
        visible_mask=visible_mask,
        get_flag=get_flag,
        metric_map=metric_map,
        mask_opacity=False,
    )
    
def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, visible_mask=None, get_flag=None, metric_map=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # xyz, color, opacity, scaling, rot = generate_neural_gaussians(
    # viewpoint_camera, pc, visible_mask, is_training=False, mask_opacity=True)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    metric_map_tensor = _ensure_metric_map(metric_map,
                                           int(viewpoint_camera.image_height),
                                           int(viewpoint_camera.image_width),
                                           bg_color.device)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        get_flag=get_flag,
        metric_map = metric_map_tensor
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0