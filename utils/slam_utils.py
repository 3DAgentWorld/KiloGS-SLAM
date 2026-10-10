import torch
import numpy as np
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, l2_loss, nearMean_map, image2canny

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False, save_dir=None):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint, save_dir=save_dir)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint, save_dir=save_dir)


def get_loss_mapping_rgb(config, image, depth, viewpoint, save_dir=None):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    
    # ---- New: visualize mask if save_dir is provided ----
    # print("save_dir:", save_dir)
    # if save_dir is not None:

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False, save_dir=None):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = viewpoint.depth
    
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    # l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    # l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    
    # ===== Debug Print =====
    # print(f"\n[DEBUG] Frame {viewpoint.uid}")
    # print(f"  RGB mask shape: {rgb_pixel_mask.shape}, dtype: {rgb_pixel_mask.dtype}")
    # print(f"  RGB mask unique values: {torch.unique(rgb_pixel_mask)}")
    # print(f"  RGB mask True count: {rgb_pixel_mask.sum().item()}")
    # print(f"  Depth mask shape: {depth_pixel_mask.shape}, dtype: {depth_pixel_mask.dtype}")
    # print(f"  Depth mask unique values: {torch.unique(depth_pixel_mask)}")
    # print(f"  Depth mask True count: {depth_pixel_mask.sum().item()}")
    # =======================
    
    # print("l1_rgb:", l1_rgb.mean().item(), "l1_depth:", l1_depth.mean().item())
    
    # if save_dir is not None:
    
    # --- RGB photometric loss ---
    Ll1 = l1_loss(image, gt_image)
    ssim_loss = (1.0 - ssim(image, gt_image))
    rgb_photo = 0.8 * Ll1 + 0.2 * ssim_loss

    # --- Depth loss (keep original masking) ---
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask).mean()
    
    return alpha * rgb_photo + (1.0 - alpha) * l1_depth

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach().squeeze()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

def get_matched_camera_points_vectorized(depth_map1: np.ndarray, feat_points1: np.ndarray, 
                                         depth_map2: np.ndarray, feat_points2: np.ndarray, K: np.ndarray):
    """
    Compute the matched 3D points from two depth maps in their respective camera coordinate systems.

    Parameters:
      depth_map1: Depth map of the first viewpoint, shape [H, W]
      feat_points1: Feature points from the first viewpoint, shape [n, 2], format (u, v)

      depth_map2: Depth map of the second viewpoint, shape [H, W]
      feat_points2: Feature points from the second viewpoint, shape [n, 2], format (u, v)

      K: Camera intrinsic matrix, shape [3, 3]

    Returns:
      pts_cam1: 3D points in the first camera's coordinate system, shape [n, 3]
      pts_cam2: 3D points in the second camera's coordinate system, shape [n, 3]
    """
    # Inverse of the intrinsic matrix
    K_inv = np.linalg.inv(K)

    # ------------------------------
    # First camera points
    # ------------------------------
    n = feat_points1.shape[0]
    pixels1 = np.hstack((feat_points1, np.ones((n, 1))))
    u1 = np.round(feat_points1[:, 0]).astype(int)
    v1 = np.round(feat_points1[:, 1]).astype(int)
    u1 = np.clip(u1, 0, depth_map1.shape[1]-1)
    v1 = np.clip(v1, 0, depth_map1.shape[0]-1)
    depth1 = depth_map1[v1, u1]
    pts_cam1 = depth1[:, np.newaxis] * (K_inv @ pixels1.T).T  # shape [n, 3]

    # ------------------------------
    # Second camera points
    # ------------------------------
    n2 = feat_points2.shape[0]
    pixels2 = np.hstack((feat_points2, np.ones((n2, 1))))
    u2 = np.round(feat_points2[:, 0]).astype(int)
    v2 = np.round(feat_points2[:, 1]).astype(int)
    u2 = np.clip(u2, 0, depth_map2.shape[1]-1)
    v2 = np.clip(v2, 0, depth_map2.shape[0]-1)
    depth2 = depth_map2[v2, u2]
    pts_cam2 = depth2[:, np.newaxis] * (K_inv @ pixels2.T).T

    return pts_cam1, pts_cam2


import open3d as o3d

def compute_sim3_open3d(pts1, pts2):
    """
    Compute the Sim(3) transformation between two point clouds using Open3D.
    
    Parameters:
      pts1: Source point cloud with shape [N, 3]
      pts2: Target point cloud with shape [N, 3]
    
    Returns:
      scale: Scaling factor
      R: Rotation matrix (3x3)
      t: Translation vector (3,)
    """
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd2.points = o3d.utility.Vector3dVector(pts2)

    # Using ICP
    threshold = 1.0  # Maximum correspondence distance allowed
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    T = reg_p2p.transformation  # 4x4 transformation matrix
    R = T[:3, :3].copy()  # Copy R to make it writable
    t = T[:3, 3].copy()   # Copy t to make it writable
    scale = np.linalg.det(R) ** (1/3)  # Compute scale factor

    # Normalize R to make it an orthogonal matrix
    R /= scale

    return scale, R, t

def update_viewpoints_from_poses(viewpoints: dict, poses_update: torch.Tensor):
    """
    Efficiently update the R and T of Camera objects using batched matrix inversion.
    """
    assert len(viewpoints) == poses_update.shape[0], "Mismatched number of poses and viewpoints!"
    sorted_keys = sorted(viewpoints.keys())
    
    w2c_all = torch.linalg.inv(poses_update)

    R_all = w2c_all[:, :3, :3]
    T_all = w2c_all[:, :3, 3]

    for i, key in enumerate(sorted_keys):
        camera = viewpoints[key]
        camera.update_RT(R_all[i], T_all[i])


def update_point_cloud_vectorized(
    pcd: torch.Tensor,
    poses_old, # np.ndarray or torch.tensor
    poses_update, # np.ndarray or torch.tensor
    index_pose: torch.Tensor,
    keyframe_indices: list[int]  # e.g., [1, 4, 5, 9]
) -> torch.Tensor:
    """
    Update 3D point coordinates after pose optimization (e.g. loop closure),
    using a fully vectorized approach on the GPU.

    This function applies a rigid transformation to each point, based on
    the difference between old and updated camera poses. The transformation aligns
    point cloud data with the updated poses after global optimization.

    This function is designed for use before voxelization, so the updated points
    can then be discretized into a voxel grid.

    Args:
        pcd: Tensor of shape [N, 3], original point cloud in world coordinates
        poses_old: np.ndarray or torch.Tensor of shape [n, 4, 4], original camera poses (C2W)
        poses_update: np.ndarray or torch.Tensor of shape [n, 4, 4], updated camera poses (C2W)
        index_pose: Tensor of shape [N], pose index for each point in the point cloud
        keyframe_indices: [K], list mapping each pose to its keyframe index

    Returns:
        updated_pcd: Tensor of shape [N, 3], point cloud updated to the corrected poses
    """
    device = pcd.device

    if isinstance(poses_old, np.ndarray):
        poses_old = torch.from_numpy(poses_old)
    if isinstance(poses_update, np.ndarray):
        poses_update = torch.from_numpy(poses_update)

    poses_old = poses_old.float().to(device)
    poses_update = poses_update.float().to(device)

    # Build mapping from keyframe index to index in pose list
    kf_map = {kf_idx: i for i, kf_idx in enumerate(keyframe_indices)}
    pose_ids = torch.tensor([kf_map[int(i.item())] for i in index_pose], device=device)

    pose_old_batch = poses_old[pose_ids] # [N, 4, 4]
    pose_update_batch = poses_update[pose_ids] # [N, 4, 4]

    ones = torch.ones((pcd.shape[0], 1), device=device)
    pcd_h = torch.cat([pcd, ones], dim=1).unsqueeze(-1) # [N, 4, 1]

    pose_old_inv_batch = torch.linalg.inv(pose_old_batch) # [N, 4, 4]
    pcd_cam = torch.bmm(pose_old_inv_batch, pcd_h)
    pcd_updated_h = torch.bmm(pose_update_batch, pcd_cam)

    return pcd_updated_h[:, :3, 0]


def update_point_cloud_in_batches(
    pcd: torch.Tensor,
    poses_old, # np.ndarray or torch.tensor
    poses_update, # np.ndarray or torch.tensor
    index_pose: torch.Tensor,
    keyframe_indices: list[int],
    batch_size: int = 500_000
) -> torch.Tensor:
    """
    Update 3D point coordinates after pose graph optimization (e.g. loop closure),
    using batched processing to reduce GPU memory usage.

    This function performs the same rigid transformation as func update_point_cloud_vectorized(),
    but processes the point cloud in chunks to avoid out-of-memory errors on large-scale datasets.

    This function is designed for use before voxelization, so the updated points
    can then be discretized into a voxel grid.

    Args:
        pcd: Tensor of shape [N, 3], original point cloud in world coordinates
        poses_old: np.ndarray of shape [n, 4, 4], original camera poses (C2W)
        poses_update: np.ndarray of shape [n, 4, 4], updated camera poses (C2W)
        index_pose: Tensor of shape [N], pose index for each point in the point cloud
        keyframe_indices: [K], list mapping each pose to keyframe ID
        batch_size: Number of points to process per batch (adjust based on GPU memory)

    Returns:
        updated_pcd: Tensor of shape [N, 3], point cloud after applying updated poses

    A small exp for 10_000_000 pcds and 2000 poses below on a 4090 (torch.float32):
        Vectorized
            Time taken: 0.03 seconds
            Initial memory: 208.53 MB
            Final memory:   368.54 MB
            Peak memory:    3328.94 MB

        Batched (500k/batch)
            Time taken: 0.04 seconds
            Initial memory: 368.54 MB
            Final memory:   488.54 MB
            Peak memory:    677.25 MB
    """
    updated_pcd_list = []
    N = pcd.shape[0]

    for i in range(0, N, batch_size):
        pcd_batch = pcd[i:i + batch_size]
        index_batch = index_pose[i:i + batch_size]
        updated = update_point_cloud_vectorized(
            pcd_batch, poses_old, poses_update, index_batch, keyframe_indices
        )
        updated_pcd_list.append(updated)

    return torch.cat(updated_pcd_list, dim=0)