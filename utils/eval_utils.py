import json
import os
import time

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.pose_utils import get_pose
from utils.logging_utils import Log
from utils.scale_diagnostics import (
    compute_depth_max,
    compute_depth_median,
    compute_depth_ratio_median,
    save_scale_diagnostics,
)

from utils.anchor_utils import anchor_in_frustum

import torchvision
import torch.nn.functional as F


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log(r"RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate

def plot_rendering_metrics(indices, psnr_list, ssim_list, lpips_list, iteration, save_dir):
    """Plot per-frame PSNR, SSIM and LPIPS before or after refinement."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    
    x = np.array(indices)
    y_psnr = np.array(psnr_list)
    y_ssim = np.array(ssim_list)
    y_lpips = np.array(lpips_list)
    
    
    mean_psnr = np.mean(y_psnr)
    mean_ssim = np.mean(y_ssim)
    mean_lpips = np.mean(y_lpips)


    plt.switch_backend('Agg') 
    
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    
    axs[0].plot(x, y_psnr, color='#1f77b4', linewidth=1.5, linestyle='-', marker='o', markersize=2, alpha=0.8)
    axs[0].set_title(f'PSNR (Mean: {mean_psnr:.4f} dB)', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('PSNR (dB)', fontsize=10)
    axs[0].set_xlabel('Frame Index', fontsize=12) 
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    
    axs[1].plot(x, y_ssim, color='#ff7f0e', linewidth=1.5, linestyle='-', marker='o', markersize=2, alpha=0.8)
    axs[1].set_title(f'SSIM (Mean: {mean_ssim:.4f})', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('SSIM', fontsize=10)
    axs[1].set_xlabel('Frame Index', fontsize=12) 
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    
    axs[2].plot(x, y_lpips, color='#2ca02c', linewidth=1.5, linestyle='-', marker='o', markersize=2, alpha=0.8)
    axs[2].set_title(f'LPIPS (Mean: {mean_lpips:.4f})', fontsize=12, fontweight='bold')
    axs[2].set_ylabel('LPIPS', fontsize=10)
    axs[2].set_xlabel('Frame Index', fontsize=12) 
    axs[2].grid(True, linestyle='--', alpha=0.6)
    
    
    plt.tight_layout()
    
    
    file_name = f"{iteration}.png"
    save_path = os.path.join(save_dir, file_name)
    
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Metric plot saved to: {save_path}")


def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):  
        return obj.cpu().tolist()  
    elif isinstance(obj, (list, tuple)):  
        return [tensor_to_serializable(item) for item in obj]  
    elif isinstance(obj, dict):  
        return {key: tensor_to_serializable(value) for key, value in obj.items()}  
    else:
        return obj  

def get_adaptive_figsize(h_ori, w_ori, rows=3, base_width=12):
    """Choose figure dimensions from the image aspect ratio."""
    aspect = h_ori / w_ori  
    fig_w = base_width
    fig_h = base_width * aspect * rows * 1 + 2  
    return (fig_w, fig_h)

def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    intrinsics,
    height,
    width,
    iteration="final",
    upsampling_method = 'bicubic'
):

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array, idx_array = [], [], [], []
    depth_scale_ratios = []

    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    img_save_path = os.path.join(save_dir, "img")
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path, exist_ok=True)

    viz_dir = os.path.join(save_dir, "viz")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    pose_txt_path = os.path.join(save_dir, 'poses_est.txt')
    pose_idx_txt_path = os.path.join(save_dir, 'poses_idx.txt')
    with open(pose_idx_txt_path, 'w') as f:
        for idx in frames.keys():
            f.write(f'{idx}' + '\n')

    with open(pose_txt_path, 'w') as f:
        for idx in frames.keys():
            saved_frame_idx.append(idx)
            frame = frames[idx]
            gt_image, _, gt_image_ori, _, _ = dataset[idx]

            gt_image_ori = gt_image_ori / 255

            h_ori, w_ori = gt_image_ori.shape[1], gt_image_ori.shape[2]

            pose_est = np.linalg.inv(gen_pose_matrix(frame.R, frame.T)) # W2C

            flat_matrix = pose_est.flatten()
            line = ' '.join(map(str, flat_matrix))
            f.write(line + '\n')


            opt_mask = torch.zeros(gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
            m = anchor_in_frustum(anchors=gaussians.get_anchor, 
                                        intrinsics=intrinsics, 
                                        pose=get_pose(frame), 
                                        cam_center=frame.camera_center,
                                        distance_lis = gaussians.distance_lis,
                                        levels=gaussians.get_level, 
                                        h=height, 
                                        w=width)
            opt_mask.bitwise_or_(m)

            render_pkg = render(
                frame, 
                gaussians, 
                pipe, 
                background, 
                visible_mask=opt_mask
            )
            render_depth_median = compute_depth_median(render_pkg["depth"])
            mono_depth_median = compute_depth_median(frame.depth)
            mono_depth_max = compute_depth_max(frame.depth)
            render_depth_max = compute_depth_max(render_pkg["depth"])
            depth_ratio_median = compute_depth_ratio_median(
                frame.depth,
                render_pkg["depth"],
                min_depth=0.1,
                upper_quantile=0.75,
            )
            depth_scale_ratios.append(
                {
                    "frame_idx": int(idx),
                    "ratio_median": depth_ratio_median,
                    "mono_depth_median": mono_depth_median,
                    "mono_depth_max": mono_depth_max,
                    "render_depth_median": render_depth_median,
                    "render_depth_max": render_depth_max,
                    "clip_min_depth": 0.1,
                    "clip_upper_quantile": 0.75,
                }
            )
            # render_depth_medians.append(
            #     {"frame_idx": int(idx), "median": render_depth_median}
            # )
            
            rendering = render_pkg['render']
            # torchvision.utils.save_image(
            #     rendering,
            #     os.path.join(img_save_path, f'downsample_img_{idx}.png')
            # )

            # Resize predictions to the original image size before evaluation.
            rendering = F.interpolate(
                rendering.unsqueeze(0),
                size=(h_ori, w_ori),
                mode=upsampling_method,
                align_corners=False,
            ).squeeze(0)

            # Replace non-finite predictions before computing metrics.
            rendering = torch.nan_to_num(rendering, nan=0.5, posinf=1.0, neginf=0.0)
            rendering = torch.clamp(rendering, 0.0, 1.0)

            # if iteration == "after_opt":
            #     torchvision.utils.save_image(
            #         rendering,
            #         os.path.join(img_save_path, f'img_{idx}.png')
            #     )

            
            # GT tensor (C,H,W)
            gt_image_tensor = torch.tensor(gt_image_ori, dtype=torch.float32, device="cuda")
            gt_image_tensor = torch.nan_to_num(gt_image_tensor, nan=0.5)
            gt_image_tensor = torch.clamp(gt_image_tensor, 0.0, 1.0)

            # Pred (C,H,W)
            pred_tensor = rendering

            
            gt = (gt_image_tensor.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
            pred = (pred_tensor.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)

            
            residual = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
            residual = np.clip(residual, 0, 255).astype(np.uint8)

            # Evaluate rendering quality at the original image resolution.
            valid_mask = (gt_image_tensor.mean(dim=0) > 0).float()   # (H,W)
            valid_mask_3 = valid_mask.unsqueeze(0)                   # (1,H,W)

            psnr_score = psnr(
                (pred_tensor * valid_mask_3).unsqueeze(0),
                (gt_image_tensor * valid_mask_3).unsqueeze(0)
            ).detach().cpu().item()

            ssim_score = ssim(
                pred_tensor.unsqueeze(0),
                gt_image_tensor.unsqueeze(0)
            ).detach().cpu().item()

            lpips_score = cal_lpips(
                pred_tensor.unsqueeze(0),
                gt_image_tensor.unsqueeze(0)
            ).detach().cpu().item()

            psnr_array.append(psnr_score)
            ssim_array.append(ssim_score)
            lpips_array.append(lpips_score)
            idx_array.append(idx)
            
            # Save ground truth, prediction and residual images.
            # plt.figure(figsize=(12, 15))
            # if iteration == "after_opt":
            #     figsize = get_adaptive_figsize(h_ori, w_ori, rows=3)
            #     plt.figure(figsize=figsize)

            #     plt.subplot(3, 1, 1)
            #     plt.imshow(gt)
            #     plt.title('Ground Truth')
            #     plt.axis('off')

            #     plt.subplot(3, 1, 2)
            #     plt.imshow(pred)
            #     plt.title('Rendered RGB')
            #     plt.axis('off')

            #     plt.subplot(3, 1, 3)
            #     plt.imshow(residual)
            #     plt.title('Residual')
            #     plt.axis('off')

            #     plt.figtext(
            #         0.5, 0.01,
            #         f"PSNR: {psnr_score:.2f}",
            #         ha="center", fontsize=14
            #     )
                
            #     save_path = os.path.join(viz_dir, f"{idx}.png")
            #     plt.tight_layout()
            #     plt.savefig(save_path, bbox_inches='tight')
            #     plt.close()

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    plot_rendering_metrics(
    indices=idx_array,
    psnr_list=psnr_array,
    ssim_list=ssim_array,
    lpips_list=lpips_array,
    iteration=iteration,  
    save_dir=save_dir
    )
    
    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    diag_path = save_scale_diagnostics(
        save_dir,
        f"depth_scale_ratio_{iteration}",
        depth_scale_ratios,
    )
    Log(f"[INFO] Saved depth scale ratios to {diag_path}")

    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
