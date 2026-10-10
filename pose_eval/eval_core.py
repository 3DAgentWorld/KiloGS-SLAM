import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
from pathlib import Path
import importlib.util
from evo.core import trajectory, sync, metrics
from evo.core.metrics import PoseRelation, Unit
from evo.tools import plot
import matplotlib.gridspec as gridspec
from .utils import get_relative_motion

ROOT_DIR = Path(__file__).resolve().parents[1]
scale_diag_spec = importlib.util.spec_from_file_location(
    "gigaslam_scale_diagnostics",
    ROOT_DIR / "utils" / "scale_diagnostics.py",
)
scale_diag_module = importlib.util.module_from_spec(scale_diag_spec)
scale_diag_spec.loader.exec_module(scale_diag_module)
save_scale_diagnostics = scale_diag_module.save_scale_diagnostics

def create_evo_traj(poses_np):
    """Convert numpy poses (Nx4x4) to evo Trajectory object"""
    timestamps = np.arange(len(poses_np))
    xyz = poses_np[:, :3, 3]
    return trajectory.PoseTrajectory3D(
        positions_xyz=xyz,
        orientations_quat_wxyz=None, 
        poses_se3=poses_np,
        timestamps=timestamps
    )


def compute_translation_scale_ratios(gt_poses, est_poses_aligned, indices):
    ratios = []
    for i in range(len(gt_poses) - 1):
        frame_idx = int(indices[i]) if indices is not None else int(i)
        next_frame_idx = int(indices[i + 1]) if indices is not None else int(i + 1)
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        est_rel = np.linalg.inv(est_poses_aligned[i]) @ est_poses_aligned[i + 1]

        gt_t = np.linalg.norm(gt_rel[:3, 3])
        est_t = np.linalg.norm(est_rel[:3, 3])
        ratio = None if (frame_idx == 0 or gt_t <= 1e-8) else float(est_t / gt_t)

        ratios.append(
            {
                "frame_idx": frame_idx,
                "next_frame_idx": next_frame_idx,
                "ratio": ratio,
                "gt_translation": float(gt_t),
                "pred_translation_aligned": float(est_t),
            }
        )
    return ratios

def calculate_metrics_and_plot(gt_poses, est_poses, indices, save_path, do_scale_alignment=False, title_prefix="GigaSLAM"):
    """
    Main evaluation function.
    Calculates ATE, t_rel, r_rel.
    Plots the 3x3 academic figure.
    Returns:
        result_str: The formatted result string for logging.
    """
    
    # 1. Prepare Trajectories
    traj_ref = create_evo_traj(gt_poses)
    traj_est = create_evo_traj(est_poses)
    
    # 2. Alignment
    # copy to avoid modifying original data for second run
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_ref_aligned = copy.deepcopy(traj_ref)
    
    # alignment (Umeyama). 
    # correct_scale=True -> Sim3 (7 DoF)
    # correct_scale=False -> SE3 (6 DoF)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est_aligned, traj_ref_aligned, correct_scale=do_scale_alignment
    )
    est_poses_aligned = traj_est_aligned.poses_se3
    
    # 3. Calculate ATE (APE in evo terms)
    # Translation Part
    ape_metric = metrics.APE(PoseRelation.translation_part)
    ape_metric.process_data((traj_ref_aligned, traj_est_aligned))
    ate_rmse = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    
    # Get APE error per frame for coloring
    ape_errors = ape_metric.error
    
    # 4. Calculate RPE (Relative errors)
    # t_rel
    rpe_trans = metrics.RPE(PoseRelation.translation_part, delta=1.0, all_pairs=False)
    rpe_trans.process_data((traj_ref_aligned, traj_est_aligned))
    t_rel_rmse = rpe_trans.get_statistic(metrics.StatisticsType.rmse)
    t_rel_per_frame = rpe_trans.error # array of length N-1
    
    # r_rel
    rpe_rot = metrics.RPE(PoseRelation.rotation_angle_rad, delta=1.0, all_pairs=False)
    rpe_rot.process_data((traj_ref_aligned, traj_est_aligned))
    r_rel_rmse = rpe_rot.get_statistic(metrics.StatisticsType.rmse)
    # Convert radians to degrees for reporting
    r_rel_per_frame = np.degrees(rpe_rot.error) 
    
    # 5. Print Result & Return String
    tag = "[Pose with scale alignment]" if do_scale_alignment else "[Pose no scale alignment]"
    result_str = f"{title_prefix}: {tag}: ATE RMSE: {ate_rmse:.4f}, t_rel: {t_rel_rmse:.4f}, r_rel: {r_rel_rmse:.4f}"
    print(result_str)
    
    # 6. Plotting (3x3 Grid)
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.5, 1, 1])
    
    # Extract aligned data for matplotlib
    gt_xyz = traj_ref_aligned.positions_xyz
    est_xyz = traj_est_aligned.positions_xyz
    
    # Row 1: Trajectories (XZ, XY, YX)
    
    # Plot 1: X-Z (Top view for KITTI usually)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'k--', linewidth=1, label='GT')
    sc1 = ax1.scatter(est_xyz[:, 0], est_xyz[:, 2], c=ape_errors, cmap='jet', s=5, label='Est')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('z (m)')
    ax1.set_title('Trajectory X-Z')
    ax1.legend()
    plt.colorbar(sc1, ax=ax1, label='ATE (m)')
    
    # Plot 2: X-Y
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'k--', linewidth=1, label='GT')
    sc2 = ax2.scatter(est_xyz[:, 0], est_xyz[:, 1], c=ape_errors, cmap='jet', s=5)
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Trajectory X-Y')
    
    # Plot 3: Y-X
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(gt_xyz[:, 1], gt_xyz[:, 0], 'k--', linewidth=1, label='GT')
    sc3 = ax3.scatter(est_xyz[:, 1], est_xyz[:, 0], c=ape_errors, cmap='jet', s=5)
    ax3.set_xlabel('y (m)')
    ax3.set_ylabel('x (m)')
    ax3.set_title('Trajectory Y-X')
    
    # Calculate GT Motion Magnitude for Rows 2 & 3
    gt_motion_t, gt_motion_r = get_relative_motion(gt_poses)
    
    # --- Correct X-Axis using Indices ---
    # RPE is calculated between frames (N poses -> N-1 errors).
    # We use indices[:-1] to represent "error starting from this frame".
    if indices is not None and len(indices) == len(est_poses):
        frames_x = np.array(indices[:-1])
    else:
        frames_x = np.arange(len(t_rel_per_frame))
    
    # Row 2: Translation Error Analysis
    # Plot 4: t_rel per frame
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(frames_x, t_rel_per_frame, 'b-', linewidth=1)
    ax4.set_title('Rel. Translation Error ($t_{rel}$)')
    ax4.set_ylabel('Error (m)')
    ax4.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 5: GT Translation Magnitude
    ax5 = fig.add_subplot(gs[1, 1])
    # Slice GT motion to match the length of errors (N-1)
    ax5.plot(frames_x, gt_motion_t[:-1], 'g-', linewidth=1)
    ax5.set_title('GT Translation Magnitude')
    ax5.set_ylabel('Motion (m)')
    ax5.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 6: Empty
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Row 3: Rotation Error Analysis
    # Plot 7: r_rel per frame
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(frames_x, r_rel_per_frame, 'r-', linewidth=1)
    ax7.set_title('Rel. Rotation Error ($r_{rel}$)')
    ax7.set_ylabel('Error (deg)')
    ax7.set_xlabel('Original Frame Index')
    ax7.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 8: GT Rotation Magnitude
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(frames_x, gt_motion_r[:-1], 'orange', linewidth=1)
    ax8.set_title('GT Rotation Magnitude')
    ax8.set_ylabel('Motion (deg)')
    ax8.set_xlabel('Original Frame Index')
    ax8.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 9: Empty
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    if do_scale_alignment:
        scale_entries = compute_translation_scale_ratios(
            gt_poses,
            est_poses_aligned,
            indices,
        )
        diag_path = save_scale_diagnostics(
            os.path.dirname(save_path),
            "pose_translation_scale_ratio_sim3",
            scale_entries,
        )
        print(f"Saved pose translation scale ratios to: {diag_path}")

    return result_str
