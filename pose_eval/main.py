import argparse
import os
import sys
from pathlib import Path
import yaml

# Ensure current directory is in python path to find local modules if run from elsewhere
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from pose_eval import io_utils, eval_core

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Pose Evaluation for SLAM")
    parser.add_argument("--dataset", type=str, default="KITTI", help="Dataset name (e.g., KITTI)")
    parser.add_argument("--code", type=str, default="giga", help="Codebase name (e.g., giga)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory containing estimated poses")
    parser.add_argument("--scene", type=str, required=True, help="Scene identifier (e.g., 07)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Start Evaluation: Dataset={args.dataset}, Code={args.code}, Scene={args.scene}")
    
    # 1. Load Ground Truth
    try:
        config_path = Path(args.save_dir) / "config.yml"
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else None
        all_gt_poses = io_utils.get_gt_poses(args.dataset, args.scene, config=config)
        print(f"Loaded GT poses: {len(all_gt_poses)} frames.")
    except Exception as e:
        print(f"Error loading GT: {e}")
        raise SystemExit(1)

    # 2. Load Estimated Poses and Indices
    try:
        est_poses, indices = io_utils.get_est_poses(args.code, args.save_dir)
        print(f"Loaded Estimated poses: {len(est_poses)} frames.")
        print(f"Loaded Indices: {len(indices)} frames.")
    except Exception as e:
        print(f"Error loading Estimates: {e}")
        raise SystemExit(1)
        
    # 3. Data Synchronization
    # Filter GT poses to match the keyframes in estimation
    # Check if indices are within bounds
    if not indices or min(indices) < 0 or max(indices) >= len(all_gt_poses):
        print(f"Error: Invalid estimate indices for GT length {len(all_gt_poses)}")
        raise SystemExit(1)
        
    gt_poses_matched = all_gt_poses[indices]
    
    assert len(gt_poses_matched) == len(est_poses), "GT and Est poses count mismatch after syncing!"
    
    # Prepare list to store log strings
    results_log = []
    
    # 4. Evaluation 1: No Scale Alignment (Rigid alignment only)
    save_path_noscale = os.path.join(args.save_dir, "pose_noscale.png")
    res_str_1 = eval_core.calculate_metrics_and_plot(
        gt_poses_matched, 
        est_poses, 
        indices, # Pass indices for plotting
        save_path_noscale, 
        do_scale_alignment=False,
        title_prefix="KiloGS-SLAM"
    )
    results_log.append(res_str_1)
    
    # 5. Evaluation 2: With Scale Alignment (Sim3)
    save_path_scale = os.path.join(args.save_dir, "pose_scale.png")
    res_str_2 = eval_core.calculate_metrics_and_plot(
        gt_poses_matched, 
        est_poses, 
        indices, # Pass indices for plotting
        save_path_scale, 
        do_scale_alignment=True,
        title_prefix="KiloGS-SLAM"
    )
    results_log.append(res_str_2)
    
    # 6. Save results to txt file
    txt_save_path = os.path.join(args.save_dir, "pose_eval.txt")
    with open(txt_save_path, "w") as f:
        for line in results_log:
            f.write(line + "\n")
            
    print("Evaluation Finished.")
    print(f"Numerical results saved to: {txt_save_path}")

if __name__ == "__main__":
    main()