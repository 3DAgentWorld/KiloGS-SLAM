import os
import numpy as np
from .utils import inverse_pose

# ==========================================
# Helper: Rotation Sanitization
# ==========================================

def sanitize_poses(poses):
    """
    Force the rotation part (top-left 3x3) of 4x4 poses to be valid SO(3).
    Using SVD to find the closest orthogonal matrix.
    R_clean = U * V.T
    """
    cleaned_poses = []
    for pose in poses:
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # SVD decomposition to fix rotation
        U, S, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        
        # Correct determinant if it's -1 (reflection)
        if np.linalg.det(R_clean) < 0:
            U[:, -1] *= -1
            R_clean = U @ Vt
            
        # Reconstruct 4x4 matrix
        new_pose = np.eye(4)
        new_pose[:3, :3] = R_clean
        new_pose[:3, 3] = t
        cleaned_poses.append(new_pose)
        
    return np.array(cleaned_poses)

# ==========================================
# Ground Truth Loaders
# ==========================================

def load_kitti_gt(path):
    """
    Load KITTI GT poses.
    Format: 12 floats per line (3x4 matrix), C2W.
    """
    poses = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = list(map(float, line.strip().split()))
            # Reshape to 3x4
            pose_3x4 = np.array(data).reshape(3, 4)
            # Add bottom row to make 4x4
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = pose_3x4
            poses.append(pose_4x4)
    return np.array(poses)

def load_waymo_gt(folder_path):
    """
    Load Waymo GT poses.
    Path: .../waymo/{scene}/FRONT/gt
    Format: One txt file per frame (e.g., 000001.txt), containing 4x4 matrix.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Waymo GT folder not found: {folder_path}")

    
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    # Sort filenames to preserve the image/pose correspondence.
    
    files.sort()
    
    poses = []
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            
            
            content = f.read().split()
            data = list(map(float, content))
            
            if len(data) != 16:
                 
                if len(data) == 12:
                    pose_4x4 = np.eye(4)
                    pose_4x4[:3, :] = np.array(data).reshape(3, 4)
                else:
                    raise ValueError(f"Invalid pose data length in {filename}: {len(data)}")
            else:
                pose_4x4 = np.array(data).reshape(4, 4)
            
            poses.append(pose_4x4)
            
    return np.array(poses)

def load_kitti360_gt(pose_path, mapping_path):
    """
    Load KITTI-360 GT poses using a Frame Mapping File.
    
    Args:
        pose_path: Path to cam0_to_world.txt (Raw GT)
        mapping_path: Path to Seq_XX_Frames.txt (Mapping from SLAM index to Original Frame ID)
    """
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"KITTI-360 GT file not found: {pose_path}")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}\nPlease run the frame extraction script first to generate this mapping.")

    print(f"[DEBUG] Loading Mapping from: {mapping_path}")
    
    # Map tracking indices to source image IDs using the frame selection.
    
    target_original_ids = []
    with open(mapping_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 1:
                target_original_ids.append(int(parts[0]))
    
    print(f"[DEBUG] Found {len(target_original_ids)} frames used in SLAM (e.g., starts with {target_original_ids[0]}).")

    # Index raw ground truth by source image ID.
    
    full_gt_dict = {}
    with open(pose_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 17: continue
            if any("nan" in p.lower() for p in parts): continue

            frame_id = int(parts[0])
            pose_data = list(map(float, parts[1:]))
            pose_4x4 = np.array(pose_data).reshape(4, 4)
            
            if np.all(pose_4x4 == 0): continue
            
            full_gt_dict[frame_id] = pose_4x4

    
    # Preserve the selected image order.
    final_poses = []
    missing_count = 0
    
    for original_id in target_original_ids:
        if original_id in full_gt_dict:
            final_poses.append(full_gt_dict[original_id])
        else:
            
            
            print(f"[WARNING] Pose for original frame {original_id} is missing in GT file!")
            missing_count += 1
            
            
            if len(final_poses) > 0:
                final_poses.append(final_poses[-1]) # Legacy fallback: repeat the preceding pose when GT is missing.
            else:
                final_poses.append(np.eye(4))

    if missing_count > 0:
        print(f"[WARNING] Total missing GT poses: {missing_count}")

    final_poses = np.array(final_poses)
    
    # Express ground truth relative to the first selected camera.
    if len(final_poses) > 0:
        
        first_pose_inv = np.linalg.inv(final_poses[0])
        
        
        for i in range(len(final_poses)):
            final_poses[i] = first_pose_inv @ final_poses[i]

    # 5. Sanitize
    return sanitize_poses(final_poses)

def load_giga_est(save_dir):
    """
    Load GigaSLAM estimated poses.
    """
    pose_path = os.path.join(save_dir, "poses_est.txt")
    idx_path = os.path.join(save_dir, "poses_idx.txt")
    
    if not os.path.exists(pose_path) or not os.path.exists(idx_path):
        raise FileNotFoundError(f"Estimated pose files not found in {save_dir}")
    
    # Load Indices
    with open(idx_path, 'r') as f:
        indices = [int(line.strip()) for line in f.readlines() if line.strip()]
    
    # Load Poses
    poses = []
    
    
    with open(pose_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = list(map(float, line.strip().split()))
            pose_c2w = np.array(data).reshape(4, 4)
            poses.append(pose_c2w)
            
    # Project estimated rotations onto SO(3) before evaluation.
    
    return sanitize_poses(np.array(poses)), indices

# ==========================================
# Unified Interface
# ==========================================

def get_gt_poses(dataset_name, scene_name, base_path=None, config=None):
    """Resolve ground truth from the run's dataset paths or a dataset root.

    Dataset.gt_path can override the inferred file/directory. Existing loaders
    retain their coordinate conventions and KITTI-360 frame-index mapping.
    """
    from pathlib import Path

    dataset = (config or {}).get("Dataset", {})
    color = Path(dataset["color_path"]) if dataset.get("color_path") else None
    root = Path(base_path or os.environ.get("KILOGS_DATA_ROOT", "datasets"))
    explicit = dataset.get("gt_path")
    name = dataset_name.lower().replace("-", "")
    if name == "kitti":
        dataset_root = color.parents[1] if color else root / "KITTI_stereo"
        path = Path(explicit) if explicit else dataset_root / "poses" / f"{int(scene_name):02d}.txt"
        return load_kitti_gt(path)
    if name == "waymo":
        path = Path(explicit) if explicit else (color.parent / "gt" if color else root / "waymo" / scene_name / "FRONT/gt")
        return load_waymo_gt(path)
    if name == "kitti360":
        dataset_root = color.parents[3] if color else root / "KITTI-360"
        sequence = f"{int(scene_name):04d}"
        path = Path(explicit) if explicit else dataset_root / "data_poses" / f"2013_05_28_drive_{sequence}_sync" / "cam0_to_world.txt"
        mapping = dataset.get("index_path", dataset_root / "Information" / f"Seq_{sequence}_Frames.txt")
        return load_kitti360_gt(path, mapping)
    raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

def get_est_poses(code_name, save_dir):
    if code_name == "giga":
        return load_giga_est(save_dir)
    else:
        raise NotImplementedError(f"Code type {code_name} not implemented yet.")