import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch
import cv2
import os
from utils.logging_utils import Log

color_purple = '#602357'
color_dark_red = '#B6443F'

def plot_camera_poses(pred_poses, hight_light_poses=None, save_path='camera_poses_xz.png', xyz='xz'):
    """
    Plot scatter points of camera poses and save as PNG format, ensuring equal aspect ratio.

    Args:
    - pred_poses (torch.Tensor): Camera pose data with shape [N,7], [N,3], or [N,4,4].
      If shape is [N,4,4], it will be treated as transformation matrices and automatically extract translation vectors.
    - hight_light_poses (torch.Tensor or np.ndarray): Indices of camera poses to highlight, shape [n].
    - save_path (str): Path to save the output image, default 'camera_poses_xz.png'.
    - xyz (str): Coordinate plane to plot, options: 'xz' (default), 'xy', or 'yz'.
    """

    is_tensor = True
    if not isinstance(pred_poses, torch.Tensor):
        is_tensor = False

    if pred_poses.ndim == 3:
        if pred_poses.shape[1:] != (4, 4):
            raise ValueError("When input shape is [N,4,4], it must be 4x4 transformation matrices")
        translations = pred_poses[:, :3, 3]
    elif pred_poses.ndim == 2:
        if pred_poses.shape[1] < 3:
            raise ValueError("pred_poses must have shape [N,3], [N,7], or [N,4,4]")
        translations = pred_poses[:, :3]
    else:
        raise ValueError("Invalid dimensions for pred_poses, should be 2D or 3D tensor")
    
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if xyz == 'xz':
        x = translations[:, 0]
        z = translations[:, 2]
    elif xyz == 'xy':
        x = translations[:, 0]
        z = translations[:, 1]
    elif xyz == 'yz':
        x = translations[:, 1]
        z = translations[:, 2]
    else:
        raise ValueError("xyz parameter must be 'xz', 'xy', or 'yz'")
    
    if is_tensor:
        x = x.numpy()
        z = z.numpy()

    plt.figure()
    if hight_light_poses is not None:
        hight_light_poses = np.array(hight_light_poses)
        mask = np.ones(len(x), dtype=bool)
        mask[hight_light_poses] = False
        plt.scatter(x[mask], z[mask], c=color_purple, s=8, alpha=0.75, label='Camera Poses')
        plt.scatter(x[hight_light_poses], z[hight_light_poses], c=color_dark_red, s=20, label='Loop Closure')
    else:
        plt.scatter(x, z, c=color_purple, s=8, label='Camera Poses')

    plt.xlabel('x' if xyz in ['xz', 'xy'] else 'y')
    plt.ylabel('z' if xyz in ['xz', 'yz'] else 'y')
    plt.title(f'Camera Poses in {xyz.upper()} Plane')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    Log(f"Camera poses have been saved to path: {save_path}")


def load_poses_monogs(pose_file, inv = False):

    """Load ground truth poses (T_w_cam0) from file."""
    poses = []

    with open(pose_file, 'r') as f:
        lines = f.readlines()
        data = np.array([list(map(float, line.split())) for line in lines])


        for line in data:
            T_w_cam0 = line.reshape(4, 4)
            if inv:
                T_w_cam0 = np.linalg.inv(T_w_cam0)
            poses.append(T_w_cam0)

    return np.array(poses)

def load_poses_kitti(pose_file, index = False, inv = False):
    """Load ground truth poses (T_w_cam0) from file."""
    # pose_file = os.path.join(self.pose_path, self.sequence + '.txt')

    # Read and parse the poses
    poses = []
    idx = []

    with open(pose_file, 'r') as f:
        lines = f.readlines()
        data = np.array([list(map(float, line.split())) for line in lines])

        for line in data:
            if index == False:
                T_w_cam0 = line[:12].reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            else:
                T_w_cam0 = line[1:13].reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                idx.append(line[0])
            
            if inv == False:
                poses.append(T_w_cam0)
            else:
                poses.append(np.linalg.inv(T_w_cam0))
    if index == False:
        return np.array(poses)
    else:
        return np.array(poses), idx
    
def calculate_trajectory_length(poses):
    translations = poses[:, :3, 3]
    
    distances = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    
    total_length = np.sum(distances)
    
    return total_length

def draw_concat_keypoints(img1, keypoints1, img2, keypoints2, output_path):
    """
    Draw keypoints on two grayscale images and save vertically concatenated result
    
    Args:
        img1 (np.ndarray): First grayscale image, shape (h1, w)
        keypoints1 (np.ndarray): Keypoints for first image, shape (n, 2)
        img2 (np.ndarray): Second grayscale image, shape (h2, w)
        keypoints2 (np.ndarray): Keypoints for second image, shape (m, 2)
        output_path (str): Output filename
    """

    if img1.shape[1] != img2.shape[1]:
        raise ValueError("Image widths must be the same for concatenation")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for (x, y) in keypoints1:
        cv2.circle(img1_color, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)

    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for (x, y) in keypoints2:
        cv2.circle(img2_color, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)

    concatenated = cv2.vconcat([img1_color, img2_color])
    
    cv2.imwrite(output_path, concatenated)
