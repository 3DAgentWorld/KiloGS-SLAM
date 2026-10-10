import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

def convert_pose_numpy_to_opencv_vectorized(pose_numpy, inv=False):
    """
    Converts a batch of camera-to-world (C2W) transformation matrices into OpenCV quaternion format.
    
    If inv=True, the inverse of each pose matrix is calculated, i.e., it assumes the input is 
    in camera-to-world coordinates, and we compute the world-to-camera transformation.

    This function takes an array of 4×4 transformation matrices, extracts the translation components 
    (x, y, z), and converts the rotation matrices to quaternions in OpenCV format [qw, qx, qy, qz].

    Parameters:
    -----------
    pose_numpy: np.ndarray, shape (n, 4, 4)
        A batch of 4×4 transformation matrices representing the camera poses in world coordinates.

    inv: bool, optional, default=False
        If True, computes the inverse of each pose matrix (world-to-camera transformation).

    Returns:
    --------
    pose_opencv : np.ndarray, shape (n, 7)
        The transformed poses in OpenCV format, where each row has the structure:
        [x, y, z, qw, qx, qy, qz], with (x, y, z) as the translation vector and
        (qw, qx, qy, qz) as the quaternion representing rotation.
    """

    if inv:
        pose_numpy = np.linalg.inv(pose_numpy)

    translations = pose_numpy[:, :3, 3]  # Shape: (n, 3)

    rotations = pose_numpy[:, :3, :3]  # Shape: (n, 3, 3)

    # (w, x, y, z)
    quats = R.from_matrix(rotations).as_quat()  # Shape: (n, 4)

    pose_opencv = np.hstack([translations, quats])  # Shape: (n, 7)

    return pose_opencv

def convert_quat_opencv_to_c2w_vectorized(pose_opencv):
    """
    Converts OpenCV-format quaternion poses to Camera-to-World (C2W) matrices
    
    Parameters:
    -----------
    pose_opencv : np.ndarray, shape (n, 7)
        Pose array in OpenCV format, each row as [x, y, z, qw, qx, qy, qz]
    
    Returns:
    --------
    c2w_matrices : np.ndarray, shape (n, 4, 4)
        Corresponding C2W transformation matrices, each being a 4x4 homogeneous matrix
    """
    translations = pose_opencv[:, :3]  # shape (n, 3)
    
    #  qx,qy,qz,qw
    quats = pose_opencv[:, 3:]         # shape (n, 4)
    quats_scipy = np.roll(quats, shift=-1, axis=1)  # [qw, qx, qy, qz] -> [qx, qy, qz, qw]
    
    rotations = R.from_quat(quats_scipy).as_matrix()  # shape (n, 3, 3)
    
    c2w_matrices = np.zeros((pose_opencv.shape[0], 4, 4))
    c2w_matrices[:, :3, :3] = rotations
    c2w_matrices[:, :3, 3] = translations
    c2w_matrices[:, 3, 3] = 1.0
    
    return c2w_matrices


def poses_to_c2w_tensor(pose: torch.Tensor) -> torch.Tensor:
    assert pose.shape[1] == 7, "Input must have shape [n, 7]"

    t = pose[:, :3]  # [n, 3]
    q = pose[:, 3:]  # [n, 4] -> (qx, qy, qz, qw)

    # Normalize quaternion
    q = F.normalize(q, dim=1)

    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw),
        2 * (qx*qy + qz*qw),     1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw),
        2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx**2 + qy**2)
    ], dim=1).reshape(-1, 3, 3)

    c2w = torch.eye(4, device=pose.device).unsqueeze(0).repeat(pose.shape[0], 1, 1)  # [n, 4, 4]
    c2w[:, :3, :3] = R
    c2w[:, :3, 3] = t

    return c2w

def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged

def get_pose(camera):
    '''get updated pose just for visible mask'''

    T_w2c = torch.eye(4, device='cuda')
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    if camera.cam_trans_delta is not None and camera.cam_rot_delta is not None:
        tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)
        new_w2c = SE3_exp(tau) @ T_w2c

        return new_w2c
    else:
        return T_w2c
    

import torch
import math

def adjust_camera(R, t, back_step=7.0, lift_step=3.0, pitch_angle_deg=-15.0):
    """
    Adjusts the camera position and orientation by moving it backward, lifting it up, and applying a pitch rotation to look downward.
        for W2C only.

    Parameters:
    - R (torch.Tensor): The 3x3 rotation matrix representing the camera's orientation.
    - t (torch.Tensor): The 3x1 translation vector representing the camera's position in world space.
    - back_step (float): Distance to move the camera backward along its local z-axis. Default is 1.0.
    - lift_step (float): Distance to lift the camera along the world y-axis. Default is 1.0.
    - pitch_angle_deg (float): The angle in degrees by which to rotate the camera downward (pitch). Default is 10 degrees.
    
    Returns:
    - torch.Tensor: Updated rotation matrix after applying pitch rotation.
    - torch.Tensor: Updated translation vector after moving the camera.
    
    Example usage:
    >>> R = torch.eye(3)
    >>> t = torch.tensor([0.0, 0.0, 0.0])
    >>> R_new, t_new = adjust_camera(R, t, back_step=2.0, lift_step=1.5, pitch_angle_deg=15)
    >>> print(R_new)
    >>> print(t_new)
    """

    W2C = torch.eye(4, device=R.device, dtype=R.dtype)
    W2C[0:3, 0:3] = R
    W2C[0:3, 3] = t

    C2W = torch.linalg.inv(W2C)
    R = C2W[0:3, 0:3]
    t = C2W[0:3, 3]

    print('======')
    print(t)    
    # Move the camera backward along its local z-axis
    t = t - R[:, 2] * back_step  # R[:, 2] is the camera's z-axis

    # Lift the camera upward along the world y-axis
    t = t - torch.tensor([0.0, 1.0, 0.0], device=t.device) * lift_step  # Move along world y-axis
    print(t)   
    print('======')
    # Convert pitch angle from degrees to radians
    theta = torch.tensor(math.radians(pitch_angle_deg), dtype=R.dtype)

    # Create the rotation matrix for pitching (rotation around x-axis)
    Rx = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, torch.cos(theta), -torch.sin(theta)],
        [0.0, torch.sin(theta), torch.cos(theta)]
    ], device=t.device, dtype=R.dtype)

    # Update the camera's rotation matrix by applying the pitch rotation
    R = Rx @ R

    C2W[0:3, 0:3] = R
    C2W[0:3, 3] = t

    W2C = torch.linalg.inv(C2W)
    R = W2C[0:3, 0:3]
    t = W2C[0:3, 3]

    return R, t

import torch

def poses_to_quaternions(poses: torch.Tensor) -> torch.Tensor:
    """
    Convert camera-to-world (C2W) pose matrices to quaternions.
    
    Args:
        poses (torch.Tensor): Input poses as [n, 4, 4].
    
    Returns:
        torch.Tensor: Output quaternions as [n, 7] (tx, ty, tz, qx, qy, qz, qw).
    """
    n = poses.shape[0]
    R = poses[:, :3, :3]  # Extract rotation matrices
    t = poses[:, :3, 3]   # Extract translation vectors
    
    # Convert rotation matrices to quaternions
    quaternions = torch.zeros((n, 7), device=poses.device, dtype=poses.dtype)
    quaternions[:, 3:] = rotation_matrix_to_quaternion(R)  # Store quaternions (qx, qy, qz, qw)
    quaternions[:, :3] = t  # Store translations (tx, ty, tz)
    
    return quaternions


def quaternions_to_poses(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to camera-to-world (C2W) pose matrices.
    
    Args:
        quaternions (torch.Tensor): Input quaternions as [n, 7] (tx, ty, tz, qx, qy, qz, qw).
    
    Returns:
        torch.Tensor: Output poses as [n, 4, 4].
    """
    n = quaternions.shape[0]
    t = quaternions[:, :3]                                # Extract translation vectors (tx, ty, tz)
    R = quaternion_to_rotation_matrix(quaternions[:, 3:7]) # Convert quaternions to rotation matrices
    
    # Create pose matrices
    poses = torch.eye(4, device=quaternions.device, dtype=quaternions.dtype).repeat(n, 1, 1)
    poses[:, :3, :3] = R
    poses[:, :3, 3] = t
    
    return poses


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions.
    
    Args:
        R (torch.Tensor): Rotation matrices as [n, 3, 3].
    
    Returns:
        torch.Tensor: Quaternions as [n, 4] (qx, qy, qz, qw).
    """
    n = R.shape[0]
    quaternions = torch.zeros((n, 4), device=R.device, dtype=R.dtype)
    
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    trace = m00 + m11 + m22
    
    # Compute quaternions based on trace
    t0 = trace > 0
    t1 = (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2]) & ~t0
    t2 = (R[:, 1, 1] > R[:, 2, 2]) & ~t1 & ~t0
    t3 = ~t0 & ~t1 & ~t2
    
    s0 = torch.sqrt(trace[t0] + 1.0) * 2
    quaternions[t0, 3] = 0.25 * s0
    quaternions[t0, 0] = (R[t0, 2, 1] - R[t0, 1, 2]) / s0
    quaternions[t0, 1] = (R[t0, 0, 2] - R[t0, 2, 0]) / s0
    quaternions[t0, 2] = (R[t0, 1, 0] - R[t0, 0, 1]) / s0

    s1 = torch.sqrt(1.0 + R[t1, 0, 0] - R[t1, 1, 1] - R[t1, 2, 2]) * 2
    quaternions[t1, 3] = (R[t1, 2, 1] - R[t1, 1, 2]) / s1
    quaternions[t1, 0] = 0.25 * s1
    quaternions[t1, 1] = (R[t1, 0, 1] + R[t1, 1, 0]) / s1
    quaternions[t1, 2] = (R[t1, 0, 2] + R[t1, 2, 0]) / s1

    s2 = torch.sqrt(1.0 + R[t2, 1, 1] - R[t2, 0, 0] - R[t2, 2, 2]) * 2
    quaternions[t2, 3] = (R[t2, 0, 2] - R[t2, 2, 0]) / s2
    quaternions[t2, 0] = (R[t2, 0, 1] + R[t2, 1, 0]) / s2
    quaternions[t2, 1] = 0.25 * s2
    quaternions[t2, 2] = (R[t2, 1, 2] + R[t2, 2, 1]) / s2

    s3 = torch.sqrt(1.0 + R[t3, 2, 2] - R[t3, 0, 0] - R[t3, 1, 1]) * 2
    quaternions[t3, 3] = (R[t3, 1, 0] - R[t3, 0, 1]) / s3
    quaternions[t3, 0] = (R[t3, 0, 2] + R[t3, 2, 0]) / s3
    quaternions[t3, 1] = (R[t3, 1, 2] + R[t3, 2, 1]) / s3
    quaternions[t3, 2] = 0.25 * s3
    
    return quaternions


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.
    
    Args:
        quaternions (torch.Tensor): Quaternions as [n, 4] (qx, qy, qz, qw).
    
    Returns:
        torch.Tensor: Rotation matrices as [n, 3, 3].
    """
    qx, qy, qz, qw = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    R = torch.zeros((quaternions.shape[0], 3, 3), device=quaternions.device, dtype=quaternions.dtype)

    R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)

    return R