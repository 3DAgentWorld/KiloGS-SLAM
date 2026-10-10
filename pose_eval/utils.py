import numpy as np

def inverse_pose(pose_matrix):
    """
    Inverse a 4x4 pose matrix.
    If input is W2C, returns C2W, and vice versa.
    """
    return np.linalg.inv(pose_matrix)

def get_relative_motion(poses):
    """
    Compute relative translation and rotation magnitude between consecutive frames.
    Used for analyzing if large motion causes large errors.
    Returns:
        trans_mags: list of translation magnitudes (meters)
        rot_mags: list of rotation magnitudes (degrees)
    """
    trans_mags = []
    rot_mags = []
    
    for i in range(len(poses) - 1):
        P1 = poses[i]
        P2 = poses[i+1]

        P_rel = np.linalg.inv(P1) @ P2
        
        # Translation magnitude
        t_mag = np.linalg.norm(P_rel[:3, 3])
        
        # Rotation magnitude (using trace for quick angle calculation)
        R_rel = P_rel[:3, :3]
        trace = np.trace(R_rel)
        trace = np.clip(trace, -1.0, 3.0) # Numerical stability

        angle = np.arccos((trace - 1) / 2.0)
        
        trans_mags.append(t_mag)
        rot_mags.append(np.degrees(angle))
        
    # Pad the last element to match length
    trans_mags.append(0)
    rot_mags.append(0)
    
    return np.array(trans_mags), np.array(rot_mags)