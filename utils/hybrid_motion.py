"""Motion-prior trigger and geometric validation, independent of ground truth."""
import cv2
import numpy as np


def rotation_angle(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def assess_motion(rotation, translation, inliers, config, prior=None, gap_ratio=1.0):
    translation = np.asarray(translation).reshape(3)
    norm = float(np.linalg.norm(translation))
    info = dict(inliers=int(inliers), translation_m=norm,
                rotation_deg=rotation_angle(rotation), prior_rotation_error_deg=None,
                prior_direction_error_deg=None, gap_ratio=float(gap_ratio))
    reasons = []
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        return True, ['nonfinite_pose'], info
    if inliers < config.get('min_inliers', 120):
        reasons.append('low_inliers')
    # Optional vehicle-specific assumptions are not generic matching-failure tests.
    if config.get('ghost_rotation', False) and norm < 0.05 and info['rotation_deg'] > 0.5:
        reasons.append('small_translation_rotation')
    if config.get('side_slip_deg') is not None and norm > config.get('min_translation_m', 0.1):
        angle = float(np.degrees(np.arccos(np.clip(abs(translation[2]) / norm, 0, 1))))
        info['optical_axis_angle_deg'] = angle
        if angle > config['side_slip_deg']:
            reasons.append('side_slip')
    if prior is not None:
        prev_r, prev_t = prior
        # Account for different numbers of original input frames between accepted poses.
        if config.get('normalize_frame_gap', True):
            vector = cv2.Rodrigues(prev_r)[0] * gap_ratio
            predicted_r = cv2.Rodrigues(vector)[0]
        else:
            predicted_r = prev_r
        info['prior_rotation_error_deg'] = rotation_angle(rotation @ predicted_r.T)
        if info['prior_rotation_error_deg'] > config.get('rotation_threshold_deg', 5.0):
            reasons.append('rotation_prior')
        prev_t = np.asarray(prev_t).reshape(3)
        prev_norm = np.linalg.norm(prev_t)
        minimum = config.get('min_translation_m', 0.1)
        if norm > minimum and prev_norm > minimum:
            cosine = np.dot(translation, prev_t) / (norm * prev_norm)
            info['prior_direction_error_deg'] = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            if info['prior_direction_error_deg'] > config.get('translation_angle_threshold_deg', 15.0):
                reasons.append('translation_prior')
    return bool(reasons), reasons, info


def valid_depth_matches(kp_ref, kp_cur, depth, max_depth=50):
    h, w = depth.shape
    valid = np.isfinite(kp_ref).all(axis=1) & np.isfinite(kp_cur).all(axis=1)
    for points in (kp_ref, kp_cur):
        valid &= ((points[:, 0] >= 0) & (points[:, 0] < w)
                  & (points[:, 1] >= 0) & (points[:, 1] < h))
    kp_ref, kp_cur = kp_ref[valid], kp_cur[valid]
    uv = kp_ref.astype(int)  # Same depth sampling convention as the existing PnP solver.
    z = depth[uv[:, 1], uv[:, 0]]
    valid = np.isfinite(z) & (z > 0) & (z < max_depth)
    return kp_ref[valid], kp_cur[valid], z[valid]


def reprojection_metrics(rotation, translation, kp_ref, kp_cur, depth, intrinsics, threshold=3.0):
    ref, cur, z = valid_depth_matches(kp_ref, kp_cur, depth)
    if len(ref) == 0:
        return dict(valid_matches=0, support=0.0, median_error_px=None)
    points = np.c_[ref, np.ones(len(ref))] @ np.linalg.inv(intrinsics).T * z[:, None]
    # Candidate is current->reference; projection requires reference->current.
    current = (points - np.asarray(translation).reshape(1, 3)) @ rotation
    projected = current @ intrinsics.T
    projected = projected[:, :2] / np.maximum(projected[:, 2:], 1e-9)
    errors = np.linalg.norm(projected - cur, axis=1)
    errors[current[:, 2] <= 0] = np.inf
    return dict(valid_matches=len(ref), support=float(np.mean(errors < threshold)),
                median_error_px=float(np.median(errors)))
