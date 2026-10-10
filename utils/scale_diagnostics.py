import json
import os

import torch


def compute_depth_median(depth_tensor, valid_mask=None):
    depth = torch.as_tensor(depth_tensor).detach()
    if depth.ndim == 4:
        depth = depth.squeeze(0)
    if depth.ndim == 3:
        depth = depth.squeeze(0)

    valid = torch.isfinite(depth) & (depth > 0)

    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, device=depth.device).detach()
        if mask.ndim == 4:
            mask = mask.squeeze(0)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        valid &= mask > 0

    valid_depth = depth[valid]
    if valid_depth.numel() == 0:
        return None

    return float(valid_depth.median().item())


def compute_depth_max(depth_tensor, valid_mask=None):
    depth = torch.as_tensor(depth_tensor).detach()
    if depth.ndim == 4:
        depth = depth.squeeze(0)
    if depth.ndim == 3:
        depth = depth.squeeze(0)

    valid = torch.isfinite(depth) & (depth > 0)

    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, device=depth.device).detach()
        if mask.ndim == 4:
            mask = mask.squeeze(0)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        valid &= mask > 0

    valid_depth = depth[valid]
    if valid_depth.numel() == 0:
        return None

    return float(valid_depth.max().item())


def compute_depth_ratio_median(depth_a, depth_b, valid_mask=None, min_depth=0.1, upper_quantile=0.75):
    depth_a = torch.as_tensor(depth_a).detach()
    depth_b = torch.as_tensor(depth_b).detach()

    if depth_a.ndim == 4:
        depth_a = depth_a.squeeze(0)
    if depth_a.ndim == 3:
        depth_a = depth_a.squeeze(0)

    if depth_b.ndim == 4:
        depth_b = depth_b.squeeze(0)
    if depth_b.ndim == 3:
        depth_b = depth_b.squeeze(0)

    valid = (
        torch.isfinite(depth_a)
        & torch.isfinite(depth_b)
        & (depth_a > 0)
        & (depth_b > 0)
    )

    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, device=depth_a.device).detach()
        if mask.ndim == 4:
            mask = mask.squeeze(0)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        valid &= mask > 0

    valid_a = depth_a[valid]
    valid_b = depth_b[valid]
    if valid_a.numel() == 0 or valid_b.numel() == 0:
        return None

    # Keep the previous unclipped version for reference:
    # ratios = (depth_a[valid] / depth_b[valid]).flatten()
    upper_a = torch.quantile(valid_a, upper_quantile)
    upper_b = torch.quantile(valid_b, upper_quantile)
    clipped_valid = valid & (depth_a >= min_depth) & (depth_b >= min_depth)
    clipped_valid &= depth_a <= upper_a
    clipped_valid &= depth_b <= upper_b

    ratios = (depth_a[clipped_valid] / depth_b[clipped_valid]).flatten()
    if ratios.numel() == 0:
        return None

    return float(ratios.median().item())


def save_scale_diagnostics(save_dir, key, entries):
    scale_dir = os.path.join(save_dir, "scale")
    os.makedirs(scale_dir, exist_ok=True)
    diag_path = os.path.join(scale_dir, "scale_diagnostics.json")
    data = {}
    if os.path.exists(diag_path):
        with open(diag_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    data[key] = entries

    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return diag_path
