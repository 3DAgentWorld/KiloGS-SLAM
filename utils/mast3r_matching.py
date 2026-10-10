"""MASt3R RGB adapter with explicit resize/crop coordinate transforms.

Preprocessing follows naver/dust3r utils/image.py (PIL resize, patch crop,
normalization to [-1, 1]); reciprocal matching follows naver/mast3r's example.
"""
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


def prepare_image(image, size=512, idx=0):
    """Accept uint8 RGB HWC/CHW or float RGB [0,1]. Return CPU view and geometry."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("MASt3R adapter expects one image")
        image = image[0]
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB, got {image.shape}")
    if image.dtype != np.uint8:
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating point RGB must be in [0,1]")
        image = np.rint(image * 255).astype(np.uint8)
    if size != 512:
        raise ValueError("This adapter validates the 512 MASt3R checkpoint at size=512")
    h0, w0 = image.shape[:2]
    w, h = [round(x * size / max(w0, h0)) for x in (w0, h0)]
    resized = Image.fromarray(image).resize(
        (w, h), Image.Resampling.LANCZOS if max(w0, h0) > size else Image.Resampling.BICUBIC)
    cx, cy = w // 2, h // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if w == h:  # Official load_images(square_ok=False).
        halfh = 3 * halfw // 4
    crop = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
    pixels = np.asarray(resized.crop(crop)).copy()
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
    tensor = (tensor - 0.5) / 0.5
    view = dict(img=tensor[None], true_shape=np.int32([[pixels.shape[0], pixels.shape[1]]]),
                idx=idx, instance=str(idx))
    geometry = dict(original_hw=(h0, w0), resized_hw=(h, w),
                    crop_xy=(crop[0], crop[1]), scale_xy=(w / w0, h / h0))
    return view, geometry


def to_original(points, geometry):
    """Invert PIL's center-aligned resize. Inputs/outputs use integer pixel centers."""
    points = np.asarray(points, dtype=np.float64)
    return ((points + np.asarray(geometry['crop_xy']) + 0.5)
            / np.asarray(geometry['scale_xy']) - 0.5).astype(np.float32)


class MASt3RMatcher:
    def __init__(self, config, device):
        repo = Path(config['repo_path'])
        weights = Path(config['weights_path'])
        if not repo.is_dir() or not weights.is_file():
            raise FileNotFoundError(f"Missing MASt3R source/weights: {repo}, {weights}")
        for path in (repo, repo / 'dust3r'):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from mast3r.model import AsymmetricMASt3R
        from dust3r.inference import inference
        from mast3r.fast_nn import fast_reciprocal_NNs
        self.model = AsymmetricMASt3R.from_pretrained(str(weights)).eval().to(device)
        self.inference = inference
        self.fast_nn = fast_reciprocal_NNs
        self.device = device
        self.config = config
        self.last_stats = {}

    @torch.inference_mode()
    def match(self, reference, current, depth=None, depth_threshold=65):
        size = self.config.get('resize_size', 512)
        ref, geom_ref = prepare_image(reference, size, 0)
        cur, geom_cur = prepare_image(current, size, 1)
        out = self.inference([(ref, cur)], self.model, self.device, batch_size=1, verbose=False)
        desc0 = out['pred1']['desc'].squeeze(0).detach()
        desc1 = out['pred2']['desc'].squeeze(0).detach()
        # MASt3R has dense per-pixel descriptors, not patch-center keypoints.
        for desc, view in ((desc0, ref), (desc1, cur)):
            if tuple(desc.shape[:2]) != tuple(view['img'].shape[-2:]):
                raise ValueError("Descriptor grid and cropped image resolution disagree")
        p0, p1 = self.fast_nn(desc0, desc1,
                              subsample_or_initxy1=self.config.get('match_stride', 2),
                              device=self.device, dist='dot', block_size=2**13)
        p0, p1 = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in (p0, p1)]
        valid = np.ones(len(p0), dtype=bool)
        border = self.config.get('border', 3)
        for points, desc in ((p0, desc0), (p1, desc1)):
            h, w = desc.shape[:2]
            valid &= ((points[:, 0] >= border) & (points[:, 0] < w - border)
                      & (points[:, 1] >= border) & (points[:, 1] < h - border))
        raw_count = len(p0)
        kp_ref = to_original(p0[valid], geom_ref)
        kp_cur = to_original(p1[valid], geom_cur)
        if depth is not None:
            h, w = depth.shape
            if (h, w) != tuple(geom_cur['original_hw']):
                raise ValueError("Current depth is not aligned with original current RGB")
            uv = np.rint(kp_cur).astype(int)
            valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            z = np.zeros(len(uv))
            z[valid] = depth[uv[valid, 1], uv[valid, 0]]
            valid &= np.isfinite(z) & (z > 0)
            if depth_threshold is not None:
                valid &= z < depth_threshold
            kp_ref, kp_cur = kp_ref[valid], kp_cur[valid]
        self.last_stats = dict(raw_matches=raw_count, filtered_matches=len(kp_ref),
                               ref_geometry=geom_ref, cur_geometry=geom_cur,
                               network_hw=list(desc0.shape[:2]))
        return kp_ref, kp_cur
