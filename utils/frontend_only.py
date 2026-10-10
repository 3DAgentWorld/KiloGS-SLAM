"""Pose-only execution using the same FrontEnd models and ClassicTracking API."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from munch import munchify

from utils.dataset import load_dataset
from utils.logging_utils import Log
from utils.slam_frontend import FrontEnd


def track_sequence(frontend, max_frames=None):
    """Run the original frame selection/depth/tracking path without map state.

    Frame IDs refer to dataset indices, including skipped images. History in
    ClassicTracking is indexed by accepted frames, and includes loop corrections.
    No interpolated poses are invented for skipped frames.
    """
    tracker = frontend.classic_tracking
    count = len(frontend.dataset)
    if max_frames is not None:
        count = min(count, max_frames)
    if count == 0:
        raise ValueError("Dataset has no images")
    indices = []
    loop_count = 0
    stages = dict(input_and_selection=0.0, depth=0.0, tracking=0.0)
    started = time.perf_counter()
    for frame_idx in range(count):
        tick = time.perf_counter()
        # Same dataset preprocessing as Camera.init_from_dataset in full mode.
        _, _, image, _, _ = frontend.dataset[frame_idx]
        flow = 0.0
        if frame_idx > 2:
            flow = tracker.motion_flow_keyframe(image, prev_kf=False)
            if flow < frontend.motion_thresh:
                stages["input_and_selection"] += time.perf_counter() - tick
                if (frame_idx + 1) % 100 == 0:
                    Log(f"[Frontend] input={frame_idx + 1}/{count}, poses={len(indices)}")
                continue
        stages["input_and_selection"] += time.perf_counter() - tick
        if tracker.kp_frame_id >= len(tracker.pose_history):
            raise ValueError("SLAM.n_max is smaller than the accepted frame count")

        tick = time.perf_counter()
        depth_input = torch.tensor(image).cuda().to(torch.uint8)
        prediction = frontend.depth_model.infer(depth_input, frontend.intrinsics_ori)
        depth = prediction["depth"].detach().squeeze().cpu().numpy()
        mask = prediction.get("mask")
        mask = mask.detach().squeeze().cpu().numpy() if mask is not None else None
        stages["depth"] += time.perf_counter() - tick

        tick = time.perf_counter()
        tracker.loop_kf(image, tracker.kp_frame_id)
        loop_flag, _, _ = tracker.update(
            image, depth, tracker.kp_frame_id, frame_idx,
            last_img=None, last_depth=None, flow_mag=flow, sky_mask=mask,
        )
        tracker.kp_frame_id += 1
        tracker.set_curr_keyframe(image)
        indices.append(frame_idx)
        loop_count += int(loop_flag)
        if frontend.viz and frame_idx - frontend.viz_frame_id_prev > frontend.viz_frame_interval:
            tracker.viz_pose(tracker.kp_frame_id)
            frontend.viz_frame_id_prev = frame_idx
        torch.cuda.synchronize()
        stages["tracking"] += time.perf_counter() - tick
        del prediction, depth_input
        if len(indices) % 50 == 0 or frame_idx == count - 1:
            Log(f"[Frontend] input={frame_idx + 1}/{count}, poses={len(indices)}, "
                f"elapsed={time.perf_counter() - started:.1f}s")

    poses = tracker.pose_history[:len(indices)].copy()
    if not np.isfinite(poses).all():
        raise ValueError("Non-finite frontend trajectory")
    return poses, np.asarray(indices, dtype=np.int64), {
        "input_frames": count,
        "tracked_frames": len(indices),
        "skipped_frames": count - len(indices),
        "loop_updates": loop_count,
        "tracking_loop_seconds": time.perf_counter() - started,
        "stage_seconds": stages,
    }


def run_frontend_only(config, save_dir):
    """Instantiate no Gaussian model, GUI, renderer, backend or mapping queues."""
    started = time.perf_counter()
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    config["Training"]["monocular"] = config["Dataset"]["sensor_type"] == "monocular"
    config["Results"]["save_dir"] = str(output)
    params = munchify(config["model_params"])
    dataset = load_dataset(params, params.source_path, config)
    frontend = None
    try:
        frontend = FrontEnd(config, dataset, save_dir=str(output))
        frontend.set_hyperparams()
        torch.cuda.synchronize()
        init_seconds = time.perf_counter() - started
        poses, indices, stats = track_sequence(frontend, config["SLAM"].get("max_frames"))
        # Same C2W 4x4 format and dataset indices as the existing pose evaluator.
        np.savetxt(output / "poses_est.txt", poses.reshape(-1, 16), fmt="%.12g")
        np.savetxt(output / "poses_idx.txt", indices, fmt="%d")
        np.savetxt(output / "poses_kitti.txt", poses[:, :3, :].reshape(-1, 12), fmt="%.12g")
        if frontend.classic_tracking.diagnoser.history["frame_id"]:
            frontend.classic_tracking.diagnoser.save_and_plot()
        stats.update(
            mode="frontend", status="completed", initialization_seconds=init_seconds,
            total_seconds=time.perf_counter() - started,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_cuda_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            pose_convention="camera_to_world", backend_started=False,
            gaussian_model_created=False,
        )
        stats["input_fps"] = stats["input_frames"] / stats["tracking_loop_seconds"]
        stats["tracked_fps"] = stats["tracked_frames"] / stats["tracking_loop_seconds"]
        (output / "frontend_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        Log(f"[Frontend] Complete: {stats['tracked_frames']} poses, "
            f"{stats['total_seconds']:.2f}s including initialization and export")
        return stats
    finally:
        # Retrieval is a frontend subprocess and may be enabled in either mode.
        retrieval = getattr(getattr(frontend, "classic_tracking", None), "retrieval", None)
        if retrieval is not None:
            if retrieval.proc.is_alive():
                retrieval.proc.terminate()
            retrieval.proc.join(timeout=5)
            for queue in (retrieval.in_queue, retrieval.out_queue):
                queue.cancel_join_thread()
                queue.close()
