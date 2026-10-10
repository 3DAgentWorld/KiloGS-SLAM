"""CPU regression: the pose-only driver must preserve the original tracking calls."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

# Legacy loss_utils constructs a CUDA module at import time; keep CPU tests CPU-only.
with patch.object(torch.nn.Module, 'cuda', lambda self, *a, **k: self):
    from utils.frontend_only import track_sequence
    from utils.slam_frontend import FrontEnd


class Dataset:
    width = height = 2
    fx = fy = 1
    cx = cy = 0

    def __len__(self):
        return 7

    def __getitem__(self, idx):
        return torch.ones(3, 2, 2), None, np.full((3, 2, 2), idx, dtype=np.uint8), None, None


class Tracker:
    def __init__(self):
        self.kp_frame_id = 0
        self.pose_history = np.repeat(np.eye(4)[None], 10, axis=0)
        self.calls = []
        self.references = []

    def motion_flow_keyframe(self, image, prev_kf):
        assert prev_kf is False
        return 0 if image[0, 0, 0] in (3, 5) else 30

    def loop_kf(self, image, frame_id):
        pass

    def update(self, image, depth, frame_id, true_frame_id, last_img, last_depth, flow_mag, sky_mask):
        self.calls.append((frame_id, true_frame_id, depth.copy(), sky_mask.copy(), flow_mag))
        self.pose_history[frame_id, 0, 3] = true_frame_id
        if true_frame_id == 6:
            self.pose_history[:frame_id, 0, 3] += 10  # Simulate a historical loop correction.
        return true_frame_id == 6, 1.0, (frame_id, 0)

    def set_curr_keyframe(self, image):
        self.references.append(int(image[0, 0, 0]))

    def get_loop_update(self):
        poses = self.pose_history[:self.kp_frame_id]
        return poses.copy(), poses.copy()


def frontend():
    queue = SimpleNamespace(empty=lambda: True)
    instance = SimpleNamespace(
        dataset=Dataset(), classic_tracking=Tracker(),
        depth_model=SimpleNamespace(infer=lambda image, intrinsics: {
            "depth": torch.full((1, 1, 2, 2), 2.0), "mask": torch.ones(1, 1, 2, 2)}),
        motion_thresh=24, intrinsics_ori=torch.eye(3), viz=False,
        viz_frame_id_prev=0, viz_frame_interval=50, device="cpu",
        q_vis2main=queue, frontend_queue=queue, pause=False,
        requested_init=False, single_thread=True, requested_keyframe=0,
        initialized=True, last_img=None, last_depth=None,
        cameras={}, keyframe_idx_list=[], kf_indices=[], current_window=[],
        reset=True, window_size=4, monocular=False, config={},
    )
    def initialize(idx, viewpoint):
        instance.reset = False
        instance.kf_indices.append(idx)
    instance.initialize = initialize
    instance.add_to_window = lambda idx, window: ([idx] + window[:3], None)
    instance.tracking = lambda *args: None
    instance.add_new_keyframe = lambda idx, init=False: instance.cameras[idx].depth[0]
    instance.request_keyframe = lambda *args: None
    instance.request_loop_update = lambda *args: None
    return instance


class FrontendParityTest(unittest.TestCase):
    def test_same_selection_depth_tracking_and_loop_corrected_export(self):
        legacy, pure = frontend(), frontend()
        def camera(dataset, idx, projection):
            color, _, image, _, _ = dataset[idx]
            return SimpleNamespace(uid=idx, original_image=color, image_ori=image,
                                   compute_grad_mask=lambda config: None)
        event = SimpleNamespace(record=lambda: None, elapsed_time=lambda other: 1000)
        with patch.object(torch.Tensor, "cuda", lambda self, *a, **k: self), \
             patch("torch.cuda.synchronize"), patch("torch.cuda.Event", return_value=event), \
             patch("utils.slam_frontend.time.sleep"), \
             patch("utils.slam_frontend.Camera.init_from_dataset", side_effect=camera), \
             patch("utils.slam_frontend.update_viewpoints_from_poses"):
            FrontEnd.run(legacy)
            poses, indices, stats = track_sequence(pure)
        np.testing.assert_array_equal(indices, [0, 1, 2, 4, 6])
        self.assertEqual(legacy.classic_tracking.references, pure.classic_tracking.references)
        for a, b in zip(legacy.classic_tracking.calls, pure.classic_tracking.calls):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        self.assertEqual(len(legacy.classic_tracking.calls), len(pure.classic_tracking.calls))
        np.testing.assert_array_equal(poses, legacy.classic_tracking.pose_history[:5])
        np.testing.assert_array_equal(poses[:, 0, 3], [10, 11, 12, 14, 6])
        self.assertEqual(stats["loop_updates"], 1)
        self.assertEqual(stats["skipped_frames"], 2)

    def test_frontend_dispatch_never_constructs_backend(self):
        import slam
        config = {"SLAM": {"mode": "frontend"}}
        original = copy.deepcopy(config)
        with patch("utils.frontend_only.run_frontend_only", return_value={"total_seconds": 1}) as run:
            instance = slam.SLAM(config, "/unused")
        run.assert_called_once_with(config, "/unused")
        self.assertFalse(hasattr(instance, "backend"))
        self.assertFalse(hasattr(instance, "gaussians"))
        self.assertEqual(config, original)


if __name__ == "__main__":
    unittest.main()
