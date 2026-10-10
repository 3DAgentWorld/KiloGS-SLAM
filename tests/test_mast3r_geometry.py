import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np

from utils.mast3r_matching import prepare_image, to_original
from utils.hybrid_motion import assess_motion, reprojection_metrics


class MatchingGeometryTest(unittest.TestCase):
    def test_kitti_crop_and_distinct_axis_scales(self):
        view, geometry = prepare_image(np.zeros((376, 1241, 3), dtype=np.uint8))
        self.assertEqual(tuple(view['img'].shape), (1, 3, 144, 512))
        self.assertEqual(geometry['resized_hw'], (155, 512))
        self.assertEqual(geometry['crop_xy'], (0, 5))
        self.assertNotEqual(*geometry['scale_xy'])
        self.assertTrue((view['img'] == -1).all())
        points = np.array([[0, 0], [100, 70], [511, 143]], dtype=float)
        recovered = to_original(points, geometry)
        forward = (recovered + 0.5) * geometry['scale_xy'] - 0.5 - geometry['crop_xy']
        np.testing.assert_allclose(forward, points, atol=3e-5)

    def test_rotation_prior_accounts_for_accepted_frame_gap(self):
        one_deg = cv2.Rodrigues(np.array([0.0, np.radians(1), 0.0]))[0]
        six_deg = cv2.Rodrigues(np.array([0.0, np.radians(6), 0.0]))[0]
        cfg = dict(rotation_threshold_deg=5, translation_angle_threshold_deg=15)
        bad, reasons, info = assess_motion(six_deg, [0, 0, 1], 200, cfg,
                                           (one_deg, [0, 0, 1]), gap_ratio=6)
        self.assertFalse(bad, reasons)
        self.assertAlmostEqual(info['prior_rotation_error_deg'], 0, places=5)

    def test_translation_angle_and_near_stationary_case(self):
        cfg = dict(rotation_threshold_deg=5, translation_angle_threshold_deg=15)
        direction = np.array([[np.sin(np.radians(16))], [0], [np.cos(np.radians(16))]])
        bad, reasons, _ = assess_motion(np.eye(3), direction, 200, cfg, (np.eye(3), [0, 0, 1]))
        self.assertTrue(bad)
        self.assertIn('translation_prior', reasons)
        bad, reasons, _ = assess_motion(np.eye(3), direction * 0.01, 200, cfg, (np.eye(3), [0, 0, 1]))
        self.assertFalse(bad, reasons)

    def test_current_to_reference_pnp_projection_convention(self):
        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1.]])
        ref = np.array([[40., 40.], [60., 60.]])
        # Camera moves +1m in x at z=10m; pixels shift -10 in the current view.
        cur = ref - [10, 0]
        depth = np.full((100, 100), 10.)
        metrics = reprojection_metrics(np.eye(3), np.array([[1.], [0], [0]]), ref, cur, depth, K)
        self.assertEqual(metrics['support'], 1.)
        self.assertAlmostEqual(metrics['median_error_px'], 0)

    def test_failed_heavy_pnp_cannot_overwrite_valid_base_pose(self):
        import torch
        with patch.object(torch.nn.Module, 'cuda', lambda self, *a, **k: self):
            from utils.visual_odometry import ClassicTracking
        tracker = ClassicTracking.__new__(ClassicTracking)
        tracker.track_cfg = {'hybrid': {}}
        tracker.config = {'SLAM': {'3d2d_thread': 1}}
        tracker.accepted_frame_ids = [0, 1]
        tracker.K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1.]])
        tracker.cam = SimpleNamespace(mat=tracker.K)
        tracker.prev_depth = np.full((100, 100), 10.)
        tracker.prev_rgb = np.zeros((3, 100, 100), dtype=np.uint8)
        xx, yy = np.meshgrid(np.arange(20, 80, 5), np.arange(20, 80, 5))
        points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        tracker.heavy_matching = lambda *a: (points, points - [10, 0])
        tracker.heavy_matcher = SimpleNamespace(last_stats={})
        tracker.diagnoser = SimpleNamespace(log_extra=lambda *a: None)
        with tempfile.TemporaryDirectory() as directory:
            tracker.save_dir = directory
            with patch('utils.visual_odometry.compute_pose_3d2d_parallel', return_value=(np.eye(3), np.zeros((3, 1)), 0)):
                R, t, scale, _, mode = tracker._hybrid_refine(
                    tracker.prev_rgb, tracker.prev_depth, 2, np.eye(3), np.array([[0.], [0], [1]]), 2., 0, 0)
            np.testing.assert_array_equal(t.reshape(3), [0, 0, 1])
            self.assertEqual(scale, 2.)
            self.assertEqual(mode, 0)
            record = json.loads((Path(directory) / 'hybrid_events.jsonl').read_text())
            self.assertTrue(record['triggered'])
            self.assertFalse(record['accepted'])

    def test_linear_memory_epipolar_residual_matches_original(self):
        import torch
        with patch.object(torch.nn.Module, 'cuda', lambda self, *a, **k: self):
            from utils.visual_odometry import compute_fundamental_residual
        rng = np.random.default_rng(123)
        p0, p1 = rng.normal(size=(2, 57, 2))
        F = rng.normal(size=(3, 3))
        m0, m1 = np.c_[p0, np.ones(len(p0))].T, np.c_[p1, np.ones(len(p1))].T
        a, b = F @ m0, F.T @ m1
        reference = (a.T @ m1).diagonal()**2 / (np.sum(a[:2]**2,axis=0) + np.sum(b[:2]**2,axis=0))
        np.testing.assert_allclose(compute_fundamental_residual(F,p0,p1), reference, atol=1e-12, rtol=1e-12)

    def test_degenerate_correspondences_fail_without_crashing(self):
        import torch
        with patch.object(torch.nn.Module, 'cuda', lambda self, *a, **k: self):
            from utils.visual_odometry import compute_pose_2d2d_parallel
        points = np.full((32, 2), 50., dtype=np.float32)
        camera = SimpleNamespace(cx=50., cy=50., fx=100., mat=np.eye(3))
        R, t, count = compute_pose_2d2d_parallel(points, points, camera)
        np.testing.assert_array_equal(R, np.eye(3))
        np.testing.assert_array_equal(t, np.zeros((3, 1)))
        self.assertEqual(count, 0)

    def test_valid_pure_rotation_can_recover_failed_base(self):
        import torch
        with patch.object(torch.nn.Module, 'cuda', lambda self, *a, **k: self):
            from utils.visual_odometry import ClassicTracking
        tracker = ClassicTracking.__new__(ClassicTracking)
        tracker.track_cfg = {'hybrid': {}}
        tracker.config = {'SLAM': {'3d2d_thread': 1}}
        tracker.accepted_frame_ids = [0, 1]
        tracker.K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1.]])
        tracker.cam = SimpleNamespace(mat=tracker.K)
        tracker.prev_depth = np.full((100, 100), 10.)
        tracker.prev_rgb = np.zeros((3, 100, 100), dtype=np.uint8)
        xx, yy = np.meshgrid(np.arange(20, 80, 5), np.arange(20, 80, 5))
        ref = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        rotation = cv2.Rodrigues(np.array([0., 0., np.radians(15)]))[0]
        cur = (ref - [50, 50]) @ rotation[:2, :2] + [50, 50]
        tracker.heavy_matching = lambda *a: (ref, cur)
        tracker.heavy_matcher = SimpleNamespace(last_stats={})
        tracker.diagnoser = SimpleNamespace(log_extra=lambda *a: None)
        with tempfile.TemporaryDirectory() as directory:
            tracker.save_dir = directory
            with patch('utils.visual_odometry.compute_pose_3d2d_parallel',
                       return_value=(rotation, np.zeros((3, 1)), len(ref))):
                R, t, scale, count, mode = tracker._hybrid_refine(
                    tracker.prev_rgb, tracker.prev_depth, 2, np.eye(3), np.zeros((3, 1)), 1., 0, 1)
            np.testing.assert_array_equal(R, rotation)
            np.testing.assert_array_equal(t, np.zeros((3, 1)))
            self.assertEqual((scale, count, mode), (1., len(ref), 2))
            event = json.loads((Path(directory) / 'hybrid_events.jsonl').read_text())
            self.assertTrue(event['accepted'])


if __name__ == '__main__':
    unittest.main()
