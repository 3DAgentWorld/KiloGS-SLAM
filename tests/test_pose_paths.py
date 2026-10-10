"""Ground-truth paths follow scene configuration, independent of this machine."""
from pathlib import Path

from pose_eval import io_utils


def test_waymo_gt_beside_rgb(monkeypatch):
    monkeypatch.setattr(io_utils, 'load_waymo_gt', lambda path: Path(path))
    config = {'Dataset': {'color_path': '/example/waymo/100613/FRONT/rgb'}}
    assert io_utils.get_gt_poses('Waymo', '100613', config=config) == Path('/example/waymo/100613/FRONT/gt')


def test_kitti_gt_at_dataset_root(monkeypatch):
    monkeypatch.setattr(io_utils, 'load_kitti_gt', lambda path: Path(path))
    config = {'Dataset': {'color_path': '/example/KITTI_stereo/06/image_2'}}
    assert io_utils.get_gt_poses('KITTI', '06', config=config) == Path('/example/KITTI_stereo/poses/06.txt')


def test_kitti360_uses_same_frame_mapping(monkeypatch):
    monkeypatch.setattr(io_utils, 'load_kitti360_gt', lambda path, mapping: (Path(path), Path(mapping)))
    config = {'Dataset': {
        'color_path': '/example/KITTI-360/data_2d_raw/2013_05_28_drive_0005_sync/image_00/data_rect',
        'index_path': '/example/selection.txt',
    }}
    assert io_utils.get_gt_poses('KITTI-360', '05', config=config) == (
        Path('/example/KITTI-360/data_poses/2013_05_28_drive_0005_sync/cam0_to_world.txt'),
        Path('/example/selection.txt'),
    )


def test_gt_override(monkeypatch):
    monkeypatch.setattr(io_utils, 'load_kitti_gt', lambda path: Path(path))
    assert io_utils.get_gt_poses('KITTI', '06', config={'Dataset': {'gt_path': '/custom/poses.txt'}}) == Path('/custom/poses.txt')
