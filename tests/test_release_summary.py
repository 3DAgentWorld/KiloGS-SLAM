"""Check publication defaults and the historical CSV layout."""
from pathlib import Path

import pandas as pd

from summary_result import parse_log_file, process_dataset
from utils.config_utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_defaults(monkeypatch):
    monkeypatch.chdir(ROOT)
    for dataset, mode, loop in (
        ("Waymo", "single", False),
        ("KITTI", "hybrid", False),
        ("KITTI-360", "single", True),
    ):
        for path in (ROOT / "configs" / dataset).glob("*.yaml"):
            config = load_config(str(path))
            assert config["SLAM"]["loop_closure"] == loop, path
            assert config["tracking"]["feature_mode"] == mode, path
            assert config["Results"]["save_dir"] == f"results/{dataset}", path
            assert config["Training"]["kf_iter_num"] == 3, path
            assert config["Hierarchical"]["color_refinement_iter"] == 20, path
            assert config["Hierarchical"]["n_offsets"] == 24, path
            assert config["Results"]["save_ckpt"] is False, path
            assert config["Hierarchical"]["rendering_width"] == (960 if dataset == "Waymo" else 480), path


def test_kitti360_summary_layout(tmp_path):
    for scene, psnr in (("0005", 20), ("0009", 24)):
        run = tmp_path / "KITTI-360" / f"Seq_{scene}_Frames" / "2026-10-10"
        run.mkdir(parents=True)
        (run / "kf_3_log.txt").write_text(
            f"PSNR after: {psnr} dB\n[Pose no scale alignment] ATE RMSE: 1e-2\n"
        )
    path = process_dataset("kitti360", tmp_path, tmp_path / "summary")
    frame = pd.read_csv(path, index_col=0)
    assert list(frame.columns) == ["0005", "0009", "avg"]
    assert frame.loc["PSNR After (dB)", "avg"] == 22
    assert frame.loc["ATE (No Scale)", "avg"] == 0.01
    assert not any("hybrid" in name.lower() for name in frame.index)


def test_missing_metric_is_not_zero(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("PSNR after: N/A dB\nSLAM core time: 12.34 s\n")
    metrics = parse_log_file(path)
    assert "PSNR After (dB)" not in metrics
    assert metrics["SLAM Time (s)"] == 12.34
