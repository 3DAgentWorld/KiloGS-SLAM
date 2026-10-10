"""Summarize the latest run per scene, with metrics as rows and scenes as columns."""
import argparse
from datetime import datetime
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET_CONFIG = {
    "waymo": {"directory": "Waymo", "pattern": r"(\d+)_FRONT_rgb"},
    "kitti": {"directory": "KITTI", "pattern": r"KITTI_stereo_(\d+)_image_2"},
    "kitti360": {"directory": "KITTI-360", "pattern": r"(?:Seq_|KITTI-360_)(\d+)(?:_Frames)?"},
}
METRIC_MAPPING = {
    "SLAM core time": "SLAM Time (s)",
    "Color refine time": "cr time",
    "Total time": "Total Time (s)",
    "FPS (frames / total_time)": "FPS",
    "Total Gaussians after": "GS Num",
    "Storage Size": "Storage (MB)",
    "PSNR before": "PSNR Before (dB)",
    "PSNR after": "PSNR After (dB)",
    "Peak GPU Memory": "GPU Mem (MiB)",
}
NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def parse_log_file(file_path):
    """Read the batch log format used by the original evaluation scripts."""
    data = {}
    for line in Path(file_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        for prefix, name in METRIC_MAPPING.items():
            if line.startswith(prefix + ":"):
                match = re.search(r":\s*" + NUMBER, line)
                if match:
                    data[name] = float(match[1])
        for label, name in (("no scale", "ATE (No Scale)"), ("with scale", "ATE (Scale)")):
            if f"[Pose {label} alignment]" in line:
                match = re.search(r"ATE RMSE:\s*" + NUMBER, line)
                if match:
                    data[name] = float(match[1])
    return data


def get_latest_result_folder(scene_path):
    folders = [p for p in Path(scene_path).iterdir() if p.is_dir()]
    return max(folders, key=lambda p: p.stat().st_mtime) if folders else None


def process_dataset(dataset_name, results_root, output_dir=None):
    spec = DATASET_CONFIG[dataset_name]
    root = Path(results_root)
    # Accept both the new dataset hierarchy and historical flat result folders.
    if (root / spec["directory"]).is_dir():
        root = root / spec["directory"]
    if not root.is_dir():
        print(f"[Skip] Missing results directory: {root}")
        return None
    scenes = {}
    for folder in root.iterdir():
        match = re.fullmatch(spec["pattern"], folder.name)
        if not folder.is_dir() or not match:
            continue
        latest = get_latest_result_folder(folder)
        logs = sorted(latest.glob("kf_*_log.txt")) if latest else []
        if len(logs) != 1:
            print(f"[Skip] {folder.name}: expected one summary log in latest run, found {len(logs)}")
            continue
        metrics = parse_log_file(logs[0])
        if metrics:
            scenes[match[1]] = metrics
    if not scenes:
        print(f"[Skip] No summarized runs for {dataset_name} in {root}")
        return None
    frame = pd.DataFrame({scene: scenes[scene] for scene in sorted(scenes, key=int)})
    frame["avg"] = frame.mean(axis=1)
    destination = Path(output_dir) if output_dir is not None else ROOT / "results/summary"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{dataset_name}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    frame.round(4).to_csv(path, encoding="utf-8-sig")
    print(f"Saved summary: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=ROOT / "results", help="Results root or dataset directory")
    parser.add_argument("--dataset", choices=("all", *DATASET_CONFIG), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/summary")
    args = parser.parse_args()
    datasets = DATASET_CONFIG if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        process_dataset(dataset, args.path, args.output_dir)


if __name__ == "__main__":
    main()
