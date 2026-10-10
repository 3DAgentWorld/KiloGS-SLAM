"""Run configured scenes sequentially and write the original per-scene summary format."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config_utils import load_config
from summary_result import process_dataset

DATASETS = {
    "Waymo": ("configs/Waymo/waymo_{}.yaml", "waymo"),
    "KITTI": ("configs/KITTI/kitti_{}.yaml", "kitti"),
    "KITTI-360": ("configs/KITTI-360/{}.yaml", "kitti360"),
}


def gpu_usage(gpu):
    raw = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    return [float(value.strip()) for value in raw.strip().split(",")]


def run_slam(command, gpu, log_path):
    memory, utilization = gpu_usage(gpu)
    if memory > 200 or utilization > 5:
        raise RuntimeError(f"GPU {gpu} is occupied: {memory} MiB, {utilization}% utilization")
    stopped = threading.Event()
    samples = [memory]

    def monitor():
        while not stopped.wait(0.2):
            try:
                samples.append(gpu_usage(gpu)[0])
            except (subprocess.SubprocessError, ValueError):
                pass

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    try:
        with log_path.open("w") as log:
            # Inherit the terminal process group so Ctrl-C reaches SLAM's cleanup.
            with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
    finally:
        stopped.set()
        worker.join()
    return max(samples)


def number(text, label):
    match = re.search(re.escape(label) + r":\s*([\d.]+)", text)
    return float(match[1]) if match else None


def write_summary(run, config, stdout, peak_memory):
    timing = (run / "time_memory.txt").read_text()
    core = number(timing, "SLAM time (before color refine)")
    refine = number(timing, "Color refinement time")
    if core is None or refine is None:
        raise RuntimeError(f"Missing full-mode timing in {run}")
    total = core + refine
    values = {
        "SLAM core time": core,
        "Color refine time": refine,
        "Total time": total,
        "FPS (frames / total_time)": config["SLAM"]["frame_num"] / total if total > 0 else None,
        "Total Gaussians after": number(stdout, "Total Gaussians after color refinement"),
        "Storage Size": number(stdout, "Total Storage Size"),
        "Peak GPU Memory": peak_memory,
    }
    for phase, label in (("before_opt", "PSNR before"), ("after_opt", "PSNR after")):
        metrics = json.loads((run / "psnr" / phase / "final_result.json").read_text())
        values[label] = metrics["mean_psnr"]
    lines = ["KiloGS-SLAM Summary"]
    lines.extend(f"{key}: {value if value is not None else 'N/A'}" for key, value in values.items())
    lines.append((run / "pose_eval.txt").read_text())
    (run / f"kf_{config['Training']['kf_iter_num']}_log.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seqs", nargs="+", required=True)
    parser.add_argument("--kf-iter-num", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Show resolved commands without using a GPU")
    args = parser.parse_args()
    if args.kf_iter_num < 1:
        parser.error("--kf-iter-num must be positive")
    os.chdir(ROOT)
    template, summary_name = DATASETS[args.dataset]
    for scene in args.seqs:
        source = template.format(scene)
        config = load_config(source)
        config["Training"]["kf_iter_num"] = args.kf_iter_num
        dataset = config["Dataset"]
        experiment = (Path(dataset["index_path"]).stem if dataset.get("index_path")
                      else "_".join(Path(dataset["color_path"]).parts[-3:]))
        suffix = "LC" if config["SLAM"]["loop_closure"] else "No-LC"
        run = ROOT / config["Results"]["save_dir"] / experiment / f"{datetime.now():%Y-%m-%d-%H-%M-%S-%f}-{suffix}"
        command = [sys.executable, "slam.py", "--config", source, "--mode", "full",
                   "--kf_iter_num", str(args.kf_iter_num), "--output_dir", str(run)]
        print(f"{args.dataset} {scene}: {config['tracking']['feature_mode']}, loop={config['SLAM']['loop_closure']}")
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        with tempfile.TemporaryDirectory(prefix="kilogs-") as temporary:
            log = Path(temporary) / "run.log"
            try:
                peak = run_slam(command, args.gpu, log)
            finally:
                if log.exists():
                    run.mkdir(parents=True, exist_ok=True)
                    (run / "run.log").write_text(log.read_text())
        (run / "resources.json").write_text(json.dumps({"peak_device_gpu_memory_mib": peak}, indent=2) + "\n")
        subprocess.run([sys.executable, "pose_eval/main.py", "--dataset", args.dataset,
                        "--code", "giga", "--scene", scene, "--save_dir", str(run)], check=True)
        write_summary(run, config, (run / "run.log").read_text(), peak)
        process_dataset(summary_name, ROOT / "results")


if __name__ == "__main__":
    main()
