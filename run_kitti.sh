#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
GPU_ID=${GPU_ID:-0}
SEQS=(00 02 03 04 05 06 07 08 09 10)
# Dataset YAML controls matching and loop closure; use the active Python environment.
exec python scripts/run_dataset.py --dataset KITTI --gpu "$GPU_ID" --seqs "${SEQS[@]}" "$@"
