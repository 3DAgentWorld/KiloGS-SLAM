#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
GPU_ID=${GPU_ID:-0}
SEQS=(13476 100613 106762 132384 152706 153495 158686 163453 405841)
# Dataset YAML controls matching and loop closure; use the active Python environment.
exec python scripts/run_dataset.py --dataset Waymo --gpu "$GPU_ID" --seqs "${SEQS[@]}" "$@"
