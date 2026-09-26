#!/usr/bin/env bash
# Serialize ALL GPU work of this repo: one job, one card, never both.
# Usage: scripts/gpu_run.sh <command...>   (waits for the lock; picks A6000 if idle else A5000)
set -euo pipefail
LOCK="$(cd "$(dirname "$0")/.." && pwd)/runs/.gpu.lock"
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock 9
pick() {
  # prefer the A6000 (PCI index 1) when its used memory is low, else the A5000 (index 0)
  for idx in 1 0; do
    used=$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    free=$(( $(nvidia-smi -i "$idx" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ') - used ))
    if [ "$free" -gt 21000 ]; then echo "$idx"; return; fi
  done
  echo "NONE"
}
while true; do g=$(pick); [ "$g" != NONE ] && break; sleep 60; done
echo "[gpu_run] card=$g pid=$$ cmd=$*" >&2
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec nice -n 10 "$@"
