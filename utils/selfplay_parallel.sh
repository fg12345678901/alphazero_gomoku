#!/usr/bin/env bash
# Usage: bash utils/selfplay_parallel.sh <total_games> <gpu_ids...>
# e.g.   bash utils/selfplay_parallel.sh 200 0 1 2 3
set -e

TOTAL=$1; shift
GPUS=("$@")                 # e.g. (0 1 2 3)
N_GPU=${#GPUS[@]}
EACH=$(( TOTAL / N_GPU ))   # 四卡 200 局 -> 50 / 卡
PIDS=()

echo "[+] Launching $TOTAL games = $EACH per GPU (${GPUS[*]})"

for g in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=$g \
  python main.py selfplay --num-games "$EACH" &
  PIDS+=($!)
done

# 等全部子进程完成
for p in "${PIDS[@]}"; do
  wait "$p"
done
echo "[✓] Self-play finished on GPUs ${GPUS[*]}"
