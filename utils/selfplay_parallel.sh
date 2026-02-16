#!/usr/bin/env bash
# Usage:
#   bash utils/selfplay_parallel.sh <total_games> [game] <gpu_ids...>
# Examples:
#   bash utils/selfplay_parallel.sh 200 0 1 2 3
#   bash utils/selfplay_parallel.sh 200 gomoku 0 1 2 3
set -e

TOTAL=$1
shift

if [ "$#" -eq 0 ]; then
  echo "Usage: bash utils/selfplay_parallel.sh <total_games> [game] <gpu_ids...>"
  exit 1
fi

if [[ "$1" =~ ^[0-9]+$ ]]; then
  GAME="gomoku"
else
  GAME="$1"
  shift
fi

GPUS=("$@")
N_GPU=${#GPUS[@]}
if [ "$N_GPU" -eq 0 ]; then
  echo "No GPU ids provided."
  exit 1
fi

EACH=$(( TOTAL / N_GPU ))
PIDS=()

echo "[+] Launching $TOTAL games = $EACH per GPU (${GPUS[*]}) for game=$GAME"

for g in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=$g \
    python main.py selfplay --game "$GAME" --num-games "$EACH" &
  PIDS+=($!)
done

for p in "${PIDS[@]}"; do
  wait "$p"
done
echo "[+] Self-play finished on GPUs ${GPUS[*]}"
