#!/usr/bin/env bash
set -e
trap 'echo "[!] Caught SIGINT - killing children"; kill 0; exit 130' INT TERM

GAME=${GAME:-gomoku}
GPU_IDS=(0 1 2 3)
NPROC_PER_NODE=${#GPU_IDS[@]}
GPU_CSV=$(IFS=,; echo "${GPU_IDS[*]}")

read -r SELFPLAY_GAMES EVAL_GAMES < <(
python - "$GAME" <<'PY'
import sys

from config import get_search_config, get_train_config

game = sys.argv[1]
print(get_train_config(game).selfplay_games, get_search_config(game).eval_games)
PY
)

echo "[+] loop_ddp config: game=${GAME}, selfplay_games=${SELFPLAY_GAMES}, eval_games=${EVAL_GAMES}, gpus=${GPU_CSV}"

while true; do
  # Parallel self-play
  bash utils/selfplay_parallel.sh "$SELFPLAY_GAMES" "$GAME" "${GPU_IDS[@]}"

  # DDP training
  CUDA_VISIBLE_DEVICES="$GPU_CSV" \
    torchrun --nproc_per_node="$NPROC_PER_NODE" main.py train --game "$GAME" --ddp

  # Parallel arena evaluation (monitoring only in AZ mode)
  bash utils/eval_parallel.sh "$EVAL_GAMES" "$GAME" "${GPU_IDS[@]}"
done
