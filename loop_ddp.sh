#!/usr/bin/env bash
set -e
trap 'echo "[!] Caught SIGINT - killing children"; kill 0; exit 130' INT TERM

GAME=${GAME:-gomoku}

while true; do
  # Parallel self-play
  bash utils/selfplay_parallel.sh 1000 "$GAME" 0 1 2 3

  # DDP training
  torchrun --nproc_per_node=4 main.py train --game "$GAME" --ddp

  # Parallel arena evaluation (monitoring only in AZ mode)
  bash utils/eval_parallel.sh 104 "$GAME" 0 1 2 3
done
