#!/usr/bin/env bash
set -e
trap 'echo "[!] Caught SIGINT - killing children"; kill 0; exit 130' INT TERM

GAME=${GAME:-gomoku}

while true; do
  # Parallel self-play
  bash utils/selfplay_parallel.sh 1000 "$GAME" 0 1 2 3

  # DataParallel training
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train --game "$GAME"

  # Parallel arena evaluation (monitoring only in AZ mode)
  bash utils/eval_parallel.sh 104 "$GAME" 0 1 2 3
done
