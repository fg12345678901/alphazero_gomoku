#!/usr/bin/env bash
set -e
trap 'echo "[!] Caught SIGINT – killing children"; kill 0; exit 130' INT TERM

while true; do
  # ------- 并行 self-play -------
  bash utils/selfplay_parallel.sh 300 0 1 2 3

  # ------- DataParallel训练 -------
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train --updates 4000
  # CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train --updates 4000

  # ------- 并行 Arena -------
  bash utils/eval_parallel.sh 104 0 1 2 3
done

