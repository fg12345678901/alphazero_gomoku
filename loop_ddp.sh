#!/usr/bin/env bash
set -e
trap 'echo "[!] Caught SIGINT – killing children"; kill 0; exit 130' INT TERM

while true; do
  # ------- 并行 self-play -------
  bash utils/selfplay_parallel.sh 1000 0 1 2 3

  # ------- DDP 训练 -------
  torchrun --nproc_per_node=4 main.py train --ddp

  # ------- 并行 Arena -------
  bash utils/eval_parallel.sh 104 0 1 2 3

done
