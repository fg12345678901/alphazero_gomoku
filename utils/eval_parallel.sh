#!/usr/bin/env bash
# 并行评测：bash utils/eval_parallel.sh 100 0 1 2 3
set -e
trap 'kill 0; exit 130' INT TERM

TOTAL=$1; shift
GPUS=("$@"); N=${#GPUS[@]}
PER=$(( TOTAL / N ))
DIR="arena_tmp"; rm -rf "$DIR"; mkdir "$DIR"

echo "[+] Arena $TOTAL games → ${#GPUS[@]}×$PER"

for g in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=$g \
  python main.py evaluate \
         --num-games "$PER" --no-update \
         --out "$DIR/result_gpu${g}.json" &
done
wait

python utils/arena_reduce.py "$DIR"
