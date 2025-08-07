#!/usr/bin/env bash
# 并行评测：bash utils/eval_parallel.sh 100 0 1 2 3
set -e
trap 'kill 0; exit 130' INT TERM

TOTAL=$1; shift
GPUS=("$@"); N=${#GPUS[@]}

# Skip evaluation if there is no previous model
MODEL_DIR="models"
if [ $(ls "$MODEL_DIR"/net_*.pt 2>/dev/null | wc -l) -lt 2 ]; then
  echo "[Arena] no previous model, skipping evaluation"
  exit 0
fi

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
