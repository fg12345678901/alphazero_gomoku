#!/usr/bin/env bash
# Usage:
#   bash utils/eval_parallel.sh <total_games> [game] <gpu_ids...>
# Examples:
#   bash utils/eval_parallel.sh 100 0 1 2 3
#   bash utils/eval_parallel.sh 100 gomoku 0 1 2 3
set -e
trap 'kill 0; exit 130' INT TERM

TOTAL=$1
shift

if [ "$#" -eq 0 ]; then
  echo "Usage: bash utils/eval_parallel.sh <total_games> [game] <gpu_ids...>"
  exit 1
fi

if [[ "$1" =~ ^[0-9]+$ ]]; then
  GAME="gomoku"
else
  GAME="$1"
  shift
fi

GPUS=("$@")
N=${#GPUS[@]}
if [ "$N" -eq 0 ]; then
  echo "No GPU ids provided."
  exit 1
fi

MODEL_DIR="models"
if [ "$GAME" != "gomoku" ]; then
  MODEL_DIR="$MODEL_DIR/$GAME"
fi

if [ "$(ls "$MODEL_DIR"/net_*.pt 2>/dev/null | wc -l)" -lt 2 ]; then
  echo "[Arena] no previous model in $MODEL_DIR, skipping evaluation"
  exit 0
fi

PER=$(( TOTAL / N ))
DIR="arena_tmp"
rm -rf "$DIR"
mkdir "$DIR"

echo "[+] Arena $TOTAL games -> ${#GPUS[@]}x$PER for game=$GAME"

for g in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=$g \
    python main.py evaluate \
      --game "$GAME" \
      --num-games "$PER" \
      --no-update \
      --out "$DIR/result_gpu${g}.json" &
done
wait

python utils/arena_reduce.py "$DIR" --game "$GAME"
