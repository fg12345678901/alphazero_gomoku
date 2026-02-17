#!/usr/bin/env bash
# Usage:
#   bash utils/selfplay_parallel.sh <total_games> [game] <gpu_ids...>
# Examples:
#   bash utils/selfplay_parallel.sh 200 0 1 2 3
#   bash utils/selfplay_parallel.sh 200 gomoku 0 1 2 3
set -e
trap 'kill 0; exit 130' INT TERM

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

WORKERS_PER_GPU=${SELFPLAY_WORKERS_PER_GPU:-2}
if ! [[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "SELFPLAY_WORKERS_PER_GPU must be a positive integer, got '$WORKERS_PER_GPU'"
  exit 1
fi

OMP_THREADS=${SELFPLAY_OMP_NUM_THREADS:-${OMP_NUM_THREADS:-1}}
MKL_THREADS=${SELFPLAY_MKL_NUM_THREADS:-${MKL_NUM_THREADS:-1}}
OPENBLAS_THREADS=${SELFPLAY_OPENBLAS_NUM_THREADS:-${OPENBLAS_NUM_THREADS:-1}}
NUMEXPR_THREADS=${SELFPLAY_NUMEXPR_NUM_THREADS:-${NUMEXPR_NUM_THREADS:-1}}

TOTAL_WORKERS=$(( N_GPU * WORKERS_PER_GPU ))
BASE=$(( TOTAL / TOTAL_WORKERS ))
REMAINDER=$(( TOTAL % TOTAL_WORKERS ))
PIDS=()
IDX=0

echo "[+] Launching $TOTAL self-play games for game=$GAME | gpus=${GPUS[*]} | workers_per_gpu=$WORKERS_PER_GPU | total_workers=$TOTAL_WORKERS"
echo "[+] Thread env: OMP=$OMP_THREADS MKL=$MKL_THREADS OPENBLAS=$OPENBLAS_THREADS NUMEXPR=$NUMEXPR_THREADS"

for g in "${GPUS[@]}"; do
  for ((w=0; w<WORKERS_PER_GPU; w++)); do
    GAMES=$BASE
    if [ "$IDX" -lt "$REMAINDER" ]; then
      GAMES=$(( GAMES + 1 ))
    fi
    IDX=$(( IDX + 1 ))

    if [ "$GAMES" -le 0 ]; then
      continue
    fi

    echo "[+] GPU $g worker $w -> $GAMES games"
    CUDA_VISIBLE_DEVICES=$g \
      OMP_NUM_THREADS=$OMP_THREADS \
      MKL_NUM_THREADS=$MKL_THREADS \
      OPENBLAS_NUM_THREADS=$OPENBLAS_THREADS \
      NUMEXPR_NUM_THREADS=$NUMEXPR_THREADS \
      python main.py selfplay --game "$GAME" --num-games "$GAMES" &
    PIDS+=($!)
  done
done

for p in "${PIDS[@]}"; do
  wait "$p"
done
echo "[+] Self-play finished on GPUs ${GPUS[*]}"
