#!/usr/bin/env bash
set -e

GAME=${GAME:-gomoku}

while true; do
  python main.py selfplay --game "$GAME" --num-games 100
  python main.py selfplay --game "$GAME" --num-games 100
  python main.py train --game "$GAME" --updates 2500
  python main.py evaluate --game "$GAME" --num-games 100
done
