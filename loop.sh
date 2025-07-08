#!/usr/bin/env bash
while true; do
  python main.py selfplay  --num-games 100
  python main.py selfplay  --num-games 100
  python main.py train     --updates 2500
  python main.py evaluate  --num-games 100
done
