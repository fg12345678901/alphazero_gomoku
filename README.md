# AlphaZero Gomoku (AZ-Core Refactor)

This repository contains a PyTorch AlphaZero training pipeline for Gomoku.

Recent refactor goals:

1. Move from AGZ-style model gating to AlphaZero-style continuous updates.
2. Introduce game abstraction so new games (for example Go) can be plugged in.
3. Keep current Gomoku behavior working while preparing for multi-game support.

## What changed

### 1) AlphaZero update logic

- The training loop now uses **continuous model updates** (single latest network).
- Arena evaluation is now **monitoring-only**:
  - no checkpoint rejection
  - no automatic model deletion
- Arena move selection is now greedy:
  - `temp=0`
  - `add_noise=False`

### 2) Game abstraction

- Added `games/` registry and `GameLike` protocol.
- Main pipeline (`main.py`, `selfplay`, `trainer`, `arena`) no longer hardcodes `GomokuGame`.
- Current supported game list:
  - `gomoku`

### 3) Per-game runtime paths

- Added `runtime_paths.py`.
- For `gomoku`, legacy paths are preserved:
  - `models/`, `data/`, `logs/`, `tb/`
- For future games, outputs are namespaced:
  - `models/<game>/`, `data/<game>/`, ...

## Install

```bash
pip install -r requirements.txt
```

## CLI

All core commands now accept `--game`:

```bash
python main.py selfplay --game gomoku --num-games 100
python main.py train --game gomoku --updates 2000
python main.py evaluate --game gomoku --num-games 400
```

If omitted, `--game` defaults to `gomoku`.

## Continuous loop scripts

Single GPU:

```bash
bash loop.sh
```

Multi GPU (DataParallel):

```bash
bash loop_mult.sh
```

Multi GPU (DDP):

```bash
bash loop_ddp.sh
```

You can override game by environment variable:

```bash
GAME=gomoku bash loop_mult.sh
```

## Notes for future Go integration

- The pipeline is now game-injectable.
- The remaining work for Go is to implement/register a Go game adapter and map its
  action/state representation to the current network and MCTS interfaces.
