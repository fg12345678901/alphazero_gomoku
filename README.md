# AlphaZero Multi-Game Core (Gomoku + Go/OpenSpiel)

This repository contains a PyTorch AlphaZero training pipeline with:

- AlphaZero-style continuous updates (no AGZ gating/rejection)
- game abstraction (`games/`) for pluggable environments
- Gomoku support
- Go support via OpenSpiel adapter (`--game go`)

## Install

Base dependencies (Gomoku + core pipeline):

```bash
pip install -r requirements.txt
```

Go support (OpenSpiel backend):

```bash
pip install -r requirements-go.txt
```

Windows note: `open-spiel` may build from source (no prebuilt wheel for some Python versions).
If install fails, you need:
- CMake available in `PATH`
- a C++17 toolchain (Visual Studio Build Tools / clang-cl)

Linux server setup is strongly recommended for Go training.

## Quick Start

Gomoku:

```bash
python main.py selfplay --game gomoku --num-games 100
python main.py train --game gomoku --updates 2000
python main.py evaluate --game gomoku --num-games 200
```

Go (OpenSpiel):

```bash
python main.py selfplay --game go --num-games 40
python main.py train --game go --updates 500
python main.py evaluate --game go --num-games 60
```

## 19x19 Go Training (Linux / DDP)

Recommended environment variables before training:

```bash
export GO_BOARD_SIZE=19
export GO_KOMI=7.5
export GO_MCTS_SIMS=800
export GO_CPUCT=1.5
export GO_N_TEMP_MOVES=30
```

Single command for DDP training stage:

```bash
torchrun --nproc_per_node=4 main.py train --game go --ddp
```

Or use loop script with game switch:

```bash
GAME=go bash loop_ddp.sh
```

## Runtime Paths

- Gomoku keeps legacy paths:
  - `models/`, `data/`, `logs/`, `tb/`
- New games are namespaced:
  - `models/go/`, `data/go/`, `logs/go/`, `tb/go/`

## Checkpoint Format

New checkpoints include metadata (`game_name`, `board_size`, `action_size`, `input_planes`, etc.)
for compatibility checks. Older plain `state_dict` checkpoints are still readable.

## Game Interface

Core pipeline depends on `games.base.GameLike` and `GameSpec`.

Important: training/search code should use game interface methods only (for example,
`getCurrentPlayer`) and must not depend on board internals.

## Local Inference

- Gomoku web app: `python webapp/app.py`
- CLI evaluator: `python evaluate.py --model1 <path_to_model.pt> --human`

If you need Go inference/training locally, install `requirements-go.txt` first.
