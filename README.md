# AlphaZero Gomoku

This repository provides a minimal PyTorch implementation of AlphaZero for Gomoku. It includes scripts for generating self-play data, training neural networks, and evaluating new models.

## Installation

```bash
pip install -r requirements.txt
```

## Self-Play

Generate training data by playing games against the current best model:

```bash
python main.py selfplay --num-games 100
```

## Training

Train a network from collected self-play data:

```bash
python main.py train --updates 2000
```

## Evaluation

Pit the latest model against the current best to decide whether to update it:

```bash
python main.py evaluate --num-games 400
```

## Continuous Loop

For a fully automated workflow you can run everything in a loop. Use `loop.sh`
for single-GPU training or `loop_mult.sh` for multi-GPU machines:

```bash
# single GPU
bash loop.sh

# multi GPU
bash loop_mult.sh
```

