# # evaluate.py
# import argparse, os
# import torch
# import numpy as np
#
# from gomoku.game import GomokuGame
# from mcts.mcts import MCTS
# from network.model import AlphaZeroNet
# from config import *
#
# def human_vs_ai(model_path):
#     game = GomokuGame(BOARD_SIZE, N_IN_ROW)
#     net = AlphaZeroNet().to(DEVICE)
#     net.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     mcts = MCTS(game, net)
#
#     board = game.getInitBoard()
#     while True:
#         print(board)
#         if board.current_player == 1:
#             # AI 执黑先
#             pi = mcts.get_action_probs(board, temp=0)
#             move = np.argmax(pi)
#             print("AI 落子:", move)
#         else:
#             move = int(input("请输入落子索引 0‑224: "))
#         board, _ = game.getNextState(board, move)
#         winner = board.get_winner()            # 1=黑, ‑1=白, 0=平, None=继续
#         if winner is not None:
#             print(board)
#             if winner == 1:
#                 print("黑棋胜")
#             elif winner == -1:
#                 print("白棋胜")
#             else:
#                 print("平局")
#             break
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", required=True, help="模型文件 (.pt)")
#     args = parser.parse_args()
#     human_vs_ai(args.model)




#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py – Evaluation / play script for the AlphaZero Gomoku project.

Features
--------
1. Human vs AI
2. AI vs AI
3. Coordinate input & rich board display
See docstring and CLI help (-h) for details.
"""
from __future__ import annotations

import argparse
import os
import string
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch

from gomoku.game import GomokuGame
from mcts.mcts import MCTS
from network.model import AlphaZeroNet
from config import BOARD_SIZE, N_IN_ROW, DEVICE

# -----------------------------------------------------------------------------#
# Helper utilities
# -----------------------------------------------------------------------------#
_COORD_LETTERS = string.ascii_uppercase[:BOARD_SIZE]  # e.g. "ABCDEFGHIJKLMNO"


def _print_board(board) -> None:
    """Print board with coordinates (letters = columns, numbers = rows)."""
    stone = {1: "●", -1: "○", 0: "·"}
    header = "   " + " ".join(_COL for _COL in _COORD_LETTERS)
    print(header)
    for i in range(board.size):
        row_num = str(i + 1).rjust(2, " ")
        row_str = " ".join(stone[int(s)] for s in board.board[i])
        print(f"{row_num} {row_str}")
    print(f"Current turn: {'Black (●)' if board.current_player == 1 else 'White (○)'}\n")


def _parse_move(s: str, size: int) -> Optional[Tuple[int, int]]:
    """
    Convert user input to (row, col). Accepted formats:

      H8 / h8       -> algebraic (letter + row number, 1‑based)
      8 8 , 8,8     -> whitespace or comma separated (1‑based row col)
      112           -> flat index (0‑(size^2‑1)) for兼容旧脚本
    """
    s = s.strip().lower()

    # flat index
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < size * size:
            return divmod(idx, size)
        return None

    # row/col separated
    if "," in s or " " in s:
        for sep in (",", " "):
            if sep in s:
                parts = [p for p in s.split(sep) if p]
                break
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            r, c = map(int, parts)
            if 1 <= r <= size and 1 <= c <= size:
                return r - 1, c - 1
        return None

    # algebraic
    if len(s) >= 2 and s[0].isalpha() and s[1:].isdigit():
        col = _COORD_LETTERS.find(s[0].upper())
        row = int(s[1:]) - 1
        if 0 <= col < size and 0 <= row < size:
            return row, col

    return None


def _load_net(path: str | Path) -> AlphaZeroNet:
    """Load network weights to DEVICE."""
    net = AlphaZeroNet().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net


# -----------------------------------------------------------------------------#
# Human vs AI
# -----------------------------------------------------------------------------#
def _human_loop(model_path: str | Path, human_color: str) -> None:
    colour_map = {"black": 1, "white": -1}
    human_player = colour_map[human_color]
    ai_player = -human_player

    game = GomokuGame(BOARD_SIZE, N_IN_ROW)
    board = game.getInitBoard()

    net = _load_net(model_path)
    mcts = MCTS(game, net)

    print(f"Human plays {human_color}. Enter moves like \"H8\" / \"8 8\" / \"7,7\".\n")

    while True:
        _print_board(board)

        if board.current_player == human_player:
            # – Human move –
            while True:
                try:
                    raw = input("Your move: ")
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.")
                    sys.exit(0)

                coord = _parse_move(raw, board.size)
                if coord is None:
                    print("Cannot parse input – try again.")
                    continue
                if board.board[coord] != 0:
                    print("Occupied – try again.")
                    continue
                move = board.coord_to_move(*coord)
                break
        else:
            # – AI move –
            pi = mcts.get_action_probs(board, temp=0)
            move = int(np.argmax(pi))
            coord = board.move_to_coord(move)
            print(f"AI move: {_COORD_LETTERS[coord[1]]}{coord[0]+1}")

        board, _ = game.getNextState(board, move)
        winner = board.get_winner()  # 1=black, -1=white, 0=draw, None=ongoing
        if winner is not None:
            _print_board(board)
            if winner == 1:
                print("Black (●) wins!")
            elif winner == -1:
                print("White (○) wins!")
            else:
                print("Draw.")
            break


# -----------------------------------------------------------------------------#
# AI vs AI
# -----------------------------------------------------------------------------#
def _ai_vs_ai(model1_path: str | Path,
              model2_path: str | Path,
              games: int = 1) -> None:
    """
    Let two checkpoints play `games` games, swapping colours each game.
    """
    net1, net2 = _load_net(model1_path), _load_net(model2_path)
    game = GomokuGame(BOARD_SIZE, N_IN_ROW)
    mcts1, mcts2 = MCTS(game, net1), MCTS(game, net2)

    score = {"model1": 0, "model2": 0, "draw": 0}
    for g in range(1, games + 1):
        print(f"\n=== Game {g} ===")
        board = game.getInitBoard()
        black_is_model1 = (g % 2 == 1)  # 轮换黑棋

        while True:
            _print_board(board)
            current_mcts = mcts1 if (board.current_player == 1) == black_is_model1 else mcts2
            pi = current_mcts.get_action_probs(board, temp=0)
            move = int(np.argmax(pi))
            coord = board.move_to_coord(move)
            colour = "Black" if board.current_player == 1 else "White"
            who = "model1" if current_mcts is mcts1 else "model2"
            print(f"{colour} ({who}) -> {_COORD_LETTERS[coord[1]]}{coord[0]+1}")

            board, _ = game.getNextState(board, move)
            winner = board.get_winner()
            if winner is not None:
                _print_board(board)
                if winner == 0:
                    print("Draw.")
                    score["draw"] += 1
                else:
                    if (winner == 1 and black_is_model1) or (winner == -1 and not black_is_model1):
                        score["model1"] += 1
                        print("model1 wins.")
                    else:
                        score["model2"] += 1
                        print("model2 wins.")
                break

    # 统计
    total = sum(score.values())
    print("\n--- Final score ---")
    for k, v in score.items():
        print(f"{k:<7}: {v:>3}")
    if total:
        print(f"model1 win‑rate: {score['model1'] / total:.1%}")


# -----------------------------------------------------------------------------#
# CLI entry
# -----------------------------------------------------------------------------#
def main() -> None:
    parser = argparse.ArgumentParser(description="Gomoku AlphaZero evaluator")
    parser.add_argument("--model1", required=True, help="Path to first model (.pt)")
    parser.add_argument("--model2", help="Path to second model (.pt). "
                                         "Omit for self‑play unless --human is used.")
    parser.add_argument("--human", action="store_true",
                        help="Activate human‑vs‑AI mode (model1 is the AI).")
    parser.add_argument("--human-color", choices=("black", "white"), default="white",
                        help="Your colour when --human is set (default: black).")
    parser.add_argument("--games", type=int, default=1,
                        help="Number of games for AI‑vs‑AI (default: 1).")
    args = parser.parse_args()

    # 快速校验
    if args.human and args.model2:
        parser.error("Cannot specify --model2 when --human is set.")
    if not args.human and args.games < 1:
        parser.error("--games must be >= 1.")

    if args.human:
        _human_loop(args.model1, args.human_color)
    else:
        second = args.model2 or args.model1
        _ai_vs_ai(args.model1, second, games=args.games)


if __name__ == "__main__":
    main()
