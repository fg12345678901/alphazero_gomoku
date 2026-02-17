#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generic evaluation/play script for AlphaZero games.

Supported games are provided by ``games.registry`` (currently gomoku/go).
"""
from __future__ import annotations

import argparse
import os
import string
import sys
from pathlib import Path

import numpy as np

from config import DEVICE, GAME_NAME, get_model_config, get_search_config
from games.registry import available_games, create_game, normalize_game_name
from mcts.mcts import MCTS
from network.checkpoint import (
    build_model_for_game,
    load_checkpoint,
    validate_checkpoint_meta,
)


def _winner_from_state(game, board) -> int | None:
    result_for_black = float(game.getGameEnded(board, 1))
    if result_for_black == 0:
        return None
    if abs(result_for_black) < 1e-3:
        return 0
    return 1 if result_for_black > 0 else -1


def _go_columns(board, size: int) -> list[str]:
    cols: list[str] = []
    for action in range(size):
        try:
            coord = str(board.action_to_string(board.current_player(), int(action))).split()[-1]
        except Exception:
            cols = []
            break
        if not coord:
            cols = []
            break
        cols.append(coord[0].upper())
    if len(cols) == size and len(set(cols)) == size:
        return cols

    fallback = [c for c in string.ascii_uppercase if c != "I"]
    return fallback[:size]


def _columns_for_game(game_name: str, board, size: int) -> list[str]:
    if game_name == "go":
        return _go_columns(board, size)
    return list(string.ascii_uppercase[:size])


def _format_board(game_name: str, board, size: int) -> str:
    if game_name == "go":
        return str(board)

    cols = _columns_for_game(game_name, board, size)
    stone = {1: "X", -1: "O", 0: "."}
    header = "   " + " ".join(cols)
    rows = [header]
    matrix = np.asarray(board.board, dtype=np.int8)
    for i in range(size):
        row_num = str(i + 1).rjust(2, " ")
        row_cells = " ".join(stone[int(v)] for v in matrix[i])
        rows.append(f"{row_num} {row_cells}")
    return "\n".join(rows)


def _format_action(game_name: str, board, action: int, size: int) -> str:
    if game_name == "go":
        try:
            coord = str(board.action_to_string(board.current_player(), int(action))).split()[-1]
            return coord.upper()
        except Exception:
            pass
        if action == size * size:
            return "PASS"

    cols = _columns_for_game(game_name, board, size)
    row, col = divmod(int(action), size)
    if 0 <= col < len(cols):
        return f"{cols[col]}{row + 1}"
    return str(action)


def _parse_move_input(game_name: str, board, raw: str, size: int, action_size: int) -> int | None:
    text = raw.strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"q", "quit", "exit"}:
        raise KeyboardInterrupt

    # Flat action index (useful for debugging any game).
    if lowered.isdigit():
        idx = int(lowered)
        if 0 <= idx < action_size:
            return idx
        return None

    if game_name == "go":
        if lowered in {"pass", "p"}:
            return action_size - 1

        token = text.replace(" ", "")
        if len(token) >= 2 and token[0].isalpha() and token[1:].isdigit():
            cols = _columns_for_game(game_name, board, size)
            col = token[0].upper()
            if col not in cols:
                return None
            row = int(token[1:])
            if not (1 <= row <= size):
                return None
            action = (row - 1) * size + cols.index(col)
            if 0 <= action < action_size:
                return action
        return None

    # Gomoku: support "row col" / "row,col" and algebraic (A1).
    if "," in text or " " in text:
        sep = "," if "," in text else " "
        parts = [p for p in text.split(sep) if p]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            row = int(parts[0])
            col = int(parts[1])
            if 1 <= row <= size and 1 <= col <= size:
                return (row - 1) * size + (col - 1)

    token = text.upper().replace(" ", "")
    cols = _columns_for_game(game_name, board, size)
    if len(token) >= 2 and token[0].isalpha() and token[1:].isdigit():
        col_ch = token[0]
        if col_ch in cols:
            col = cols.index(col_ch)
            row = int(token[1:])
            if 1 <= row <= size:
                return (row - 1) * size + col

    return None


def _build_mcts(game_name: str, game, net, sims_override: int | None) -> MCTS:
    cfg = get_search_config(game_name)
    sims = int(sims_override) if sims_override and sims_override > 0 else cfg.mcts_sims
    return MCTS(
        game,
        net,
        sims=sims,
        cpuct=cfg.cpuct,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_eps=cfg.dirichlet_eps,
    )


def _load_net(game_name: str, game, model_path: str | None):
    model_cfg = get_model_config(game_name)
    net = build_model_for_game(
        game,
        device=DEVICE,
        channels=model_cfg.channels,
        blocks=model_cfg.num_res,
    )
    descriptor = "random(init)"

    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        state_dict, meta = load_checkpoint(str(path), map_location=DEVICE)
        validate_checkpoint_meta(
            meta,
            game.getGameSpec(),
            expected_channels=model_cfg.channels,
            expected_blocks=model_cfg.num_res,
        )
        net.load_state_dict(state_dict)
        descriptor = str(path)

    net.eval()
    return net, descriptor


def _play_human_vs_ai(game_name: str, game, net, human_color: str, sims: int | None) -> None:
    size = game.getGameSpec().board_size
    action_size = game.getActionSize()

    human_player = 1 if human_color == "black" else -1
    ai_player = -human_player
    mcts = _build_mcts(game_name, game, net, sims)

    board = game.getInitBoard()
    print(
        f"Human={human_color} ({human_player:+d}), AI={ai_player:+d}. "
        "Input: A1 / 'row col' / flat index; Go also supports 'pass'."
    )

    while True:
        print("\n" + _format_board(game_name, board, size))
        winner = _winner_from_state(game, board)
        if winner is not None:
            if winner == 0:
                print("Result: draw")
            elif winner == 1:
                print("Result: black wins")
            else:
                print("Result: white wins")
            return

        current = int(game.getCurrentPlayer(board))
        if current == human_player:
            while True:
                try:
                    raw = input("Your move> ")
                except EOFError:
                    raise KeyboardInterrupt
                action = _parse_move_input(game_name, board, raw, size, action_size)
                if action is None:
                    print("Invalid input, try again.")
                    continue
                valids = game.getValidMoves(board)
                if action < 0 or action >= action_size or int(valids[action]) == 0:
                    print("Illegal move, try again.")
                    continue
                break
            label = _format_action(game_name, board, action, size)
            print(f"You play: {label}")
        else:
            pi = mcts.get_action_probs(board, temp=0, add_noise=False)
            action = int(np.argmax(pi))
            label = _format_action(game_name, board, action, size)
            print(f"AI plays: {label}")

        board, _ = game.getNextState(board, action)


def _play_ai_vs_ai(
    game_name: str,
    game,
    net1,
    net2,
    games: int,
    sims: int | None,
    desc1: str,
    desc2: str,
) -> None:
    size = game.getGameSpec().board_size
    score = {"model1": 0, "model2": 0, "draw": 0}

    print(f"model1: {desc1}")
    print(f"model2: {desc2}")

    for game_idx in range(games):
        model1_is_black = (game_idx % 2 == 0)
        black_net = net1 if model1_is_black else net2
        white_net = net2 if model1_is_black else net1
        mcts_black = _build_mcts(game_name, game, black_net, sims)
        mcts_white = _build_mcts(game_name, game, white_net, sims)

        board = game.getInitBoard()
        print(f"\n=== Game {game_idx + 1} / {games} ===")

        while True:
            winner = _winner_from_state(game, board)
            if winner is not None:
                if winner == 0:
                    score["draw"] += 1
                    print("Result: draw")
                else:
                    model1_wins = (winner == 1 and model1_is_black) or (winner == -1 and not model1_is_black)
                    if model1_wins:
                        score["model1"] += 1
                        print("Result: model1 wins")
                    else:
                        score["model2"] += 1
                        print("Result: model2 wins")
                break

            current = int(game.getCurrentPlayer(board))
            mcts = mcts_black if current == 1 else mcts_white
            pi = mcts.get_action_probs(board, temp=0, add_noise=False)
            action = int(np.argmax(pi))
            label = _format_action(game_name, board, action, size)
            side = "B" if current == 1 else "W"
            who = "model1" if ((current == 1 and model1_is_black) or (current == -1 and not model1_is_black)) else "model2"
            print(f"{side} ({who}) -> {label}")
            board, _ = game.getNextState(board, action)

    total = sum(score.values())
    print("\n--- Final score ---")
    print(f"model1: {score['model1']}")
    print(f"model2: {score['model2']}")
    print(f"draw  : {score['draw']}")
    if total > 0:
        print(f"model1 win-rate: {score['model1'] / total:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero evaluator (multi-game)")
    parser.add_argument("--game", choices=available_games(), default=GAME_NAME, help="game type")
    parser.add_argument("--model1", default=None, help="path to first model checkpoint")
    parser.add_argument("--model2", default=None, help="path to second model checkpoint")
    parser.add_argument("--human", action="store_true", help="human vs AI mode")
    parser.add_argument("--human-color", choices=("black", "white"), default="white")
    parser.add_argument("--games", type=int, default=1, help="number of games for AI vs AI")
    parser.add_argument("--sims", type=int, default=None, help="override MCTS simulations")
    args = parser.parse_args()

    game_name = normalize_game_name(args.game)
    try:
        game = create_game(game_name)
    except ImportError as exc:
        parser.error(str(exc))

    try:
        net1, desc1 = _load_net(game_name, game, args.model1)
        model2_path = args.model2
        if model2_path is None and not args.human and args.model1 is not None:
            model2_path = args.model1
        net2, desc2 = _load_net(game_name, game, model2_path)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if args.human:
        if args.model2:
            parser.error("--model2 is not used in --human mode")
        print(f"Using model: {desc1} on device={DEVICE}")
        try:
            _play_human_vs_ai(game_name, game, net1, args.human_color, args.sims)
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)
        return

    if args.games < 1:
        parser.error("--games must be >= 1")

    _play_ai_vs_ai(game_name, game, net1, net2, args.games, args.sims, desc1, desc2)


if __name__ == "__main__":
    main()
