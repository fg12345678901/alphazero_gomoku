from __future__ import annotations

from typing import Any

import torch

from config import CHANNELS, NUM_RES
from games.base import GameLike, GameSpec
from network.model import AlphaZeroNet


CHECKPOINT_FORMAT_VERSION = 2


def model_kwargs_from_spec(
    spec: GameSpec,
    channels: int = CHANNELS,
    blocks: int = NUM_RES,
) -> dict[str, int]:
    return {
        "board_size": spec.board_size,
        "action_size": spec.action_size,
        "input_planes": spec.input_planes,
        "channels": channels,
        "blocks": blocks,
    }


def build_model_for_game(
    game: GameLike,
    device: str | torch.device | None = None,
    channels: int = CHANNELS,
    blocks: int = NUM_RES,
) -> AlphaZeroNet:
    model = AlphaZeroNet(**model_kwargs_from_spec(game.getGameSpec(), channels=channels, blocks=blocks))
    if device is not None:
        model = model.to(device)
    return model


def parse_checkpoint_payload(payload: Any):
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        meta = payload.get("meta")
        return payload["state_dict"], meta if isinstance(meta, dict) else None
    return payload, None


def load_checkpoint(path: str, map_location: str | torch.device):
    payload = torch.load(path, map_location=map_location)
    return parse_checkpoint_payload(payload)


def validate_checkpoint_meta(meta: dict[str, Any] | None, expected_spec: GameSpec) -> None:
    if not meta:
        return

    mismatch_messages: list[str] = []
    if "game_name" in meta and meta["game_name"] != expected_spec.name:
        mismatch_messages.append(
            f"game_name checkpoint={meta['game_name']} runtime={expected_spec.name}"
        )
    if "board_size" in meta and int(meta["board_size"]) != expected_spec.board_size:
        mismatch_messages.append(
            f"board_size checkpoint={meta['board_size']} runtime={expected_spec.board_size}"
        )
    if "action_size" in meta and int(meta["action_size"]) != expected_spec.action_size:
        mismatch_messages.append(
            f"action_size checkpoint={meta['action_size']} runtime={expected_spec.action_size}"
        )
    if "input_planes" in meta and int(meta["input_planes"]) != expected_spec.input_planes:
        mismatch_messages.append(
            f"input_planes checkpoint={meta['input_planes']} runtime={expected_spec.input_planes}"
        )

    if mismatch_messages:
        joined = "; ".join(mismatch_messages)
        raise ValueError(f"Checkpoint metadata mismatch: {joined}")


def checkpoint_meta(
    game_spec: GameSpec,
    game_name: str,
    channels: int,
    blocks: int,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "game_name": game_name,
        "board_size": game_spec.board_size,
        "action_size": game_spec.action_size,
        "input_planes": game_spec.input_planes,
        "channels": channels,
        "blocks": blocks,
    }


def save_checkpoint(
    path: str,
    state_dict: dict[str, torch.Tensor],
    meta: dict[str, Any],
) -> None:
    torch.save({"state_dict": state_dict, "meta": meta}, path)
