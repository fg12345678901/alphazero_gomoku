"""Utility to export AlphaZero Gomoku checkpoints to ONNX.

The script supports two ways of locating the checkpoint:

* Provide ``--checkpoint /absolute/path/to/model.pt`` (or a Windows path).
* Provide ``--name latest.pt`` to load ``config.MODEL_DIR / "latest.pt"``.
  If the ``--name`` value looks like a path it is used directly.

Example::

    python convert_to_onnx.py --name latest.pt --output exports/latest.onnx
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path, PureWindowsPath
from typing import Optional

import torch

from config import BOARD_SIZE, INPUT_PLANES, MODEL_DIR
from network.model import AlphaZeroNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an AlphaZero Gomoku PyTorch checkpoint to ONNX."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Full path to the PyTorch checkpoint (state_dict) to export.",
    )
    source.add_argument(
        "--name",
        type=str,
        help=(
            "Checkpoint filename located inside MODEL_DIR. "
            "If a path is supplied (for example C\\path\\to\\model.pt), "
            "it will be used directly."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(MODEL_DIR),
        help=(
            "Directory containing saved checkpoints when using --name. "
            "Defaults to config.MODEL_DIR."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the ONNX model. "
            "Defaults to the checkpoint name with a .onnx suffix in the same directory."
        ),
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=BOARD_SIZE,
        help=(
            "Board size used by the network. "
            "Should match the configuration used during training."
        ),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        help="ONNX opset version to target.",
    )
    return parser.parse_args()


def _name_looks_like_path(candidate: str) -> bool:
    """Return True if *candidate* should be treated as a path."""

    if not candidate:
        return False

    separators = {"/", "\\"}
    if any(sep in candidate for sep in separators):
        return True

    try:
        return bool(PureWindowsPath(candidate).drive)
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    checkpoint_path: Optional[Path]
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
    else:
        if _name_looks_like_path(args.name):
            checkpoint_path = Path(args.name).expanduser().resolve()
        else:
            model_dir = args.model_dir.expanduser().resolve()
            checkpoint_path = model_dir / args.name

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{checkpoint_path}"
        )

    if args.output is None:
        output_path = checkpoint_path.with_suffix(".onnx")
    else:
        output_path = args.output.expanduser()
        if output_path.is_dir():
            output_path = output_path / (checkpoint_path.stem + ".onnx")
    output_path = output_path.resolve()
    os.makedirs(output_path.parent, exist_ok=True)

    device = torch.device("cpu")
    model = AlphaZeroNet(board_size=args.board_size).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.zeros(1, INPUT_PLANES, args.board_size, args.board_size, device=device)

    input_names = ["board"]
    output_names = ["policy", "value"]
    dynamic_axes = {
        "board": {0: "batch_size"},
        "policy": {0: "batch_size"},
        "value": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path.as_posix(),
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
    )

    print(f"ONNX model exported to {output_path}")


if __name__ == "__main__":
    main()
