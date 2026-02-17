from __future__ import annotations

import argparse
import glob
import os

from config import DEVICE, GAME_NAME, get_train_config
from games.registry import available_games, create_game
from logging_setup import setup_logging
from runtime_paths import resolve_runtime_paths
from selfplay.selfplay import SelfPlayWorker
from trainer.trainer import Trainer

logger = setup_logging()


def latest_model(model_dir: str):
    files = sorted(glob.glob(os.path.join(model_dir, "net_*.pt")))
    return files[-1] if files else None


def _add_game_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--game",
        type=str,
        default=GAME_NAME,
        choices=available_games(),
        help=f"game type (default: {GAME_NAME})",
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sp = sub.add_parser("selfplay")
    _add_game_arg(p_sp)
    p_sp.add_argument("--num-games", type=int, default=None)

    p_tr = sub.add_parser("train")
    _add_game_arg(p_tr)
    p_tr.add_argument("--updates", type=int, default=None)
    p_tr.add_argument("--ddp", action="store_true", help="use DDP for training")
    p_tr.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)

    p_ev = sub.add_parser("evaluate")
    _add_game_arg(p_ev)
    p_ev.add_argument("--num-games", type=int, default=None, help="override default eval games")
    p_ev.add_argument("--out", type=str, default=None, help="save per-GPU arena result JSON")
    p_ev.add_argument(
        "--no-update",
        action="store_true",
        help="run arena only and skip metrics update (used by parallel eval workers)",
    )

    args = parser.parse_args()
    paths = resolve_runtime_paths(args.game)
    try:
        game = create_game(args.game)
    except ImportError as exc:
        parser.error(str(exc))

    if args.cmd == "selfplay":
        train_cfg = get_train_config(args.game)
        sp = SelfPlayWorker(
            net_path=latest_model(paths.model_dir),
            out_dir=paths.data_dir,
            num_games=args.num_games if args.num_games is not None else train_cfg.selfplay_games,
            game=game,
            game_name=args.game,
        )
        sp.run()

    elif args.cmd == "train":
        if args.local_rank is None and os.environ.get("LOCAL_RANK"):
            args.local_rank = int(os.environ["LOCAL_RANK"])
        tr = Trainer(
            distributed=args.ddp,
            local_rank=args.local_rank,
            game=game,
            game_name=args.game,
            paths=paths,
        )
        tr.train(args.updates)

    elif args.cmd == "evaluate":
        tr = Trainer(game=game, game_name=args.game, paths=paths)
        tr.evaluate_and_update(
            num_games=args.num_games,
            out=args.out,
            no_update=args.no_update,
        )


if __name__ == "__main__":
    logger.info("using device: %s", DEVICE)
    main()
