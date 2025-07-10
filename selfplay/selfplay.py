# selfplay/selfplay.py
from __future__ import annotations
import os, pickle, random, time
import numpy as np
from tqdm import trange
from typing import List
import torch

from config import *
from gomoku.game import GomokuGame
from network.model import AlphaZeroNet
from mcts.mcts import MCTS
import logging

logger = logging.getLogger(__name__)


class SelfPlayWorker:
    def __init__(self, net_path: str | None, out_dir: str, num_games: int):
        os.makedirs(out_dir, exist_ok=True)
        self.game = GomokuGame(BOARD_SIZE, N_IN_ROW)
        self.net = AlphaZeroNet().to(DEVICE)
        if net_path and os.path.exists(net_path):
            self.net.load_state_dict(torch.load(net_path, map_location=DEVICE))
            logger.info(f"Loaded model {net_path}")
        self.net.eval()  # inference mode
        self.mcts = MCTS(self.game, self.net)
        self.num_games = num_games
        self.examples: List = []

    def run(self):
        for _ in trange(self.num_games, desc="Self‑play"):
            self.play_single()
        # timestamp = int(time.time())
        timestamp = f"{int(time.time() * 1000)}_{os.getpid()}"

        fname = os.path.join(DATA_DIR, f"selfplay_{timestamp}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(self.examples, f)
        logger.info(f"Saved {len(self.examples)} examples to {fname}")

    def play_single(self):
        board = self.game.getInitBoard()

        # 为当前对局重新生成树，避免不同对局之间相互污染
        self.mcts = MCTS(self.game, self.net)

        examples = []
        step = 0
        while True:
            temp = SELFPLAY_TEMPERATURE if step < N_TEMP_MOVES else 0
            pi = self.mcts.get_action_probs(board, temp=temp, add_noise=True)
            canonical = self.game.getCanonicalForm(board, board.current_player)
            examples.append((canonical, pi, board.current_player))
            move = np.random.choice(self.game.getActionSize(), p=pi)
            board, _ = self.game.getNextState(board, move)
            # from gomoku.display import display
            # display(board)
            # print(board.current_player)
            result = self.game.getGameEnded(board, board.current_player)
            # print(result)
            if result != 0:
                # 回填 z 和进行数据增强
                final_examples = []
                for planes, pi, player in examples:
                    # 首先，计算出该局面的最终价值 z
                    z = result if player == board.current_player else -result

                    # 调用 getSymmetries 来获取所有对称的样本
                    for sym_planes, sym_pi in self.game.getSymmetries(planes, pi):
                        final_examples.append((sym_planes, sym_pi, z))

                        # 如果加的话加这里

                self.examples.extend(final_examples)
                return
            step += 1
