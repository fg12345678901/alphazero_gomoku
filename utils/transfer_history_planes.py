# utils/transfer_history_planes.py
"""
迁移脚本：
  - 将旧模型(3输入通道)的权重映射到新的带历史信息的网络
  - 保存到 models/net_<timestamp>.pt
用法：
  python utils/transfer_history_planes.py [old_checkpoint.pt]
"""
from __future__ import annotations
import os, glob, time
from pathlib import Path
import torch

# ----- 项目内 import -----
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from network.model import AlphaZeroNet
from config import MODEL_DIR, DEVICE, HISTORY_STEPS


def latest_ckpt() -> str | None:
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "net_*.pt")))
    return files[-1] if files else None


def migrate(old_path: str | None = None) -> None:
    if old_path is None:
        old_path = latest_ckpt()
    if not old_path or not os.path.isfile(old_path):
        raise FileNotFoundError("未找到旧模型，请提供 checkpoint 路径。")
    print(f"[+] Loading old checkpoint: {old_path}")

    old_sd = torch.load(old_path, map_location=DEVICE)
    new_net = AlphaZeroNet().to(DEVICE)
    new_sd = new_net.state_dict()

    # ------- conv1 权重映射 -------
    conv_old = old_sd["conv.weight"]  # (out_c, 3, k, k)
    conv_new = new_sd["conv.weight"]  # (out_c, 2*HISTORY_STEPS+1, k, k)
    with torch.no_grad():
        for i in range(HISTORY_STEPS):
            conv_new[:, i] = conv_old[:, 0]          # 当前玩家棋子
            conv_new[:, HISTORY_STEPS + i] = conv_old[:, 1]  # 对手棋子
        conv_new[:, -1] = conv_old[:, 2]            # 常数平面
    new_sd["conv.weight"] = conv_new

    # 其余参数形状一致，直接复制
    for k in new_sd.keys():
        if k == "conv.weight":
            continue
        if k in old_sd:
            new_sd[k] = old_sd[k]

    new_net.load_state_dict(new_sd)

    ts = int(time.time())
    out_path = os.path.join(MODEL_DIR, f"net_{ts}.pt")
    torch.save(new_net.state_dict(), out_path)
    print(f"[✓] New checkpoint saved to {out_path}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    migrate(ckpt)
