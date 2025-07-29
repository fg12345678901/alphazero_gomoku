# utils/transfer_history_planes.py
"""
迁移脚本：
  - 将仅含 3 个输入平面的旧模型迁移到当前以 ``HISTORY_STEPS``
    为历史步数的新网络。
  - 第一层卷积仅复制能直接对应的平面：
      * 当前玩家棋面 -> 新网络第 0 层，权重缩放 ``1/HISTORY_STEPS``；
      * 当前对手棋面 -> 新网络第 ``HISTORY_STEPS`` 层，同样缩放；
      * 常数平面保持不变。
    其他新增的历史通道保持随机初始化。
  - 最终模型保存到 ``models/net_<timestamp>.pt``

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
from config import MODEL_DIR, DEVICE


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
    conv_old = old_sd["conv.weight"]  # (out_c, old_planes, k, k)
    conv_new = new_sd["conv.weight"]  # (out_c, new_planes, k, k)

    old_history = (conv_old.size(1) - 1) // 2
    new_history = (conv_new.size(1) - 1) // 2
    copy_steps = min(old_history, new_history)
    scale = 1.0 / new_history

    with torch.no_grad():
        for i in range(copy_steps):
            conv_new[:, i] = conv_old[:, i] * scale
            conv_new[:, new_history + i] = conv_old[:, old_history + i] * scale
        conv_new[:, -1] = conv_old[:, -1]  # constant plane
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
