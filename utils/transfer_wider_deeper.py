# utils/transfer_wider_deeper.py
"""
迁移脚本：
  - 从最新模型 checkpoint 生成一个更宽(256)更深(15)的初始模型
  - 保存到 models/net_<timestamp>.pt
用法：
  python utils/transfer_wider_deeper.py
"""
from __future__ import annotations
import glob, os, random, time, math
from pathlib import Path
import torch
import numpy as np

# ----- 项目内 import -----
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))   # 项目根目录
from network.model import AlphaZeroNet, ResidualBlock
from config import MODEL_DIR, BOARD_SIZE, DEVICE

# ----------------- helpers ----------------- #
def latest_ckpt() -> str | None:
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "net_*.pt")))
    return files[-1] if files else None


def copy_overlap_(new_t: torch.Tensor, old_t: torch.Tensor) -> None:
    """把 old_t 复制到 new_t 可对齐的子张量（多出的通道留待外层处理）"""
    slices = tuple(slice(0, min(n, o)) for n, o in zip(new_t.shape, old_t.shape))
    with torch.no_grad():
        new_t[slices].copy_(old_t[slices])


def random_map_indices(new_c: int, old_c: int) -> list[int]:
    """生成 new_c 长度的映射表，前 old_c 为 1-1，剩余随机采样 old_c"""
    idx = list(range(old_c))
    idx += random.choices(range(old_c), k=new_c - old_c)
    return idx


def widen_conv_and_bn(conv_new, bn_new, conv_old, bn_old):
    """
    Net2Wider for Conv + BN：
      - 把旧权重复制到前 old_c
      - 额外通道随机复制旧通道（并按复制次数缩放 BN 权重）
    """
    old_c = conv_old.out_channels
    new_c = conv_new.out_channels
    idx_map = random_map_indices(new_c, old_c)

    with torch.no_grad():
        # conv.weight: (out_c, in_c, k, k) — 逐 out_c 复制
        for new_i, old_i in enumerate(idx_map):
            copy_overlap_(conv_new.weight[new_i:new_i+1], conv_old.weight[old_i:old_i+1])

        # BN 参数 shape=(out_c,)
        for attr in ("weight", "bias", "running_mean", "running_var"):
            new_v = getattr(bn_new, attr)
            old_v = getattr(bn_old, attr)
            copy_overlap_(new_v, old_v)

        # 若有多重复制，要把 BN.weight 按复制频次平均
        dup_count = np.bincount(idx_map, minlength=old_c)
        for new_i, old_i in enumerate(idx_map):
            bn_new.weight.data[new_i] /= max(1, dup_count[old_i])


def init_block_identity(block: ResidualBlock):
    """把新增 ResidualBlock 初始化为恒等映射"""
    with torch.no_grad():
        for m in (block.conv1, block.conv2):
            m.weight.zero_()
        block.bn1.weight.fill_(1.0)
        block.bn1.bias.zero_()
        # He-style恒等：最后 BN γ=0
        block.bn2.weight.zero_()
        block.bn2.bias.zero_()
        block.bn1.running_mean.zero_()
        block.bn1.running_var.fill_(1.0)
        block.bn2.running_mean.zero_()
        block.bn2.running_var.fill_(1.0)


# ----------------- main migrate ----------------- #
def migrate():
    old_path = latest_ckpt()
    if not old_path:
        raise FileNotFoundError("未找到旧模型，请先训练一个 baseline 再迁移。")
    print(f"[+] Loading old checkpoint: {old_path}")

    old_net = AlphaZeroNet().to(DEVICE)                    # 128 × 6
    old_net.load_state_dict(torch.load(old_path, map_location=DEVICE))
    old_net.eval()

    new_channels, new_blocks = 256, 15
    new_net = AlphaZeroNet(
        board_size=BOARD_SIZE,
        channels=new_channels,
        blocks=new_blocks
    ).to(DEVICE)

    sd_old, sd_new = old_net.state_dict(), new_net.state_dict()

    # ---------- 1) Net2Wider ----------
    # 输入层
    widen_conv_and_bn(new_net.conv, new_net.bn, old_net.conv, old_net.bn)

    # 残差块 conv1/conv2 + BN
    for i in range(old_net.res_layers.__len__()):
        b_old: ResidualBlock = old_net.res_layers[i]
        b_new: ResidualBlock = new_net.res_layers[i]
        widen_conv_and_bn(b_new.conv1, b_new.bn1, b_old.conv1, b_old.bn1)
        widen_conv_and_bn(b_new.conv2, b_new.bn2, b_old.conv2, b_old.bn2)

    # policy/value 1×1 conv
    widen_conv_and_bn(new_net.policy_conv, new_net.policy_bn,
                      old_net.policy_conv, old_net.policy_bn)
    widen_conv_and_bn(new_net.value_conv, new_net.value_bn,
                      old_net.value_conv, old_net.value_bn)

    # FC 层尺寸未变，直接复制
    for k in ("policy_fc.weight", "policy_fc.bias",
              "value_fc1.weight", "value_fc1.bias",
              "value_fc2.weight", "value_fc2.bias"):
        copy_overlap_(sd_new[k], sd_old[k])

    # ---------- 2) Net2Deeper ----------
    for idx in range(old_net.res_layers.__len__(), new_blocks):
        init_block_identity(new_net.res_layers[idx])

    # ----------------- save ----------------- #
    ts = int(time.time())
    out_path = os.path.join(MODEL_DIR, f"net_{ts}.pt")
    torch.save(new_net.state_dict(), out_path)
    print(f"[✓] New wider-deeper checkpoint saved to {out_path}")


if __name__ == "__main__":
    torch.manual_seed(2025)
    migrate()
