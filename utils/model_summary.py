# utils/model_summary.py
import torch
from torchsummary import summary          # 或 pip install torchinfo 然后: from torchinfo import summary
from config import BOARD_SIZE, DEVICE
from network.model import AlphaZeroNet

def main():
    net = AlphaZeroNet().to(DEVICE)

    # 取第一层卷积 in_channels 作为输入通道数，保持与模型一致
    in_ch = net.conv.in_channels
    input_shape = (in_ch, BOARD_SIZE, BOARD_SIZE)

    print(f"Using dummy input shape: {input_shape}")
    summary(net, input_size=input_shape, device=str(DEVICE))

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

if __name__ == "__main__":
    main()
