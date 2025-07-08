import torch

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU 可用，设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("GPU 不可用，正在使用 CPU")

if __name__ == "__main__":
    check_gpu()
