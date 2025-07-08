# selfplay/augment.py
import numpy as np

def rotate_flip(arr: np.ndarray, k: int, flip: bool):
    """对 2D 或 3D (C,H,W) 做旋转 k*90° 并可选左右翻转"""
    if arr.ndim == 3:
        arr = np.rot90(arr, k, axes=(1,2))
        if flip:
            arr = np.flip(arr, axis=2)
    else:
        arr = np.rot90(arr, k)
        if flip:
            arr = np.flip(arr, axis=1)
    return arr.copy()
