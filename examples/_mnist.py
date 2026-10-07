"""예제 공용: MNIST 원본(idx) 파일 읽기 - torch 없이. 파일이 없으면 torchvision이 있을 때만 받음

  from _mnist import mnist
  Xtr, ytr, Xte, yte = mnist(data_dir)        # X: (n, 784) float32 0~1, y: int64
"""
from pathlib import Path

import numpy as np

FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")


def _idx(path):
    raw = Path(path).read_bytes()
    nd = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * nd).reshape(dims)


def mnist(data_dir):
    """data_dir/MNIST/raw/ 의 원본 파일 → (Xtr, ytr, Xte, yte)"""
    raw = Path(data_dir) / "MNIST" / "raw"
    if not all((raw / f).exists() for f in FILES):
        try:
            from torchvision import datasets
        except ImportError:
            raise FileNotFoundError(f"MNIST 파일이 없음: {raw} - torchvision으로 받거나 (pip install torchvision) "
                                    f"원본 파일 {FILES}를 그 폴더에 둘 것") from None
        datasets.MNIST(str(data_dir), train=True, download=True)
        datasets.MNIST(str(data_dir), train=False, download=True)
    Xtr = _idx(raw / FILES[0]).reshape(-1, 784).astype(np.float32) / 255
    Xte = _idx(raw / FILES[2]).reshape(-1, 784).astype(np.float32) / 255
    return Xtr, _idx(raw / FILES[1]).astype(np.int64), Xte, _idx(raw / FILES[3]).astype(np.int64)
