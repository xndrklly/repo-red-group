"""
Utility functions: device helpers, tensor conversion, display.
"""

import argparse
from typing import Union

import numpy as np
import torch
from torch import Tensor


def get_device(preferred: str = "auto") -> torch.device:
    """Return the requested device, defaulting to CUDA when available."""
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    if preferred not in {"cpu", "cuda"}:
        raise ValueError("preferred must be one of: 'auto', 'cpu', 'cuda'")
    return torch.device(preferred)


def add_device_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the standard --device flag used by examples and tests."""
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use. Defaults to CUDA when available, otherwise CPU.",
    )
    return parser


def as_numpy(x: Union[np.ndarray, Tensor]) -> np.ndarray:
    """Convert a tensor on any device to a NumPy array."""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor(
    x: Union[np.ndarray, list, Tensor],
    device: torch.device = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Convert array-like input to a tensor with the requested device and dtype."""
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)
