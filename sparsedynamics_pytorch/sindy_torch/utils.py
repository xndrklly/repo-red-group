"""
Utility functions: device helpers, tensor conversion, display.
"""

from typing import Union

import numpy as np
import torch
from torch import Tensor


def get_device(preferred: str = "auto") -> torch.device:
    """Return the best available device.

    Parameters
    ----------
    preferred : str
        'auto' (default) — CUDA if available, else CPU.
        'cpu' or 'cuda' — force a specific device.
    """
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def to_tensor(
    x: Union[np.ndarray, list, Tensor],
    device: torch.device = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Convert array-like to Tensor with specified device and dtype.

    Parameters
    ----------
    x : array-like or Tensor
        Input data.
    device : torch.device, optional
        Target device (default: CPU).
    dtype : torch.dtype
        Target dtype (default: float64 for ODE precision).
    """
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)
