"""
Custom library: wraps user-provided callables as a SINDy library.

Each callable should map Tensor(..., n_vars) -> Tensor(..., 1).
This allows adding arbitrary basis functions without subclassing.

Example:
    lib = CustomLibrary(
        functions=[
            lambda x: (x[..., 0:1]**2 + x[..., 1:2]**2),  # r^2
            lambda x: torch.atan2(x[..., 1:2], x[..., 0:1]),  # theta
        ],
        labels=['r^2', 'theta'],
    )
"""

from typing import Callable, List

import torch
from torch import Tensor

from .base import LibraryBase


class CustomLibrary(LibraryBase):
    """Library from user-provided callable functions.

    Parameters
    ----------
    functions : list of callable
        Each function maps Tensor(..., n_vars) -> Tensor(..., 1).
    labels : list of str
        Human-readable name for each function.
    """

    def __init__(self, functions: List[Callable], labels: List[str]):
        super().__init__()
        if len(functions) != len(labels):
            raise ValueError(
                f"Got {len(functions)} functions but {len(labels)} labels"
            )
        self._functions = functions
        self._labels = labels

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([f(x) for f in self._functions], dim=-1)

    @property
    def n_features(self) -> int:
        return len(self._functions)

    def get_labels(self, var_names: List[str]) -> List[str]:
        return list(self._labels)
