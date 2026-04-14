"""
Fourier (trigonometric) library for SINDy.

Appends sin(k*x_j) and cos(k*x_j) for each state variable j and
harmonic k = 1..n_harmonics. Matches the MATLAB usesine=1 behavior.
"""

from typing import List

import torch
from torch import Tensor

from .base import LibraryBase


class FourierLibrary(LibraryBase):
    """Sine/cosine library: [sin(x), cos(x), sin(2x), cos(2x), ...].

    Parameters
    ----------
    n_vars : int
        Number of state variables.
    n_harmonics : int
        Number of harmonic frequencies (default 10, matching MATLAB).
    """

    def __init__(self, n_vars: int, n_harmonics: int = 10):
        super().__init__()
        self._n_vars = n_vars
        self._n_harmonics = n_harmonics
        self._n_features = 2 * n_harmonics * n_vars

    def forward(self, x: Tensor) -> Tensor:
        """Build Fourier library matrix.

        Parameters
        ----------
        x : Tensor, shape (..., n_vars)

        Returns
        -------
        Tensor, shape (..., 2 * n_harmonics * n_vars)
        """
        columns = []
        for k in range(1, self._n_harmonics + 1):
            for j in range(self._n_vars):
                columns.append(torch.sin(k * x[..., j : j + 1]))
                columns.append(torch.cos(k * x[..., j : j + 1]))
        return torch.cat(columns, dim=-1)

    @property
    def n_features(self) -> int:
        return self._n_features

    def get_labels(self, var_names: List[str]) -> List[str]:
        labels = []
        for k in range(1, self._n_harmonics + 1):
            for j in range(self._n_vars):
                labels.append(f"sin({k}*{var_names[j]})")
                labels.append(f"cos({k}*{var_names[j]})")
        return labels
