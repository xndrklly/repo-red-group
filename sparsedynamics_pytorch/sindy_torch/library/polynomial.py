"""
Polynomial library for SINDy.

Port of pool_data.py — builds monomial features up to a given polynomial order.
The term ordering exactly matches the MATLAB/NumPy implementations via
combinations_with_replacement, ensuring cross-validation compatibility.

Index tuples are precomputed at __init__ time so that forward() is fast
(important when called inside an ODE solver thousands of times).
"""

from itertools import combinations_with_replacement
from typing import List, Tuple

import torch
from torch import Tensor

from .base import LibraryBase


class PolynomialLibrary(LibraryBase):
    """Polynomial library: [1, x, y, x^2, xy, y^2, ...] up to poly_order.

    Parameters
    ----------
    n_vars : int
        Number of state variables.
    poly_order : int
        Maximum polynomial degree to include.
    """

    def __init__(self, n_vars: int, poly_order: int):
        super().__init__()
        self._n_vars = n_vars
        self._poly_order = poly_order

        # Precompute index tuples for all monomial terms
        self._combos: List[Tuple[int, ...]] = []
        for order in range(1, poly_order + 1):
            self._combos.extend(
                combinations_with_replacement(range(n_vars), order)
            )

        self._n_features = 1 + len(self._combos)  # +1 for constant term

    def forward(self, x: Tensor) -> Tensor:
        """Build polynomial library matrix.

        Parameters
        ----------
        x : Tensor, shape (..., n_vars)
            State data with arbitrary leading batch dimensions.

        Returns
        -------
        Tensor, shape (..., n_features)
        """
        # Constant term
        ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        columns = [ones]

        # Monomial terms
        for combo in self._combos:
            col = ones
            for idx in combo:
                col = col * x[..., idx : idx + 1]
            columns.append(col)

        return torch.cat(columns, dim=-1)

    @property
    def n_features(self) -> int:
        return self._n_features

    def get_labels(self, var_names: List[str]) -> List[str]:
        labels = ["1"]
        for combo in self._combos:
            labels.append("".join(var_names[i] for i in combo))
        return labels
