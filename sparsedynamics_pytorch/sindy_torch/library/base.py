"""
Abstract base class for SINDy library builders.

All library modules inherit from LibraryBase, ensuring a consistent interface
for building the candidate function matrix Theta(X). Being nn.Module subclasses,
they participate in autograd, support .to(device), and can contain learnable
parameters for future extensions (e.g., neural network basis functions).
"""

from abc import abstractmethod
from typing import List

import torch
from torch import Tensor, nn


class LibraryBase(nn.Module):
    """Abstract base class for SINDy library builders.

    Subclasses must implement:
        forward(x)      — compute Theta matrix from state data
        n_features       — property returning the number of output columns
        get_labels(...)  — human-readable names for each library column
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Build library matrix from state data.

        Parameters
        ----------
        x : Tensor, shape (..., n_vars)
            State data. Supports arbitrary leading batch dimensions.

        Returns
        -------
        Tensor, shape (..., n_features)
            Library matrix where each column is a candidate function.
        """
        ...

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of output features (columns in Theta)."""
        ...

    @abstractmethod
    def get_labels(self, var_names: List[str]) -> List[str]:
        """Human-readable label for each output column.

        Parameters
        ----------
        var_names : list of str
            Names for each state variable, e.g. ['x', 'y', 'z'].

        Returns
        -------
        list of str
            One label per column, same order as forward() output.
        """
        ...
