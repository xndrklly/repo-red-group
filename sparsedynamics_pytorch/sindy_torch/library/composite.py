"""
Composite library: concatenates outputs from multiple sub-libraries.

This is the main extensibility mechanism. Example usage:

    library = CompositeLibrary([
        PolynomialLibrary(n_vars=3, poly_order=3),
        FourierLibrary(n_vars=3, n_harmonics=5),
        MyCustomNeuralLibrary(n_vars=3),
    ])

Uses nn.ModuleList so that any learnable parameters in sub-libraries are
automatically registered and visible to optimizers.
"""

from typing import List

import torch
from torch import Tensor, nn

from .base import LibraryBase


class CompositeLibrary(LibraryBase):
    """Concatenates outputs from multiple library modules along the feature dimension.

    Parameters
    ----------
    libraries : list of LibraryBase
        Sub-libraries whose outputs are concatenated.
    """

    def __init__(self, libraries: List[LibraryBase]):
        super().__init__()
        self.libraries = nn.ModuleList(libraries)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([lib(x) for lib in self.libraries], dim=-1)

    @property
    def n_features(self) -> int:
        return sum(lib.n_features for lib in self.libraries)

    def get_labels(self, var_names: List[str]) -> List[str]:
        labels = []
        for lib in self.libraries:
            labels.extend(lib.get_labels(var_names))
        return labels
