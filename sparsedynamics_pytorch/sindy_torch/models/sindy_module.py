"""
Core SINDy module: dx/dt = Library(x) @ Xi.

Holds the library (nn.Module) and coefficient matrix Xi (nn.Parameter).
Its forward(t, x) signature matches torchdiffeq's func(t, y) convention,
so it plugs directly into odeint().
"""

from typing import List, Optional

import torch
from torch import Tensor, nn

from ..library.base import LibraryBase


class SINDyModule(nn.Module):
    """SINDy dynamical system: dx/dt = Theta(x) @ Xi.

    Parameters
    ----------
    library : LibraryBase
        Library module that builds Theta from state data.
    n_features : int
        Number of library features (columns in Theta). Must match library.n_features.
    n_states : int
        Number of state variables.
    xi_init : Tensor, optional
        Initial coefficient matrix, shape (n_features, n_states).
        If None, initializes to zeros.
    """

    def __init__(
        self,
        library: LibraryBase,
        n_features: int,
        n_states: int,
        xi_init: Optional[Tensor] = None,
    ):
        super().__init__()
        self.library = library
        self._n_states = n_states

        if xi_init is not None:
            self.xi = nn.Parameter(xi_init.clone().detach())
        else:
            self.xi = nn.Parameter(torch.zeros(n_features, n_states, dtype=torch.float64))

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """ODE right-hand side: dx/dt = Theta(x) @ Xi.

        Signature matches torchdiffeq convention: func(t, x) -> dx/dt.

        Parameters
        ----------
        t : Tensor
            Scalar time (unused for autonomous systems, required by odeint).
        x : Tensor, shape (..., n_states)
            Current state.

        Returns
        -------
        Tensor, shape (..., n_states)
            Time derivative.
        """
        theta = self.library(x)  # (..., n_features)
        dx = theta @ self.xi  # (..., n_states)
        return dx

    def predict_derivative(self, x: Tensor) -> Tensor:
        """Convenience: compute dx/dt without the time argument."""
        return self.forward(torch.tensor(0.0), x)

    @torch.no_grad()
    def set_xi(self, xi: Tensor):
        """Set coefficients from an external solver (e.g., STLS result).

        Parameters
        ----------
        xi : Tensor, shape (n_features, n_states)
        """
        self.xi.copy_(xi.to(self.xi.device, self.xi.dtype))

    def active_terms(self, threshold: float = 1e-10) -> Tensor:
        """Boolean mask of non-zero coefficient entries."""
        return self.xi.abs() > threshold

    def sparsity(self) -> float:
        """Fraction of zero coefficients."""
        return (self.xi.abs() < 1e-10).float().mean().item()

    def print_equations(
        self, var_names: List[str], threshold: float = 1e-10
    ) -> str:
        """Pretty-print the discovered equations.

        Parameters
        ----------
        var_names : list of str
            Names for state variables, e.g. ['x', 'y', 'z'].
        threshold : float
            Coefficients below this are hidden.

        Returns
        -------
        str
            Multi-line string of discovered equations.
        """
        labels = self.library.get_labels(var_names)
        xi_np = self.xi.detach().cpu().numpy()
        lines = []

        for col in range(xi_np.shape[1]):
            terms = []
            for row in range(xi_np.shape[0]):
                if abs(xi_np[row, col]) > threshold:
                    coeff = xi_np[row, col]
                    label = labels[row]
                    if label == "1":
                        terms.append(f"{coeff:+.4f}")
                    else:
                        terms.append(f"{coeff:+.4f}*{label}")
            eq_str = " ".join(terms) if terms else "0"
            line = f"  d{var_names[col]}/dt = {eq_str}"
            lines.append(line)
            print(line)

        return "\n".join(lines)
