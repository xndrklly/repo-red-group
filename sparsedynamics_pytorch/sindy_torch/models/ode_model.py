"""
Differentiable ODE integration model using torchdiffeq.

Wraps any dynamics module with signature forward(t, x) -> dx/dt into a
single nn.Module backed by torchdiffeq. SINDyModule and NeuralODEModule both
use this interface, and future dynamics models can use it too.
"""

from typing import Optional

import torch
from torch import Tensor, nn

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError:
    odeint = None
    odeint_adjoint = None


class ODEModel(nn.Module):
    """End-to-end differentiable ODE model.

    Integrates a dynamics module forward in time to produce predicted
    trajectories.

    Parameters
    ----------
    dynamics_module : nn.Module
        Dynamics module with signature forward(t, x) -> dx/dt.
    method : str
        ODE solver method (default 'dopri5' = RK45).
    rtol, atol : float
        Solver tolerances.
    use_adjoint : bool
        If True, use the adjoint method for O(1)-memory backpropagation.
        Important for large systems where standard backprop through the
        solver is too memory-intensive. This only affects the autograd
        training path. It is distinct from the explicit
        gradient_method="adjoint" option on the optimizer helpers.
    """

    def __init__(
        self,
        dynamics_module: Optional[nn.Module] = None,
        method: str = "dopri5",
        rtol: float = 1e-7,
        atol: float = 1e-9,
        use_adjoint: bool = False,
        *,
        sindy_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        if dynamics_module is None:
            dynamics_module = sindy_module
        elif sindy_module is not None:
            raise ValueError("Pass either dynamics_module or sindy_module, not both")
        if dynamics_module is None:
            raise ValueError("A dynamics module is required")

        self.dynamics_module = dynamics_module
        self.sindy_module = dynamics_module  # Backward-compatible alias.
        self.method = method
        self.rtol = rtol
        self.atol = atol

        if odeint is None:
            raise ImportError(
                "torchdiffeq is required for ODEModel. "
                "Install with: pip install torchdiffeq"
            )

        self._odeint = odeint_adjoint if use_adjoint else odeint

    def forward(
        self,
        x0: Tensor,
        t: Tensor,
        method: Optional[str] = None,
    ) -> Tensor:
        """Integrate the dynamics module from initial conditions.

        Parameters
        ----------
        x0 : Tensor, shape (n_states,) or (batch, n_states)
            Initial condition(s).
        t : Tensor, shape (n_times,)
            Time points at which to return the solution.
        method : str, optional
            Override the default solver method.

        Returns
        -------
        Tensor, shape (n_times, *x0.shape)
            Predicted trajectory. Time is the first dimension,
            matching torchdiffeq convention.
        """
        return self._odeint(
            self.dynamics_module,
            x0,
            t,
            method=method or self.method,
            rtol=self.rtol,
            atol=self.atol,
        )
