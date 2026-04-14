"""
Autograd-based exact derivative computation.

When the data comes from a known ODE, this computes exact derivatives by
evaluating the ODE right-hand side — no finite differences needed.

For measured data without a known ODE, a future extension is to fit a
neural network surrogate and differentiate through it.
"""

import torch
from torch import Tensor

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None

from typing import Callable, Tuple


def autograd_derivative(
    ode_func: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    t: Tensor,
    method: str = "dopri5",
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> Tuple[Tensor, Tensor]:
    """Generate (x, dx/dt) pairs by integrating a known ODE and evaluating its RHS.

    Parameters
    ----------
    ode_func : callable
        f(t, x) -> dx/dt, the true dynamics.
    x0 : Tensor, shape (n_states,)
        Initial condition.
    t : Tensor, shape (n_times,)
        Time points.
    method : str
        ODE solver method.
    rtol, atol : float
        Solver tolerances.

    Returns
    -------
    x : Tensor, shape (n_times, n_states)
        State trajectory.
    dxdt : Tensor, shape (n_times, n_states)
        Exact derivatives at each time point.
    """
    if odeint is None:
        raise ImportError("torchdiffeq required. Install with: pip install torchdiffeq")

    with torch.no_grad():
        x = odeint(ode_func, x0, t, method=method, rtol=rtol, atol=atol)

    # Evaluate RHS at each point for exact derivatives
    dxdt = torch.stack([ode_func(t[i], x[i]) for i in range(len(t))])

    return x, dxdt
