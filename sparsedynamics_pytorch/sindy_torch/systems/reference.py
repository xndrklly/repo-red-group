"""
Reference dynamical systems for testing SINDy.

All functions follow torchdiffeq convention: f(t, y) -> dy/dt.
All use (..., n_vars) indexing for batch compatibility.
"""

import torch
from torch import Tensor


def lorenz(t: Tensor, y: Tensor, sigma=10.0, beta=8.0 / 3.0, rho=28.0) -> Tensor:
    """Lorenz attractor.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """
    dx = sigma * (y[..., 1] - y[..., 0])
    dy = y[..., 0] * (rho - y[..., 2]) - y[..., 1]
    dz = y[..., 0] * y[..., 1] - beta * y[..., 2]
    return torch.stack([dx, dy, dz], dim=-1)


def hopf(t: Tensor, y: Tensor, mu=1.0, omega=1.0, A=1.0) -> Tensor:
    """Hopf bifurcation normal form.

    dx/dt = mu*x - omega*y - A*x*(x^2 + y^2)
    dy/dt = omega*x + mu*y - A*y*(x^2 + y^2)
    """
    r_sq = y[..., 0] ** 2 + y[..., 1] ** 2
    dx = mu * y[..., 0] - omega * y[..., 1] - A * y[..., 0] * r_sq
    dy = omega * y[..., 0] + mu * y[..., 1] - A * y[..., 1] * r_sq
    return torch.stack([dx, dy], dim=-1)


def logistic(t: Tensor, y: Tensor, r=3.5) -> Tensor:
    """Logistic ODE: dy/dt = r * y * (1 - y)."""
    dy = r * y[..., 0] * (1 - y[..., 0])
    return dy.unsqueeze(-1)
