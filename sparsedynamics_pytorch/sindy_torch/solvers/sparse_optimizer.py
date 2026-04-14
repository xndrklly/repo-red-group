"""
Gradient-based sparse regression for SINDy coefficients.

Xi is an nn.Parameter optimized via gradient descent with L1 regularization.
Supports two training modes:
  1. Derivative matching:  min ||Theta @ Xi - dX/dt||^2 + lambda * ||Xi||_1
  2. Trajectory matching:  min ||X_pred - X_true||^2 + lambda * ||Xi||_1
     where X_pred is obtained by integrating the discovered ODE via torchdiffeq.
"""

from typing import Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SparseOptimizer:
    """Gradient-based optimizer for SINDy coefficient matrix Xi.

    This is NOT an nn.Module — it is a training-loop helper (like torch.optim).
    The nn.Parameter (Xi) lives in SINDyModule; this class holds a reference.

    Parameters
    ----------
    xi : nn.Parameter
        The coefficient matrix to optimize, shape (n_features, n_states).
    l1_lambda : float
        L1 regularization strength.
    optimizer_cls : type
        PyTorch optimizer class (default: Adam).
    optimizer_kwargs : dict, optional
        Keyword arguments for the optimizer (default: lr=1e-3).
    """

    def __init__(
        self,
        xi: nn.Parameter,
        l1_lambda: float = 0.1,
        optimizer_cls: Type = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
    ):
        self.xi = xi
        self.l1_lambda = l1_lambda
        kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.optimizer = optimizer_cls([xi], **kwargs)

    def step_derivative_matching(
        self, theta: Tensor, dxdt: Tensor
    ) -> Dict[str, float]:
        """One optimization step minimizing ||Theta @ Xi - dxdt||^2 + lambda*||Xi||_1.

        Parameters
        ----------
        theta : Tensor, shape (n_samples, n_features)
        dxdt : Tensor, shape (n_samples, n_states)

        Returns
        -------
        dict with keys 'total', 'mse', 'l1'.
        """
        self.optimizer.zero_grad()
        pred = theta @ self.xi
        mse_loss = F.mse_loss(pred, dxdt)
        l1_loss = self.l1_lambda * self.xi.abs().sum()
        total_loss = mse_loss + l1_loss
        total_loss.backward()
        self.optimizer.step()
        return {
            "total": total_loss.item(),
            "mse": mse_loss.item(),
            "l1": l1_loss.item(),
        }

    def step_trajectory_matching(
        self,
        ode_model: "ODEModel",
        x0: Tensor,
        t: Tensor,
        x_true: Tensor,
    ) -> Dict[str, float]:
        """One optimization step minimizing ||x_pred - x_true||^2 + lambda*||Xi||_1.

        This is the end-to-end differentiable pipeline:
        x0 -> odeint(sindy_rhs, x0, t) -> x_pred -> loss -> backprop through Xi.

        Parameters
        ----------
        ode_model : ODEModel
            Wraps a dynamics module + torchdiffeq.odeint.
        x0 : Tensor, shape (n_states,) or (batch, n_states)
            Initial conditions.
        t : Tensor, shape (n_times,)
            Time points.
        x_true : Tensor, shape (n_times, n_states) or (n_times, batch, n_states)
            True trajectories at the time points.

        Returns
        -------
        dict with keys 'total', 'mse', 'l1'.
        """
        self.optimizer.zero_grad()
        x_pred = ode_model(x0, t)
        mse_loss = F.mse_loss(x_pred, x_true)
        l1_loss = self.l1_lambda * self.xi.abs().sum()
        total_loss = mse_loss + l1_loss
        total_loss.backward()
        self.optimizer.step()
        return {
            "total": total_loss.item(),
            "mse": mse_loss.item(),
            "l1": l1_loss.item(),
        }

    @torch.no_grad()
    def threshold(self, tol: float):
        """Hard-threshold small coefficients to zero (post-training sparsification).

        Parameters
        ----------
        tol : float
            Coefficients with |Xi_ij| < tol are set to zero.
        """
        mask = self.xi.abs() < tol
        self.xi[mask] = 0.0

    @torch.no_grad()
    def proximal_step(self):
        """Apply proximal operator for L1 (soft thresholding) after gradient step.

        For proximal gradient descent (ISTA):
            xi <- sign(xi) * max(|xi| - lambda*lr, 0)
        Call this AFTER optimizer.step() in each iteration.
        """
        lr = self.optimizer.param_groups[0]["lr"]
        tau = self.l1_lambda * lr
        self.xi.copy_(torch.sign(self.xi) * F.relu(self.xi.abs() - tau))
