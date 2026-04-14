"""
General gradient-based optimizer for trainable dynamics modules.

Unlike SparseOptimizer, this class is not tied to SINDy coefficient matrices.
It optimizes any nn.Module or parameter iterable and defaults to Adam.
"""

from typing import Callable, Dict, Iterable, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GradientOptimizer:
    """Model-agnostic optimizer for differentiable dynamics.

    Parameters
    ----------
    params_or_module : nn.Module or iterable
        Module or parameters to optimize.
    regularization : callable, optional
        Optional zero-argument function returning a scalar Tensor penalty.
    optimizer_cls : type
        PyTorch optimizer class. Defaults to Adam.
    optimizer_kwargs : dict, optional
        Keyword arguments for the optimizer. Defaults to lr=1e-3.
    """

    def __init__(
        self,
        params_or_module: Union[nn.Module, Iterable[nn.Parameter]],
        regularization: Optional[Callable[[], Tensor]] = None,
        optimizer_cls: Type = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
    ):
        if isinstance(params_or_module, nn.Module):
            params = list(params_or_module.parameters())
        else:
            params = list(params_or_module)
        if not params:
            raise ValueError("No trainable parameters were provided")

        self.params = params
        self.regularization = regularization
        kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.optimizer = optimizer_cls(params, **kwargs)

    def _add_regularization(self, total_loss: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        if self.regularization is None:
            return total_loss, None
        regularization_loss = self.regularization()
        return total_loss + regularization_loss, regularization_loss

    @staticmethod
    def _loss_dict(
        total_loss: Tensor,
        mse_loss: Tensor,
        regularization_loss: Optional[Tensor],
    ) -> Dict[str, float]:
        losses = {
            "total": total_loss.item(),
            "mse": mse_loss.item(),
        }
        if regularization_loss is not None:
            losses["regularization"] = regularization_loss.item()
        return losses

    def step_derivative_matching(
        self,
        dynamics_module: nn.Module,
        x: Tensor,
        dxdt: Tensor,
        t: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """One step minimizing MSE(dynamics_module(t, x), dxdt)."""
        self.optimizer.zero_grad()
        if t is None:
            t = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        pred = dynamics_module(t, x)
        mse_loss = F.mse_loss(pred, dxdt)
        total_loss, regularization_loss = self._add_regularization(mse_loss)
        total_loss.backward()
        self.optimizer.step()
        return self._loss_dict(total_loss, mse_loss, regularization_loss)

    def step_trajectory_matching(
        self,
        ode_model: "ODEModel",
        x0: Tensor,
        t: Tensor,
        x_true: Tensor,
    ) -> Dict[str, float]:
        """One step minimizing MSE(ode_model(x0, t), x_true)."""
        self.optimizer.zero_grad()
        x_pred = ode_model(x0, t)
        mse_loss = F.mse_loss(x_pred, x_true)
        total_loss, regularization_loss = self._add_regularization(mse_loss)
        total_loss.backward()
        self.optimizer.step()
        return self._loss_dict(total_loss, mse_loss, regularization_loss)
