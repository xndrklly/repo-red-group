"""
Manual trajectory-gradient helpers for explicit sensitivity and adjoint methods.

These utilities are internal to the optimizer implementations. They provide a
parameter-gradient alternative to standard backpropagation through torchdiffeq.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


VALID_TRAJECTORY_GRADIENT_METHODS = {"autograd", "sensitivity", "adjoint"}


def validate_trajectory_gradient_method(gradient_method: str) -> None:
    if gradient_method not in VALID_TRAJECTORY_GRADIENT_METHODS:
        raise ValueError(
            "gradient_method must be one of "
            f"{sorted(VALID_TRAJECTORY_GRADIENT_METHODS)}, got {gradient_method!r}"
        )


def manual_trajectory_loss_and_gradients(
    ode_model: nn.Module,
    x0: Tensor,
    t: Tensor,
    x_true: Tensor,
    params: Sequence[nn.Parameter],
    gradient_method: str,
) -> tuple[Tensor, Tensor]:
    """Return trajectory MSE and flattened parameter gradient."""
    validate_trajectory_gradient_method(gradient_method)
    if gradient_method == "autograd":
        raise ValueError("manual_trajectory_loss_and_gradients does not handle autograd")
    _validate_manual_inputs(x0, t, x_true, gradient_method)
    if odeint is None:
        raise ImportError(
            "torchdiffeq is required for manual trajectory gradients. "
            "Install with: pip install torchdiffeq"
        )

    params = list(params)
    if not params:
        raise ValueError("At least one trainable parameter is required")

    if gradient_method == "sensitivity":
        return _sensitivity_loss_and_gradients(ode_model, x0, t, x_true, params)
    return _adjoint_loss_and_gradients(ode_model, x0, t, x_true, params)


def assign_flat_gradients(
    params: Sequence[nn.Parameter],
    flat_grad: Tensor,
) -> None:
    """Assign a flattened gradient vector onto a parameter list."""
    params = list(params)
    expected = sum(param.numel() for param in params)
    if flat_grad.numel() != expected:
        raise ValueError(
            f"Flat gradient has {flat_grad.numel()} values, expected {expected}"
        )

    offset = 0
    for param in params:
        next_offset = offset + param.numel()
        grad_chunk = flat_grad[offset:next_offset].view_as(param).clone()
        if param.grad is None:
            param.grad = grad_chunk
        else:
            param.grad.copy_(grad_chunk)
        offset = next_offset


def _validate_manual_inputs(
    x0: Tensor,
    t: Tensor,
    x_true: Tensor,
    gradient_method: str,
) -> None:
    if x0.ndim != 1 or x_true.ndim != 2:
        raise ValueError(
            f"gradient_method={gradient_method!r} currently supports only a single "
            "trajectory with x0.shape == (n_states,) and "
            "x_true.shape == (n_times, n_states); use gradient_method='autograd' "
            "for batched inputs."
        )
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if x_true.shape[0] != t.shape[0]:
        raise ValueError(
            f"Expected x_true.shape[0] == len(t), got {x_true.shape[0]} and {t.shape[0]}"
        )
    if x_true.shape[1] != x0.shape[0]:
        raise ValueError(
            f"Expected x_true.shape[1] == x0.shape[0], got {x_true.shape[1]} "
            f"and {x0.shape[0]}"
        )
    if t.shape[0] < 2:
        raise ValueError("Manual trajectory gradients require at least two time points")
    if not torch.all(t[1:] > t[:-1]):
        raise ValueError("t must be strictly increasing")


def _integrate(
    ode_model: nn.Module,
    y0: Tensor,
    t: Tensor,
    dynamics_module: nn.Module | None = None,
) -> Tensor:
    rhs = ode_model.dynamics_module if dynamics_module is None else dynamics_module
    return odeint(
        rhs,
        y0,
        t,
        method=ode_model.method,
        rtol=ode_model.rtol,
        atol=ode_model.atol,
    )


def _trajectory_loss_gradient(x_pred: Tensor, x_true: Tensor) -> Tensor:
    return 2.0 * (x_pred - x_true) / x_pred.numel()


def _flatten_grads(
    grads: Iterable[Tensor | None],
    params: Sequence[nn.Parameter],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    flat_chunks = []
    for grad, param in zip(grads, params):
        if grad is None:
            flat_chunks.append(torch.zeros(param.numel(), dtype=dtype, device=device))
        else:
            flat_chunks.append(grad.detach().reshape(-1).to(device=device, dtype=dtype))
    if not flat_chunks:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.cat(flat_chunks)


def _jacobians(
    dynamics_module: nn.Module,
    t_i: Tensor,
    x: Tensor,
    params: Sequence[nn.Parameter],
) -> tuple[Tensor, Tensor, Tensor]:
    x_req = x.detach().requires_grad_(True)
    f = dynamics_module(t_i, x_req)
    if f.shape != x_req.shape:
        raise ValueError(
            "Manual trajectory gradients currently require dynamics_module(t, x) "
            "to return the same shape as x for a single trajectory."
        )

    rows_x = []
    rows_p = []
    for row_idx in range(f.numel()):
        grads = torch.autograd.grad(
            f[row_idx],
            [x_req] + list(params),
            retain_graph=row_idx < f.numel() - 1,
            allow_unused=True,
        )
        rows_x.append(grads[0].detach())
        rows_p.append(
            _flatten_grads(
                grads[1:],
                params,
                dtype=x.dtype,
                device=x.device,
            )
        )

    j_x = torch.stack(rows_x, dim=0)
    j_p = torch.stack(rows_p, dim=0)
    return f.detach(), j_x, j_p


def _adjoint_vjps(
    dynamics_module: nn.Module,
    t_i: Tensor,
    x: Tensor,
    a: Tensor,
    params: Sequence[nn.Parameter],
) -> tuple[Tensor, Tensor, Tensor]:
    x_req = x.detach().requires_grad_(True)
    f = dynamics_module(t_i, x_req)
    if f.shape != x_req.shape:
        raise ValueError(
            "Manual trajectory gradients currently require dynamics_module(t, x) "
            "to return the same shape as x for a single trajectory."
        )

    grads = torch.autograd.grad(
        f,
        [x_req] + list(params),
        grad_outputs=a,
        allow_unused=True,
    )
    vjp_x = grads[0].detach()
    vjp_p = _flatten_grads(
        grads[1:],
        params,
        dtype=x.dtype,
        device=x.device,
    )
    return f.detach(), vjp_x, vjp_p


def _sensitivity_loss_and_gradients(
    ode_model: nn.Module,
    x0: Tensor,
    t: Tensor,
    x_true: Tensor,
    params: Sequence[nn.Parameter],
) -> tuple[Tensor, Tensor]:
    n_states = x0.numel()
    n_params = sum(param.numel() for param in params)
    z0 = torch.cat(
        [
            x0,
            torch.zeros(
                n_states * n_params,
                dtype=x0.dtype,
                device=x0.device,
            ),
        ]
    )

    def sensitivity_rhs(t_i: Tensor, z: Tensor) -> Tensor:
        x = z[:n_states]
        s_theta = z[n_states:].view(n_states, n_params)
        f, j_x, j_p = _jacobians(ode_model.dynamics_module, t_i, x, params)
        ds = j_x @ s_theta + j_p
        return torch.cat([f, ds.reshape(-1)])

    z_traj = _integrate(ode_model, z0, t, dynamics_module=sensitivity_rhs)
    x_pred = z_traj[:, :n_states]
    s_traj = z_traj[:, n_states:].view(t.shape[0], n_states, n_params)
    mse_loss = F.mse_loss(x_pred, x_true)
    dloss_dx = _trajectory_loss_gradient(x_pred, x_true)
    flat_grad = torch.einsum("tn,tnp->p", dloss_dx, s_traj)
    return mse_loss, flat_grad.detach()


def _adjoint_loss_and_gradients(
    ode_model: nn.Module,
    x0: Tensor,
    t: Tensor,
    x_true: Tensor,
    params: Sequence[nn.Parameter],
) -> tuple[Tensor, Tensor]:
    x_pred = _integrate(ode_model, x0, t).detach()
    mse_loss = F.mse_loss(x_pred, x_true)
    dloss_dx = _trajectory_loss_gradient(x_pred, x_true)

    n_states = x0.numel()
    n_params = sum(param.numel() for param in params)
    grad_theta = torch.zeros(n_params, dtype=x0.dtype, device=x0.device)
    adjoint = dloss_dx[-1].detach().clone()

    for idx in range(t.shape[0] - 1, 0, -1):
        z0 = torch.cat([x_pred[idx], adjoint, grad_theta])

        def adjoint_rhs(t_i: Tensor, z: Tensor) -> Tensor:
            x = z[:n_states]
            a = z[n_states : 2 * n_states]
            f, vjp_x, vjp_p = _adjoint_vjps(
                ode_model.dynamics_module,
                t_i,
                x,
                a,
                params,
            )
            return torch.cat([f, -vjp_x, -vjp_p])

        t_segment = torch.stack([t[idx], t[idx - 1]])
        z_prev = _integrate(
            ode_model,
            z0,
            t_segment,
            dynamics_module=adjoint_rhs,
        )[-1].detach()

        adjoint = z_prev[n_states : 2 * n_states] + dloss_dx[idx - 1]
        grad_theta = z_prev[2 * n_states :]

    return mse_loss, grad_theta.detach()
