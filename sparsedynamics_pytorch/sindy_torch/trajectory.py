"""Reusable helpers for trajectory-window training objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TrajectoryWindow:
    x0: Tensor
    t: Tensor
    x_true: Tensor
    start_idx: int
    stop_idx: int
    traj_index: int


def stack_state_trajectory(
    displacements: np.ndarray | Tensor,
    velocities: np.ndarray | Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    u = torch.as_tensor(displacements, device=device, dtype=dtype)
    v = torch.as_tensor(velocities, device=device, dtype=dtype)
    return torch.cat([u, v], dim=-1)


def make_overlapping_trajectory_windows(
    states: np.ndarray | Tensor,
    times: np.ndarray | Tensor,
    *,
    window_length: int,
    stride: int,
    traj_index: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> list[TrajectoryWindow]:
    if window_length < 2:
        raise ValueError("window_length must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")

    x = torch.as_tensor(states, device=device, dtype=dtype)
    t = torch.as_tensor(times, device=device, dtype=dtype)
    if x.ndim != 2:
        raise ValueError(f"states must have shape (n_times, n_states), got {tuple(x.shape)}")
    if t.ndim != 1:
        raise ValueError(f"times must have shape (n_times,), got {tuple(t.shape)}")
    if x.shape[0] != t.shape[0]:
        raise ValueError("states and times must share the same leading dimension")
    if x.shape[0] < window_length:
        raise ValueError(
            f"window_length={window_length} exceeds trajectory length {x.shape[0]}"
        )

    windows = []
    for start in range(0, x.shape[0] - window_length + 1, stride):
        stop = start + window_length
        x_true = x[start:stop]
        t_win = t[start:stop] - t[start]
        windows.append(
            TrajectoryWindow(
                x0=x_true[0],
                t=t_win,
                x_true=x_true,
                start_idx=start,
                stop_idx=stop,
                traj_index=traj_index,
            )
        )
    return windows


def compute_state_scales(
    windows: Iterable[TrajectoryWindow],
    *,
    n_position_states: int,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    series = [window.x_true for window in windows]
    if not series:
        raise ValueError("At least one trajectory window is required to compute scales")
    x_true = torch.cat(series, dim=0)
    u = x_true[..., :n_position_states]
    v = x_true[..., n_position_states:]
    u_scale = torch.clamp(u.std(unbiased=False), min=eps)
    v_scale = torch.clamp(v.std(unbiased=False), min=eps)
    return u_scale, v_scale


def normalized_state_mse(
    x_pred: Tensor,
    x_true: Tensor,
    *,
    n_position_states: int,
    u_scale: Tensor | float,
    v_scale: Tensor | float,
    position_weight: float = 0.5,
    velocity_weight: float = 0.5,
) -> Tensor:
    u_pred = x_pred[..., :n_position_states]
    v_pred = x_pred[..., n_position_states:]
    u_true = x_true[..., :n_position_states]
    v_true = x_true[..., n_position_states:]
    u_loss = F.mse_loss(u_pred / u_scale, u_true / u_scale)
    v_loss = F.mse_loss(v_pred / v_scale, v_true / v_scale)
    return position_weight * u_loss + velocity_weight * v_loss
