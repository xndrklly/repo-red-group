"""
Generate two spring-grid regression reports with a focused model set:
  - STLS dense (lam=0.05)
  - SINDy Adam local (no L1) using residual loss
  - SINDy Adam local trajectory v2 (no L1) by default

The trajectory-v2 path keeps the legacy full-horizon trajectory loss available
for comparison, but the default report uses a windowed, warm-started, symmetric
stiffness formulation that is better conditioned for K recovery.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sindy_torch
import spring_grid_K_regression as base
from sindy_torch.systems.spring_grid import newmark_beta_simulation_torch, zero_force_torch


STLS_NAME = "STLS dense (lam=0.05)"
ADAM_RESIDUAL_NAME = "SINDy Adam local (no L1)"
ADAM_TRAJECTORY_V2_NAME = "SINDy Adam local trajectory v2 (no L1)"
ADAM_TRAJECTORY_LEGACY_NAME = "SINDy Adam local trajectory (legacy, no L1)"


def format_model_summary(names: tuple[str, ...] | list[str]) -> str:
    code_names = [f"<code>{name}</code>" for name in names]
    if len(code_names) == 1:
        return code_names[0]
    if len(code_names) == 2:
        return f"{code_names[0]} and {code_names[1]}"
    return ", ".join(code_names[:-1]) + f", and {code_names[-1]}"


def build_physically_motivated_stiffness_matrix(n: int) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    center = (n - 1) / 2.0
    sigma = max(1.0, 0.22 * n)

    def add_pair(a: int, b: int, k: float) -> None:
        rows.extend([a, b, a, b])
        cols.extend([a, b, b, a])
        vals.extend([k, k, -k, -k])

    def support_factor(row_mid: float) -> float:
        return 1.0 + 0.65 * np.exp(-row_mid / max(1.0, 0.28 * n))

    def center_factor(col_mid: float) -> float:
        return 1.0 + 0.18 * np.exp(-((col_mid - center) ** 2) / (2.0 * sigma * sigma))

    for i in range(n):
        for j in range(n - 1):
            a = base.node_index(i, j, n)
            b = base.node_index(i, j + 1, n)
            row_mid = float(i)
            col_mid = j + 0.5
            k = 0.95 * support_factor(row_mid) * center_factor(col_mid)
            add_pair(a, b, float(k))

    for i in range(n - 1):
        for j in range(n):
            a = base.node_index(i, j, n)
            b = base.node_index(i + 1, j, n)
            row_mid = i + 0.5
            col_mid = float(j)
            k = 1.20 * support_factor(row_mid) * center_factor(col_mid)
            add_pair(a, b, float(k))

    for i in range(n - 1):
        for j in range(n - 1):
            a = base.node_index(i, j, n)
            b = base.node_index(i + 1, j + 1, n)
            row_mid = i + 0.5
            col_mid = j + 0.5
            k = 0.42 * support_factor(row_mid) * (0.9 + 0.1 * center_factor(col_mid))
            add_pair(a, b, float(k))

    N = n * n
    K = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    K.sum_duplicates()
    return K


def select_trajectory_model_names(trajectory_variant: str) -> tuple[str, ...]:
    if trajectory_variant == "legacy":
        return (STLS_NAME, ADAM_RESIDUAL_NAME, ADAM_TRAJECTORY_LEGACY_NAME)
    if trajectory_variant == "both":
        return (
            STLS_NAME,
            ADAM_RESIDUAL_NAME,
            ADAM_TRAJECTORY_V2_NAME,
            ADAM_TRAJECTORY_LEGACY_NAME,
        )
    return (STLS_NAME, ADAM_RESIDUAL_NAME, ADAM_TRAJECTORY_V2_NAME)


def build_objective_note(
    *,
    trajectory_variant: str,
    legacy_stride: int,
    v2_window: int,
    v2_window_stride: int,
) -> str:
    notes = [
        "<p><strong>Optimization note:</strong> "
        "The residual Adam local model minimizes acceleration residual MSE on pooled samples.</p>"
    ]
    if trajectory_variant in {"v2", "both"}:
        notes.append(
            "<p><strong>Trajectory v2:</strong> "
            "The improved trajectory model is warm-started from the residual Adam local estimate, "
            "fits a symmetric local stiffness matrix with windowed Newmark rollouts, and minimizes "
            "a normalized 50/50 displacement/velocity trajectory loss over overlapping windows "
            f"(window length {v2_window}, stride {v2_window_stride}).</p>"
        )
    if trajectory_variant in {"legacy", "both"}:
        notes.append(
            "<p><strong>Legacy trajectory model:</strong> "
            "The legacy path minimizes full state-trajectory MSE on stacked [u, v] trajectories, "
            "using a 3-train / 1-test trajectory split and every "
            f"{legacy_stride}th sample from the simulated time grid.</p>"
        )
    return "".join(notes)


def split_trajectory_indices(n_trajectories: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trajectories)
    n_train = max(1, n_trajectories - 1)
    train_perm = perm[:n_train]
    test_perm = perm[n_train:]
    if test_perm.size == 0:
        test_perm = train_perm[-1:]
    return train_perm, test_perm


class SpringGridTrajectoryModule(nn.Module):
    def __init__(self, xi: nn.Parameter, n_free: int):
        super().__init__()
        self.xi = xi
        self.n_free = n_free

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        u = state[..., : self.n_free]
        v = state[..., self.n_free :]
        a = u @ self.xi
        return torch.cat([v, a], dim=-1)


class SymmetricLocalStiffnessModel(nn.Module):
    def __init__(self, K_init: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.raw_K = nn.Parameter(K_init.clone())
        self.register_buffer("mask", mask.to(device=K_init.device, dtype=K_init.dtype))

    def stiffness_matrix(self) -> torch.Tensor:
        masked = self.raw_K * self.mask
        K = 0.5 * (masked + masked.T)
        return K * self.mask

    def xi_matrix(self) -> torch.Tensor:
        return -self.stiffness_matrix().T


def stack_legacy_trajectory_batch(
    trajectories: list[dict],
    indices: np.ndarray,
    *,
    stride: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = []
    for idx in indices:
        traj = trajectories[int(idx)]
        state = np.concatenate(
            [traj["U_free"][::stride], traj["V_free"][::stride]],
            axis=1,
        )
        states.append(state)
    times = trajectories[int(indices[0])]["times"][::stride]
    x_true = torch.as_tensor(np.stack(states, axis=1), device=device, dtype=dtype)
    x0 = x_true[0]
    t = torch.as_tensor(times, device=device, dtype=dtype)
    return x0, t, x_true


def legacy_trajectory_loss(
    ode_model: sindy_torch.ODEModel,
    x0: torch.Tensor,
    t: torch.Tensor,
    x_true: torch.Tensor,
) -> float:
    with torch.no_grad():
        x_pred = ode_model(x0, t)
        return F.mse_loss(x_pred, x_true).item()


def train_trajectory_legacy_method(
    trajectories: list[dict],
    K_true: torch.Tensor,
    *,
    n_states: int,
    n_epochs: int,
    lr: float,
    seed: int,
    xi_mask: torch.Tensor | None,
    history_specs: list[dict] | None,
    history_every: int,
    free: np.ndarray,
    n_total_dofs: int,
    sample_stride: int,
    n_eig: int = 5,
) -> dict:
    dtype = K_true.dtype
    device = K_true.device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_perm, test_perm = split_trajectory_indices(len(trajectories), seed)
    x0_train, t_train, x_true_train = stack_legacy_trajectory_batch(
        trajectories, train_perm, stride=sample_stride, device=device, dtype=dtype
    )
    x0_test, t_test, x_true_test = stack_legacy_trajectory_batch(
        trajectories, test_perm, stride=sample_stride, device=device, dtype=dtype
    )

    xi = nn.Parameter(torch.zeros(n_states, n_states, dtype=dtype, device=device))
    if xi_mask is not None:
        mask_t = xi_mask.to(device=device, dtype=dtype)
        xi.register_hook(lambda grad: grad * mask_t)
    else:
        mask_t = None

    dynamics = SpringGridTrajectoryModule(xi, n_states).to(device=device, dtype=dtype)
    ode_model = sindy_torch.ODEModel(
        dynamics_module=dynamics,
        method="rk4",
        rtol=1e-6,
        atol=1e-8,
    )
    optimizer = sindy_torch.SparseOptimizer(
        xi,
        l1_lambda=0.0,
        optimizer_kwargs={"lr": lr},
    )

    train_losses, test_losses = [], []
    snap_epochs, snap_evals, snap_K_err = [], [], []
    true_evals = base.k_eigenvalues(K_true, n_eig)
    history_epochs = []
    param_history = {spec["series_key"]: [] for spec in history_specs or []}

    for epoch in range(n_epochs):
        info = optimizer.step_trajectory_matching(
            ode_model,
            x0_train,
            t_train,
            x_true_train,
            gradient_method="autograd",
        )
        if mask_t is not None:
            with torch.no_grad():
                xi.mul_(mask_t)

        train_losses.append(info["mse"])
        test_losses.append(legacy_trajectory_loss(ode_model, x0_test, t_test, x_true_test))

        if history_specs and (epoch % history_every == 0 or epoch == n_epochs - 1):
            K_pred_full = base.build_full_K_from_free(base.xi_to_K(xi), free, n_total_dofs)
            history_epochs.append(epoch)
            for spec in history_specs:
                k_val = float(-K_pred_full[spec["node"], spec["neighbor"]])
                param_history[spec["series_key"]].append(k_val)

        if epoch % max(1, n_epochs // 60) == 0 or epoch == n_epochs - 1:
            K_pred = base.xi_to_K(xi)
            snap_epochs.append(epoch)
            snap_evals.append(base.k_eigenvalues(K_pred, n_eig))
            snap_K_err.append(base.relative_K_error(K_pred, K_true))

    result = {
        "xi": xi.detach().clone(),
        "train_losses": np.asarray(train_losses),
        "test_losses": np.asarray(test_losses),
        "snap_epochs": np.asarray(snap_epochs),
        "snap_evals": np.asarray(snap_evals),
        "snap_K_err": np.asarray(snap_K_err),
        "true_evals": true_evals,
        "trajectory_train_count": int(train_perm.size),
        "trajectory_test_count": int(test_perm.size),
        "trajectory_sample_stride": int(sample_stride),
    }
    if history_specs:
        result["parameter_history_epochs"] = np.asarray(history_epochs)
        result["parameter_history"] = {
            key: np.asarray(vals) for key, vals in param_history.items()
        }
    return result


def build_windowed_trajectory_set(
    trajectories: list[dict],
    indices: np.ndarray,
    *,
    window_length: int,
    stride: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[sindy_torch.TrajectoryWindow]:
    windows: list[sindy_torch.TrajectoryWindow] = []
    for idx in indices:
        traj = trajectories[int(idx)]
        states = sindy_torch.stack_state_trajectory(
            traj["U_free"],
            traj["V_free"],
            device=device,
            dtype=dtype,
        )
        windows.extend(
            sindy_torch.make_overlapping_trajectory_windows(
                states,
                traj["times"],
                window_length=window_length,
                stride=stride,
                traj_index=int(idx),
                device=device,
                dtype=dtype,
            )
        )
    return windows


def rollout_window_with_newmark(
    K_free: torch.Tensor,
    window: sindy_torch.TrajectoryWindow,
    *,
    free: np.ndarray,
    n: int,
    n_total_dofs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    free_t = torch.as_tensor(free, device=device, dtype=torch.long)
    n_free = free_t.numel()
    u0_full = torch.zeros(n_total_dofs, device=device, dtype=dtype)
    v0_full = torch.zeros(n_total_dofs, device=device, dtype=dtype)
    u0_full[free_t] = window.x0[:n_free]
    v0_full[free_t] = window.x0[n_free:]

    dt = float((window.t[1] - window.t[0]).item())
    t_end = dt * (window.t.numel() - 1)
    K_full = base.build_full_K_from_free_torch(K_free, free, n_total_dofs)
    result = newmark_beta_simulation_torch(
        K_full,
        torch.eye(n_total_dofs, device=device, dtype=dtype),
        torch.zeros((n_total_dofs, n_total_dofs), device=device, dtype=dtype),
        zero_force_torch(n_total_dofs, device=device, dtype=dtype),
        free,
        n,
        t_end=t_end,
        dt=dt,
        u0=u0_full,
        v0=v0_full,
        device=device,
        dtype=dtype,
    )
    return torch.cat(
        [result.displacements[:, free_t], result.velocities[:, free_t]],
        dim=-1,
    )


def evaluate_v2_windows(
    model: SymmetricLocalStiffnessModel,
    windows: list[sindy_torch.TrajectoryWindow],
    *,
    free: np.ndarray,
    n: int,
    n_total_dofs: int,
    n_states: int,
    u_scale: torch.Tensor,
    v_scale: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    if not windows:
        return 0.0
    losses = []
    with torch.no_grad():
        K_free = model.stiffness_matrix()
        for window in windows:
            x_pred = rollout_window_with_newmark(
                K_free,
                window,
                free=free,
                n=n,
                n_total_dofs=n_total_dofs,
                device=device,
                dtype=dtype,
            )
            losses.append(
                sindy_torch.normalized_state_mse(
                    x_pred,
                    window.x_true,
                    n_position_states=n_states,
                    u_scale=u_scale,
                    v_scale=v_scale,
                ).item()
            )
    return float(np.mean(losses))


def train_trajectory_v2_method(
    trajectories: list[dict],
    K_true: torch.Tensor,
    *,
    warm_start_xi: torch.Tensor,
    n: int,
    n_states: int,
    n_epochs: int,
    lr: float,
    seed: int,
    xi_mask: torch.Tensor,
    history_specs: list[dict] | None,
    history_every: int,
    free: np.ndarray,
    n_total_dofs: int,
    window_length: int,
    window_stride: int,
    patience: int,
    n_eig: int = 5,
) -> dict:
    dtype = K_true.dtype
    device = K_true.device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_perm, test_perm = split_trajectory_indices(len(trajectories), seed)
    train_windows = build_windowed_trajectory_set(
        trajectories,
        train_perm,
        window_length=window_length,
        stride=window_stride,
        device=device,
        dtype=dtype,
    )
    test_windows = build_windowed_trajectory_set(
        trajectories,
        test_perm,
        window_length=window_length,
        stride=window_stride,
        device=device,
        dtype=dtype,
    )
    u_scale, v_scale = sindy_torch.compute_state_scales(
        train_windows,
        n_position_states=n_states,
    )

    K_init = base.xi_to_K(warm_start_xi).to(device=device, dtype=dtype)
    mask_t = xi_mask.to(device=device, dtype=dtype)
    model = SymmetricLocalStiffnessModel(K_init, mask_t)
    optimizer = torch.optim.Adam([model.raw_K], lr=lr)

    warm_start_train_loss = evaluate_v2_windows(
        model,
        train_windows,
        free=free,
        n=n,
        n_total_dofs=n_total_dofs,
        n_states=n_states,
        u_scale=u_scale,
        v_scale=v_scale,
        device=device,
        dtype=dtype,
    )
    warm_start_test_loss = evaluate_v2_windows(
        model,
        test_windows,
        free=free,
        n=n,
        n_total_dofs=n_total_dofs,
        n_states=n_states,
        u_scale=u_scale,
        v_scale=v_scale,
        device=device,
        dtype=dtype,
    )

    train_losses, test_losses = [], []
    snap_epochs, snap_evals, snap_K_err = [], [], []
    true_evals = base.k_eigenvalues(K_true, n_eig)
    history_epochs = []
    param_history = {spec["series_key"]: [] for spec in history_specs or []}

    best_state = model.raw_K.detach().clone()
    best_test_loss = warm_start_test_loss
    best_epoch = -1
    wait = 0
    restored_best = False
    stopped_early = False

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        for window in train_windows:
            K_free = model.stiffness_matrix()
            x_pred = rollout_window_with_newmark(
                K_free,
                window,
                free=free,
                n=n,
                n_total_dofs=n_total_dofs,
                device=device,
                dtype=dtype,
            )
            loss = sindy_torch.normalized_state_mse(
                x_pred,
                window.x_true,
                n_position_states=n_states,
                u_scale=u_scale,
                v_scale=v_scale,
            )
            (loss / len(train_windows)).backward()
        optimizer.step()
        with torch.no_grad():
            model.raw_K.mul_(mask_t)

        train_loss = evaluate_v2_windows(
            model,
            train_windows,
            free=free,
            n=n,
            n_total_dofs=n_total_dofs,
            n_states=n_states,
            u_scale=u_scale,
            v_scale=v_scale,
            device=device,
            dtype=dtype,
        )
        test_loss = evaluate_v2_windows(
            model,
            test_windows,
            free=free,
            n=n,
            n_total_dofs=n_total_dofs,
            n_states=n_states,
            u_scale=u_scale,
            v_scale=v_scale,
            device=device,
            dtype=dtype,
        )
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if history_specs and (epoch % history_every == 0 or epoch == n_epochs - 1):
            K_pred_full = base.build_full_K_from_free(
                model.stiffness_matrix(),
                free,
                n_total_dofs,
            )
            history_epochs.append(epoch)
            for spec in history_specs:
                k_val = float(-K_pred_full[spec["node"], spec["neighbor"]])
                param_history[spec["series_key"]].append(k_val)

        if epoch % max(1, n_epochs // 60) == 0 or epoch == n_epochs - 1:
            K_pred = model.stiffness_matrix().detach()
            snap_epochs.append(epoch)
            snap_evals.append(base.k_eigenvalues(K_pred, n_eig))
            snap_K_err.append(base.relative_K_error(K_pred, K_true))

        if test_loss < best_test_loss - 1e-12:
            best_test_loss = test_loss
            best_state = model.raw_K.detach().clone()
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped_early = True
                break

    with torch.no_grad():
        if not torch.allclose(model.raw_K, best_state):
            model.raw_K.copy_(best_state)
            restored_best = True
        model.raw_K.mul_(mask_t)

    final_train_loss = evaluate_v2_windows(
        model,
        train_windows,
        free=free,
        n=n,
        n_total_dofs=n_total_dofs,
        n_states=n_states,
        u_scale=u_scale,
        v_scale=v_scale,
        device=device,
        dtype=dtype,
    )
    final_test_loss = evaluate_v2_windows(
        model,
        test_windows,
        free=free,
        n=n,
        n_total_dofs=n_total_dofs,
        n_states=n_states,
        u_scale=u_scale,
        v_scale=v_scale,
        device=device,
        dtype=dtype,
    )
    if train_losses:
        train_losses[-1] = final_train_loss
        test_losses[-1] = final_test_loss

    K_final = model.stiffness_matrix().detach()
    final_k_err = base.relative_K_error(K_final, K_true)
    final_evals = base.k_eigenvalues(K_final, n_eig)
    selected_epoch = best_epoch if best_epoch >= 0 else 0
    if snap_epochs:
        snap_epochs[-1] = selected_epoch
        snap_evals[-1] = final_evals
        snap_K_err[-1] = final_k_err
    else:
        snap_epochs.append(selected_epoch)
        snap_evals.append(final_evals)
        snap_K_err.append(final_k_err)

    result = {
        "xi": -K_final.T,
        "train_losses": np.asarray(train_losses, dtype=float),
        "test_losses": np.asarray(test_losses, dtype=float),
        "snap_epochs": np.asarray(snap_epochs),
        "snap_evals": np.asarray(snap_evals),
        "snap_K_err": np.asarray(snap_K_err),
        "true_evals": true_evals,
        "trajectory_window_length": int(window_length),
        "trajectory_window_stride": int(window_stride),
        "trajectory_train_window_count": int(len(train_windows)),
        "trajectory_test_window_count": int(len(test_windows)),
        "warm_start_train_loss": float(warm_start_train_loss),
        "warm_start_test_loss": float(warm_start_test_loss),
        "best_test_loss": float(best_test_loss),
        "best_epoch": int(best_epoch),
        "restored_best": bool(restored_best),
        "stopped_early": bool(stopped_early),
        "u_scale": float(u_scale.item()),
        "v_scale": float(v_scale.item()),
        "K_symmetric_max_abs_diff": float(torch.max(torch.abs(K_final - K_final.T)).item()),
    }
    if history_specs:
        K_pred_full = base.build_full_K_from_free(K_final, free, n_total_dofs)
        if history_epochs:
            history_epochs[-1] = selected_epoch
            for spec in history_specs:
                k_val = float(-K_pred_full[spec["node"], spec["neighbor"]])
                param_history[spec["series_key"]][-1] = k_val
        else:
            history_epochs.append(selected_epoch)
            for spec in history_specs:
                k_val = float(-K_pred_full[spec["node"], spec["neighbor"]])
                param_history[spec["series_key"]].append(k_val)
        result["parameter_history_epochs"] = np.asarray(history_epochs)
        result["parameter_history"] = {
            key: np.asarray(vals) for key, vals in param_history.items()
        }
    return result


def run_trajectory_variants(
    *,
    trajectory_variant: str,
    data: dict,
    grad_problem: dict,
    residual_result: dict,
    incident_specs: list[dict],
    trajectory_epochs: int,
    legacy_stride: int,
    v2_window: int,
    v2_window_stride: int,
    v2_patience: int,
) -> OrderedDict[str, dict]:
    grad_results: OrderedDict[str, dict] = OrderedDict()

    grad_results[ADAM_RESIDUAL_NAME] = residual_result
    if trajectory_variant in {"v2", "both"}:
        grad_results[ADAM_TRAJECTORY_V2_NAME] = train_trajectory_v2_method(
            data["trajectories"],
            grad_problem["K_true"],
            warm_start_xi=residual_result["xi"],
            n=data["n"],
            n_states=grad_problem["n_states"],
            n_epochs=trajectory_epochs,
            lr=1e-3,
            seed=0,
            xi_mask=grad_problem["xi_mask"],
            history_specs=incident_specs,
            history_every=1,
            free=data["free"],
            n_total_dofs=data["N"],
            window_length=v2_window,
            window_stride=v2_window_stride,
            patience=v2_patience,
            n_eig=5,
        )

    if trajectory_variant in {"legacy", "both"}:
        grad_results[ADAM_TRAJECTORY_LEGACY_NAME] = train_trajectory_legacy_method(
            data["trajectories"],
            grad_problem["K_true"],
            n_states=grad_problem["n_states"],
            n_epochs=trajectory_epochs,
            lr=1e-2,
            seed=0,
            xi_mask=grad_problem["xi_mask"],
            history_specs=incident_specs,
            history_every=1,
            free=data["free"],
            n_total_dofs=data["N"],
            sample_stride=legacy_stride,
            n_eig=5,
        )
    return grad_results


def print_method_metrics(name: str, result: dict) -> None:
    print(
        f"  {name}: train={result['train_losses'][-1]:.3e} "
        f"test={result['test_losses'][-1]:.3e} "
        f"K_err={result['snap_K_err'][-1]:.3e}"
    )


def print_v2_details(result: dict) -> None:
    best_epoch_label = "warm-start" if result["best_epoch"] < 0 else str(result["best_epoch"])
    print(
        "    warm-start metrics: "
        f"train={result['warm_start_train_loss']:.3e} "
        f"val={result['warm_start_test_loss']:.3e}"
    )
    print(
        "    v2 selection: "
        f"best_val={result['best_test_loss']:.3e} "
        f"best_epoch={best_epoch_label}"
    )
    print(
        "    v2 final: "
        f"train={result['train_losses'][-1]:.3e} "
        f"val={result['test_losses'][-1]:.3e} "
        f"K_err={result['snap_K_err'][-1]:.3e}"
    )
    print(
        "    checkpoint restore: "
        f"restored={result['restored_best']} "
        f"stopped_early={result['stopped_early']}"
    )


def generate_report(
    *,
    report_title: str,
    spring_description: str,
    out_path: Path,
    stiffness_matrix: sp.csr_matrix | None,
    dataset_seed: int,
    residual_epochs: int,
    trajectory_epochs: int,
    trajectory_stride: int,
    trajectory_variant: str,
    trajectory_window: int,
    trajectory_window_stride: int,
    trajectory_patience: int,
    execution_mode: str,
    device_arg: str,
    simulation_backend: str,
    n: int = 10,
    n_traj: int = 4,
    t_end: float = 20.0,
    dt: float = 0.01,
) -> Path:
    dtype = torch.float64
    plan = base.resolve_execution_plan(device_arg, execution_mode)
    dataset_device = plan["dataset_device"]
    stls_device = plan["stls_device"]
    training_device = plan["training_device"]
    rollout_device = plan["rollout_device"]
    dataset_backend = base.resolve_simulation_backend(simulation_backend, dataset_device)
    rollout_backend = base.resolve_simulation_backend(simulation_backend, rollout_device)
    focus_model_names = select_trajectory_model_names(trajectory_variant)

    print(f"\n=== {report_title} ===")
    print(
        "Stage devices: "
        f"dataset={dataset_device}, stls={stls_device}, training={training_device}, rollout={rollout_device}"
    )

    data = base.generate_dataset(
        n=n,
        n_traj=n_traj,
        t_end=t_end,
        dt=dt,
        seed=dataset_seed,
        device=dataset_device,
        simulation_backend=dataset_backend,
        dtype=dtype,
        stiffness_matrix=stiffness_matrix,
    )
    print(f"  free DOFs = {data['free'].size}, total samples = {data['U'].shape[0]}")

    stls_problem = base.build_regression_problem(data, device=stls_device, dtype=dtype)
    grad_problem = base.build_regression_problem(data, device=training_device, dtype=dtype)
    selected_nodes = base.select_representative_nodes(
        data["n"], data["free"], data["trajectories"][0]["U_full"]
    )
    incident_specs = base.build_incident_specs(selected_nodes, data["n"])

    stls_results = OrderedDict()
    stls_results[STLS_NAME] = base.run_stls(
        stls_problem["theta_train"],
        stls_problem["target_train"],
        stls_problem["theta_test"],
        stls_problem["target_test"],
        stls_problem["K_true"],
        lam=0.05,
        mask=None,
    )
    print(
        f"  {STLS_NAME}: train={stls_results[STLS_NAME]['train_mse']:.3e} "
        f"test={stls_results[STLS_NAME]['test_mse']:.3e} "
        f"K_err={stls_results[STLS_NAME]['K_err']:.3e}"
    )

    residual_result = base.train_gradient_method(
        grad_problem["theta_train"],
        grad_problem["target_train"],
        grad_problem["theta_test"],
        grad_problem["target_test"],
        grad_problem["K_true"],
        n_features=grad_problem["n_features"],
        n_states=grad_problem["n_states"],
        n_epochs=residual_epochs,
        lr=5e-2,
        l1_lambda=0.0,
        proximal=False,
        snapshot_every=max(1, residual_epochs // 60),
        n_eig=5,
        seed=0,
        xi_mask=grad_problem["xi_mask"],
        history_specs=incident_specs,
        history_every=1,
        free=data["free"],
        n_total_dofs=data["N"],
    )
    print_method_metrics(ADAM_RESIDUAL_NAME, residual_result)

    grad_results = run_trajectory_variants(
        trajectory_variant=trajectory_variant,
        data=data,
        grad_problem=grad_problem,
        residual_result=residual_result,
        incident_specs=incident_specs,
        trajectory_epochs=trajectory_epochs,
        legacy_stride=trajectory_stride,
        v2_window=trajectory_window,
        v2_window_stride=trajectory_window_stride,
        v2_patience=trajectory_patience,
    )
    if ADAM_TRAJECTORY_V2_NAME in grad_results:
        print_v2_details(grad_results[ADAM_TRAJECTORY_V2_NAME])
    if ADAM_TRAJECTORY_LEGACY_NAME in grad_results:
        print_method_metrics(ADAM_TRAJECTORY_LEGACY_NAME, grad_results[ADAM_TRAJECTORY_LEGACY_NAME])

    output_dirs = base.build_output_dirs(out_path)
    loss_curve_b64 = base.plot_loss_curves(
        grad_results,
        output_stem=output_dirs["figure_dir"] / "current_loss_curves",
    )
    k_err_b64 = base.plot_K_error(
        grad_results,
        output_stem=output_dirs["figure_dir"] / "current_relative_k_error",
    )

    K_preds = OrderedDict()
    K_preds[STLS_NAME] = base.xi_to_K(stls_results[STLS_NAME]["xi"])
    for name, res in grad_results.items():
        K_preds[name] = base.xi_to_K(res["xi"])
    k_matrix_b64 = base.plot_K_matrices(
        stls_problem["K_true"],
        K_preds,
        output_stem=output_dirs["figure_dir"] / "current_recovered_k_matrices",
    )
    base.plot_stls_lambda_sweep(
        stls_problem["K_true"],
        stls_problem["theta_train"],
        stls_problem["target_train"],
        theta_test=stls_problem["theta_test"],
        target_test=stls_problem["target_test"],
        lam_values=base.STLS_DENSE_LAMBDA_SWEEP,
        mask=None,
        output_stem=output_dirs["figure_dir"] / "stls_dense_lambda_sweep_k_matrices",
        return_base64=False,
    )

    eig_blocks = []
    for name, res in grad_results.items():
        b64 = base.plot_eigenvalues(
            name,
            res,
            output_stem=output_dirs["figure_dir"] / f"current_eigenvalues_{base.slugify(name)}",
        )
        eig_blocks.append(
            f'<div class="method-block"><h3>{name}</h3>'
            f'<img src="data:image/png;base64,{b64}" /></div>'
        )

    summary_table = base.build_summary_table(grad_results, stls_results)
    presentation_blocks, summary_notes = base.generate_presentation_figures(
        data,
        stls_results,
        grad_results,
        out_path,
        device=rollout_device,
        simulation_backend=rollout_backend,
        dtype=dtype,
        focus_model_names=focus_model_names,
    )

    html = base.HTML_TEMPLATE.format(
        report_title=report_title,
        n=data["n"],
        n_dofs=data["free"].size,
        n_traj=data["n_traj"],
        dt=data["dt"],
        t_end=data["t_end"],
        spring_description=spring_description,
        n_total=grad_problem["n_total"],
        n_train=grad_problem["n_train"],
        n_test=grad_problem["n_total"] - grad_problem["n_train"],
        n_active=int(data["locality_free"].sum()),
        n_full=grad_problem["n_features"] * grad_problem["n_states"],
        pct_active=100 * float(data["locality_free"].sum()) / (grad_problem["n_features"] * grad_problem["n_states"]),
        summary_table=summary_table,
        loss_curve_b64=loss_curve_b64,
        k_err_b64=k_err_b64,
        k_matrix_b64=k_matrix_b64,
        eigenvalue_blocks="\n".join(eig_blocks),
        presentation_model_summary=format_model_summary(focus_model_names),
        objective_note=build_objective_note(
            trajectory_variant=trajectory_variant,
            legacy_stride=trajectory_stride,
            v2_window=trajectory_window,
            v2_window_stride=trajectory_window_stride,
        ),
        presentation_blocks=presentation_blocks,
        summary_notes=summary_notes,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  Wrote report to: {out_path.resolve()}")
    print(f"  Assets under: {output_dirs['figure_dir']}")
    return out_path


def main(
    *,
    residual_epochs: int,
    trajectory_epochs: int,
    trajectory_stride: int,
    trajectory_variant: str,
    trajectory_window: int,
    trajectory_window_stride: int,
    trajectory_patience: int,
    execution_mode: str,
    device_arg: str,
    simulation_backend: str,
) -> list[Path]:
    model_count = 4 if trajectory_variant == "both" else 3
    reports = []
    reports.append(
        generate_report(
            report_title=f"Spring-Grid K Regression Report (10x10 Random, {model_count} Models)",
            spring_description="horizontal, vertical, and diagonal pairs with constants ~ Uniform[0.5, 1.5]",
            out_path=Path("figures/spring_grid_10x10_random_three_model_report.html"),
            stiffness_matrix=None,
            dataset_seed=0,
            residual_epochs=residual_epochs,
            trajectory_epochs=trajectory_epochs,
            trajectory_stride=trajectory_stride,
            trajectory_variant=trajectory_variant,
            trajectory_window=trajectory_window,
            trajectory_window_stride=trajectory_window_stride,
            trajectory_patience=trajectory_patience,
            execution_mode=execution_mode,
            device_arg=device_arg,
            simulation_backend=simulation_backend,
        )
    )

    reports.append(
        generate_report(
            report_title=f"Spring-Grid K Regression Report (10x10 Physical, {model_count} Models)",
            spring_description=(
                "deterministic support-weighted lattice: vertical springs are slightly stiffer than horizontal ones, "
                "diagonals are softer braces, all springs stiffen near the clamped support row, and the center of "
                "the panel has mild reinforcement"
            ),
            out_path=Path("figures/spring_grid_10x10_physical_three_model_report.html"),
            stiffness_matrix=build_physically_motivated_stiffness_matrix(10),
            dataset_seed=0,
            residual_epochs=residual_epochs,
            trajectory_epochs=trajectory_epochs,
            trajectory_stride=trajectory_stride,
            trajectory_variant=trajectory_variant,
            trajectory_window=trajectory_window,
            trajectory_window_stride=trajectory_window_stride,
            trajectory_patience=trajectory_patience,
            execution_mode=execution_mode,
            device_arg=device_arg,
            simulation_backend=simulation_backend,
        )
    )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-epochs", type=int, default=600)
    parser.add_argument("--trajectory-epochs", type=int, default=200)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument(
        "--trajectory-variant",
        choices=("v2", "legacy", "both"),
        default="v2",
    )
    parser.add_argument("--trajectory-window", type=int, default=101)
    parser.add_argument("--trajectory-window-stride", type=int, default=50)
    parser.add_argument("--trajectory-patience", type=int, default=20)
    parser.add_argument(
        "--execution-mode",
        choices=("auto", "single", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--simulation-backend",
        choices=("auto", "scipy", "torch"),
        default="auto",
    )
    sindy_torch.add_device_arg(parser)
    args = parser.parse_args()
    report_paths = main(
        residual_epochs=args.residual_epochs,
        trajectory_epochs=args.trajectory_epochs,
        trajectory_stride=args.trajectory_stride,
        trajectory_variant=args.trajectory_variant,
        trajectory_window=args.trajectory_window,
        trajectory_window_stride=args.trajectory_window_stride,
        trajectory_patience=args.trajectory_patience,
        execution_mode=args.execution_mode,
        device_arg=args.device,
        simulation_backend=args.simulation_backend,
    )
    print("\nGenerated reports:")
    for report_path in report_paths:
        print(f"  {report_path}")
