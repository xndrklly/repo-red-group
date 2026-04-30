"""
Regress the stiffness matrix K of a 10x10 spring grid from simulated
displacement / acceleration data using the methods of sindy_torch.

Pipeline:
    1. Build a random K on a 10x10 lattice (H/V/diagonal springs).
    2. Pin the bottom row, leave 90 free DOFs, M = I, no damping, no forcing.
    3. Run several initial-condition trajectories with Newmark-beta.
    4. Form theta = u (free DOFs) and target = u_tt (free DOFs).
       The relationship is u_tt = -K_free u, so SINDy recovers
       K_pred = -Xi^T (here Xi has shape (n_free, n_free)).
    5. Compare methods:
        - STLS at two thresholds
        - SINDy Adam derivative-matching, no L1
        - SINDy Adam derivative-matching, L1
        - SINDy ISTA (proximal gradient)
       For the gradient methods, snapshot Xi every few epochs and
       track the lowest 5 eigenvalues of the symmetrised K_pred.
    6. Emit an HTML report with both the existing regression diagnostics and
       presentation-oriented rollout / modal diagnostics for three methods:
        - STLS dense (lam=0.05)
        - SINDy Adam local (no L1)
        - SINDy Adam dense (no L1)

Usage:
    python examples/spring_grid_K_regression.py
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sindy_torch
from sindy_torch.systems import (
    build_locality_mask,
    build_stiffness_matrix,
    get_free_dofs,
    newmark_beta_simulation,
    newmark_beta_simulation_torch,
    node_index,
    zero_force,
    zero_force_torch,
)


plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})


FOCUS_MODEL_NAMES = (
    "STLS dense (lam=0.05)",
    "SINDy Adam local (no L1)",
    "SINDy Adam dense (no L1)",
)

MODEL_COLORS = {
    "true": "#2563eb",
    "pred": "#d97706",
}

LOCAL_DIRECTION_OFFSETS = OrderedDict([
    ("left", (0, -1)),
    ("right", (0, 1)),
    ("down", (-1, 0)),
    ("up", (1, 0)),
    ("diag_down_left", (-1, -1)),
    ("diag_up_right", (1, 1)),
    ("diag_up_left", (1, -1)),
    ("diag_down_right", (-1, 1)),
])


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n: int = 10,
    n_traj: int = 4,
    t_end: float = 20.0,
    dt: float = 0.01,
    seed: int = 0,
    device: torch.device | None = None,
    simulation_backend: str = "auto",
    dtype: torch.dtype = torch.float64,
    stiffness_matrix: sp.csr_matrix | np.ndarray | None = None,
    damping_matrix: sp.csr_matrix | np.ndarray | None = None,
    force_fn=None,
    pinned_rows: tuple = (0,),
    initial_condition_mode: str = "random",
):
    """Generate trajectories on an n x n grid, return numpy / torch-ready data.

    Parameters
    ----------
    damping_matrix : optional
        Known damping matrix C (defaults to zero). Treated as known when
        building the K-only regression target ``A_target = A + V @ C^T - F``.
    force_fn : optional callable t -> (N,) array
        Known external force. Defaults to zero. The values evaluated on free
        DOFs are stored as ``F`` and subtracted from the regression target.
    pinned_rows : tuple of int
        Which rows to pin (set to zero displacement / velocity).
    initial_condition_mode : "random" or "zero"
        "random" matches the legacy behaviour (random u0 + occasional v0 kick).
        "zero" uses identically-zero initial conditions, so the response is
        driven purely by ``force_fn``.
    """
    rng = np.random.default_rng(seed)
    N = n * n
    if stiffness_matrix is None:
        K = build_stiffness_matrix(n, seed=seed, k_low=0.5, k_high=1.5)
    elif sp.issparse(stiffness_matrix):
        K = stiffness_matrix.tocsr()
    else:
        K = sp.csr_matrix(np.asarray(stiffness_matrix, dtype=float))
    pinned, free = get_free_dofs(n, pinned_rows=pinned_rows)
    sim_backend = resolve_simulation_backend(simulation_backend, device)

    if damping_matrix is None:
        C_csr = sp.csr_matrix((N, N))
    elif sp.issparse(damping_matrix):
        C_csr = damping_matrix.tocsr()
    else:
        C_csr = sp.csr_matrix(np.asarray(damping_matrix, dtype=float))
    C_full_np = C_csr.toarray()
    C_free_np = C_full_np[np.ix_(free, free)]

    # ``force_fn`` may be a single callable (used for all trajectories) or a
    # sequence of length n_traj, one callable per trajectory. The latter form
    # is useful when running zero-IC studies where excitation diversity must
    # come from the forcing rather than from the initial conditions.
    if force_fn is None:
        force_fns_np = [zero_force(N) for _ in range(n_traj)]
    elif callable(force_fn):
        force_fns_np = [force_fn for _ in range(n_traj)]
    else:
        force_fns_np = list(force_fn)
        if len(force_fns_np) != n_traj:
            raise ValueError(
                f"force_fn sequence length {len(force_fns_np)} must match n_traj={n_traj}"
            )

    if sim_backend == "torch":
        torch_device = device or torch.device("cpu")
        K_sim = torch.as_tensor(K.toarray(), device=torch_device, dtype=dtype)
        M_sim = torch.eye(N, device=torch_device, dtype=dtype)
        C_sim = torch.as_tensor(C_full_np, device=torch_device, dtype=dtype)
    else:
        K_sim = K
        M_sim = sp.eye(N, format="csr")
        C_sim = C_csr

    if initial_condition_mode not in {"random", "zero"}:
        raise ValueError("initial_condition_mode must be 'random' or 'zero'")

    U_list, V_list, A_list, T_list, F_list, Atgt_list = [], [], [], [], [], []
    trajectories = []
    for traj in range(n_traj):
        u0 = np.zeros(N)
        v0 = np.zeros(N)
        if initial_condition_mode == "random":
            # Random initial displacement on free DOFs (excite many modes).
            u0[free] = 0.2 * rng.standard_normal(free.size)
            # Sometimes add an initial velocity kick instead.
            if traj % 2 == 1:
                v0[free] = 0.5 * rng.standard_normal(free.size)
        traj_force_fn = force_fns_np[traj]
        if sim_backend == "torch":
            result = newmark_beta_simulation_torch(
                K_sim, M_sim, C_sim, traj_force_fn, free, n,
                t_end=t_end, dt=dt, u0=u0, v0=v0,
                device=torch_device, dtype=dtype,
            )
        else:
            result = newmark_beta_simulation(
                K_sim, M_sim, C_sim, traj_force_fn, free, n,
                t_end=t_end, dt=dt, u0=u0, v0=v0,
            )
        result_times = sindy_torch.as_numpy(result.times).copy()
        result_u = sindy_torch.as_numpy(result.displacements).copy()
        result_v = sindy_torch.as_numpy(result.velocities).copy()
        result_a = sindy_torch.as_numpy(result.accelerations).copy()
        u_free = result_u[:, free]
        v_free = result_v[:, free]
        a_free = result_a[:, free]

        n_t = result_times.shape[0]
        f_full = np.zeros((n_t, N))
        for k_idx in range(n_t):
            f_full[k_idx] = np.asarray(traj_force_fn(float(result_times[k_idx]))).reshape(N)
        f_free = f_full[:, free]
        # Corrected regression target so that  target = -K_free @ u_free, i.e.
        # the K-only relation with damping and forcing moved to the LHS.
        a_target = a_free + v_free @ C_free_np.T - f_free

        U_list.append(u_free)
        V_list.append(v_free)
        A_list.append(a_free)
        T_list.append(result_times)
        F_list.append(f_free)
        Atgt_list.append(a_target)
        trajectories.append({
            "times": result_times.copy(),
            "U_full": result_u.copy(),
            "V_full": result_v.copy(),
            "A_full": result_a.copy(),
            "U_free": u_free.copy(),
            "V_free": v_free.copy(),
            "A_free": a_free.copy(),
            "F_full": f_full.copy(),
            "F_free": f_free.copy(),
            "A_target": a_target.copy(),
            "u0_full": result_u[0].copy(),
            "v0_full": result_v[0].copy(),
            "traj_index": traj,
        })

    U = np.concatenate(U_list, axis=0)
    V = np.concatenate(V_list, axis=0)
    A = np.concatenate(A_list, axis=0)
    F = np.concatenate(F_list, axis=0)
    A_target = np.concatenate(Atgt_list, axis=0)
    times = np.concatenate(T_list, axis=0)

    # K_free for ground truth.
    K_full = K.toarray()
    K_free = K[free, :][:, free].toarray()

    # Regression locality mask: 8-connected (H + V + both diagonals) + self.
    # The anti-diagonal is included even though build_stiffness_matrix does
    # not place springs there -- physically that's a real possibility, and we
    # want the regressor to discover (near-)zero anti-diagonal entries from
    # the data, not to bake that prior in.
    locality_full = build_locality_mask(n, include_anti_diagonal=True)
    locality_free = locality_full[np.ix_(free, free)]

    return {
        "n": n,
        "N": N,
        "free": free,
        "pinned": pinned,
        "U": U,
        "V": V,
        "A": A,
        "F": F,
        "A_target": A_target,
        "times": times,
        "K_free": K_free,
        "K_full": K_full,
        "C_full": C_full_np,
        "C_free": C_free_np,
        "locality_free": locality_free,
        "n_traj": n_traj,
        "dt": dt,
        "t_end": t_end,
        "trajectories": trajectories,
        "simulation_backend": sim_backend,
        "initial_condition_mode": initial_condition_mode,
        "pinned_rows": tuple(int(r) for r in pinned_rows),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def xi_to_K(xi: torch.Tensor) -> torch.Tensor:
    """Recover a symmetric stiffness matrix from the coefficient matrix."""
    K_pred = -xi.detach().T
    return 0.5 * (K_pred + K_pred.T)


def k_eigenvalues(K: torch.Tensor, n_eig: int) -> np.ndarray:
    evals = torch.linalg.eigvalsh(K).cpu().numpy()
    return evals[:n_eig]


def relative_K_error(K_pred: torch.Tensor, K_true: torch.Tensor) -> float:
    return (torch.norm(K_pred - K_true) / torch.norm(K_true)).item()


def derivative_mse(theta: torch.Tensor, xi: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = theta @ xi
        return F.mse_loss(pred, target).item()


def sync_device(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_simulation_backend(
    simulation_backend: str,
    device: torch.device | None,
) -> str:
    if simulation_backend == "auto":
        return "torch" if device is not None and device.type == "cuda" else "scipy"
    if simulation_backend not in {"scipy", "torch"}:
        raise ValueError(
            "simulation_backend must be one of: 'auto', 'scipy', 'torch'"
        )
    return simulation_backend


def resolve_execution_plan(
    device_arg: str,
    execution_mode: str,
) -> dict[str, torch.device | str]:
    if execution_mode not in {"auto", "single", "hybrid"}:
        raise ValueError("execution_mode must be one of: 'auto', 'single', 'hybrid'")

    if execution_mode == "auto":
        use_hybrid = (device_arg == "auto") and torch.cuda.is_available()
    else:
        use_hybrid = execution_mode == "hybrid"

    if use_hybrid:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Hybrid execution requires CUDA, but torch.cuda.is_available() is False."
            )
        dataset_device = torch.device("cpu")
        stls_device = torch.device("cpu")
        training_device = torch.device("cuda")
        rollout_device = torch.device("cpu")
        report_device = torch.device("cpu")
        mode = "hybrid"
    else:
        single_device = sindy_torch.get_device(device_arg)
        dataset_device = single_device
        stls_device = single_device
        training_device = single_device
        rollout_device = single_device
        report_device = single_device
        mode = "single"

    return {
        "mode": mode,
        "dataset_device": dataset_device,
        "stls_device": stls_device,
        "training_device": training_device,
        "rollout_device": rollout_device,
        "report_device": report_device,
    }


def build_regression_problem(
    data: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    U = torch.as_tensor(data["U"], device=device, dtype=dtype)
    # Use the corrected target (a + C v - f) when damping/forcing are present;
    # falls back to the raw acceleration in the legacy zero-C / zero-f case.
    target_array = data.get("A_target", data["A"])
    A = torch.as_tensor(target_array, device=device, dtype=dtype)
    K_true = torch.as_tensor(data["K_free"], device=device, dtype=dtype)

    n_total = U.shape[0]
    n_train = int(0.8 * n_total)
    rng = np.random.default_rng(123)
    perm = torch.as_tensor(rng.permutation(n_total), device=device, dtype=torch.long)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    theta_train = U[train_idx]
    target_train = A[train_idx]
    theta_test = U[test_idx]
    target_test = A[test_idx]

    xi_mask_np = data["locality_free"]
    xi_mask = torch.as_tensor(xi_mask_np, device=device)

    return {
        "U": U,
        "A": A,
        "K_true": K_true,
        "n_total": n_total,
        "n_train": n_train,
        "theta_train": theta_train,
        "target_train": target_train,
        "theta_test": theta_test,
        "target_test": target_test,
        "xi_mask_np": xi_mask_np,
        "xi_mask": xi_mask,
        "n_features": U.shape[1],
        "n_states": U.shape[1],
    }


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "model"


def path_href(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, from_dir).replace("\\", "/")


def node_coords(node: int, n: int) -> tuple[int, int]:
    return divmod(node, n)


def grid_xy(n: int) -> tuple[np.ndarray, np.ndarray]:
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    return jj.astype(float), ii.astype(float)


def grid_positions(n: int) -> np.ndarray:
    x, y = grid_xy(n)
    return np.column_stack([x.ravel(), y.ravel()])


def reshape_node_values(values: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(values).reshape(n, n)


def build_physical_edge_specs(n: int) -> list[dict]:
    specs = []
    for i in range(n):
        for j in range(n):
            a = node_index(i, j, n)
            if j + 1 < n:
                b = node_index(i, j + 1, n)
                specs.append({"a": a, "b": b, "kind": "horizontal"})
            if i + 1 < n:
                b = node_index(i + 1, j, n)
                specs.append({"a": a, "b": b, "kind": "vertical"})
            if i + 1 < n and j + 1 < n:
                b = node_index(i + 1, j + 1, n)
                specs.append({"a": a, "b": b, "kind": "diag_up_right"})
    return specs


def build_full_K_from_free(K_free: np.ndarray | torch.Tensor, free: np.ndarray, n_total: int) -> np.ndarray:
    if isinstance(K_free, torch.Tensor):
        K_free_np = K_free.detach().cpu().numpy()
    else:
        K_free_np = np.asarray(K_free, dtype=float)
    K_full = np.zeros((n_total, n_total), dtype=float)
    K_full[np.ix_(free, free)] = K_free_np
    return K_full


def build_full_K_from_free_torch(
    K_free: torch.Tensor,
    free: np.ndarray | torch.Tensor,
    n_total: int,
) -> torch.Tensor:
    free_t = torch.as_tensor(free, device=K_free.device, dtype=torch.long)
    K_full = torch.zeros((n_total, n_total), device=K_free.device, dtype=K_free.dtype)
    row_idx = free_t.unsqueeze(1).expand(-1, free_t.numel())
    col_idx = free_t.unsqueeze(0).expand(free_t.numel(), -1)
    return K_full.index_put((row_idx, col_idx), K_free)


def full_series_from_free(series_free: np.ndarray, free: np.ndarray, n_total: int) -> np.ndarray:
    series_free = np.asarray(series_free, dtype=float)
    full = np.zeros((series_free.shape[0], n_total), dtype=float)
    full[:, free] = series_free
    return full


def compute_visual_max(values_true: np.ndarray, values_pred: np.ndarray, percentile: float = 99.5) -> float:
    vmax = max(
        float(np.percentile(np.abs(values_true), percentile)),
        float(np.percentile(np.abs(values_pred), percentile)),
        1e-8,
    )
    return vmax


def pick_time_indices(times: np.ndarray, fractions: list[float]) -> list[int]:
    idxs = []
    for frac in fractions:
        idx = int(round(frac * (len(times) - 1)))
        idxs.append(max(0, min(len(times) - 1, idx)))
    return idxs


def sample_frame_indices(n_frames_total: int, max_frames: int = 120) -> np.ndarray:
    if n_frames_total <= max_frames:
        return np.arange(n_frames_total)
    return np.unique(np.linspace(0, n_frames_total - 1, max_frames).round().astype(int))


def save_static_figure(fig, output_stem: Path, dpi: int = 220) -> dict[str, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return {"png": png_path, "pdf": pdf_path}


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def maybe_save_for_html(fig, output_stem: Path | None = None, return_base64: bool = True) -> str | None:
    if output_stem is not None:
        save_static_figure(fig, output_stem)
    if return_base64:
        return fig_to_base64(fig)
    plt.close(fig)
    return None


def save_animation(anim: animation.FuncAnimation, output_stem: Path, fps: int = 20) -> dict[str, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    mp4_path = output_stem.with_suffix(".mp4")
    if animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(mp4_path, writer=writer, dpi=180)
        outputs["mp4"] = mp4_path
    else:
        gif_path = output_stem.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=140)
        outputs["gif"] = gif_path
    plt.close(anim._fig)
    return outputs


def render_image_tag(html_dir: Path, asset_path: Path, alt: str) -> str:
    href = path_href(html_dir, asset_path)
    return f'<img src="{href}" alt="{alt}" />'


def render_video_tag(html_dir: Path, asset_paths: dict[str, Path], title: str) -> str:
    if "mp4" in asset_paths:
        href = path_href(html_dir, asset_paths["mp4"])
        return (
            f'<div class="video-card"><h5>{title}</h5>'
            f'<video controls preload="none" playsinline>'
            f'<source src="{href}" type="video/mp4" /></video></div>'
        )
    href = path_href(html_dir, asset_paths["gif"])
    return (
        f'<div class="video-card"><h5>{title}</h5>'
        f'<img src="{href}" alt="{title}" /></div>'
    )


def describe_node(node: int, n: int) -> str:
    i, j = node_coords(node, n)
    return f"node {node} @ (row={i}, col={j})"


def select_representative_nodes(
    n: int,
    free: np.ndarray,
    trajectory_full: np.ndarray,
) -> OrderedDict[str, int]:
    """Pick a free corner, free edge, center-ish interior, and active interior."""
    free_set = set(int(v) for v in free.tolist())
    amplitude = np.std(trajectory_full, axis=0)

    corners = [
        node_index(0, 0, n),
        node_index(0, n - 1, n),
        node_index(n - 1, 0, n),
        node_index(n - 1, n - 1, n),
    ]
    free_corners = [idx for idx in corners if idx in free_set]
    if not free_corners:
        raise RuntimeError("No free corner nodes available for diagnostics")
    corner = max(free_corners, key=lambda idx: amplitude[idx])

    edge_nodes = []
    interior_nodes = []
    for idx in free:
        i, j = node_coords(int(idx), n)
        is_boundary = i in (0, n - 1) or j in (0, n - 1)
        is_corner = (i in (0, n - 1)) and (j in (0, n - 1))
        if is_boundary and not is_corner:
            edge_nodes.append(int(idx))
        if 0 < i < n - 1 and 0 < j < n - 1:
            interior_nodes.append(int(idx))

    if not edge_nodes or not interior_nodes:
        raise RuntimeError("Not enough free edge/interior nodes for diagnostics")

    edge = max(edge_nodes, key=lambda idx: amplitude[idx])
    center_ref = np.array([(n - 1) / 2.0, (n - 1) / 2.0])
    interior1 = min(
        interior_nodes,
        key=lambda idx: np.linalg.norm(np.array(node_coords(idx, n)) - center_ref),
    )
    interior_candidates = [idx for idx in interior_nodes if idx != interior1]
    interior2 = max(interior_candidates, key=lambda idx: amplitude[idx])

    return OrderedDict([
        ("corner", corner),
        ("edge", edge),
        ("interior1", interior1),
        ("interior2", interior2),
    ])


def build_incident_specs(selected_nodes: OrderedDict[str, int], n: int) -> list[dict]:
    specs = []
    for node_label, node in selected_nodes.items():
        i, j = node_coords(node, n)
        for direction, (di, dj) in LOCAL_DIRECTION_OFFSETS.items():
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n:
                neighbor = node_index(ni, nj, n)
                specs.append({
                    "series_key": f"{node_label}:{direction}",
                    "node_label": node_label,
                    "node": node,
                    "neighbor": neighbor,
                    "direction": direction,
                })
    return specs


def rollout_linear_model(
    K_free: np.ndarray | torch.Tensor,
    free: np.ndarray,
    n: int,
    dt: float,
    t_end: float,
    u0_full: np.ndarray,
    v0_full: np.ndarray,
    device: torch.device | None = None,
    simulation_backend: str = "auto",
    dtype: torch.dtype = torch.float64,
    damping_matrix: np.ndarray | sp.spmatrix | None = None,
    force_fn=None,
) -> dict:
    """Roll out a regressed K under the same M / C / forcing as the data.

    ``damping_matrix`` and ``force_fn`` default to zero so the legacy
    no-damping / no-forcing rollouts keep working. Pass the dataset's
    ``C_full`` and the original ``force_fn`` to compare against forced
    trajectories generated with damping.
    """
    sim_backend = resolve_simulation_backend(simulation_backend, device)
    n_total = n * n
    K_full = build_full_K_from_free(K_free, free, n_total)
    if damping_matrix is None:
        C_dense = np.zeros((n_total, n_total), dtype=float)
    elif sp.issparse(damping_matrix):
        C_dense = damping_matrix.toarray()
    else:
        C_dense = np.asarray(damping_matrix, dtype=float)
    if sim_backend == "torch":
        torch_device = device or torch.device("cpu")
        K_sim = torch.as_tensor(K_full, device=torch_device, dtype=dtype)
        M_sim = torch.eye(n_total, device=torch_device, dtype=dtype)
        C_sim = torch.as_tensor(C_dense, device=torch_device, dtype=dtype)
        force_sim = (
            zero_force_torch(n_total, device=torch_device, dtype=dtype)
            if force_fn is None
            else force_fn
        )
        result = newmark_beta_simulation_torch(
            K_sim,
            M_sim,
            C_sim,
            force_sim,
            free,
            n,
            t_end=t_end,
            dt=dt,
            u0=u0_full.copy(),
            v0=v0_full.copy(),
            device=torch_device,
            dtype=dtype,
        )
    else:
        result = newmark_beta_simulation(
            sp.csr_matrix(K_full),
            sp.eye(n_total, format="csr"),
            sp.csr_matrix(C_dense),
            zero_force(n_total) if force_fn is None else force_fn,
            free,
            n,
            t_end=t_end,
            dt=dt,
            u0=u0_full.copy(),
            v0=v0_full.copy(),
        )
    return {
        "times": sindy_torch.as_numpy(result.times).copy(),
        "U_full": sindy_torch.as_numpy(result.displacements).copy(),
        "V_full": sindy_torch.as_numpy(result.velocities).copy(),
        "A_full": sindy_torch.as_numpy(result.accelerations).copy(),
        "K_full": K_full,
        "simulation_backend": sim_backend,
    }


def draw_lattice_background(ax, edge_specs: list[dict], positions: np.ndarray, color: str = "#cbd5e1", alpha: float = 0.5):
    segments = []
    for spec in edge_specs:
        segments.append([positions[spec["a"]], positions[spec["b"]]])
    lc = LineCollection(segments, colors=color, linewidths=0.8, alpha=alpha, zorder=1)
    ax.add_collection(lc)


def plot_lattice_stiffness(K_true_full: np.ndarray, n: int, output_stem: Path) -> dict[str, Path]:
    positions = grid_positions(n)
    edge_specs = build_physical_edge_specs(n)
    segments = []
    stiffness = []
    for spec in edge_specs:
        a, b = spec["a"], spec["b"]
        segments.append([positions[a], positions[b]])
        stiffness.append(float(-K_true_full[a, b]))
    stiffness = np.asarray(stiffness)

    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    lc = LineCollection(
        segments,
        array=stiffness,
        cmap="viridis",
        linewidths=2.2 if n <= 12 else 1.2,
        zorder=2,
    )
    ax.add_collection(lc)
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=24 if n <= 12 else 10,
        c="#111827",
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    cb = fig.colorbar(lc, ax=ax, shrink=0.88)
    cb.set_label("True spring stiffness")
    ax.set_title("Figure 1: undeformed lattice, true spring stiffness")
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.grid(alpha=0.12)
    return save_static_figure(fig, output_stem)


def make_quiver_snapshot_figure(
    true_rollout: dict,
    pred_rollout: dict,
    n: int,
    model_name: str,
    output_stem: Path,
) -> dict[str, Path]:
    positions = grid_positions(n)
    edge_specs = build_physical_edge_specs(n)
    times = true_rollout["times"]
    idxs = pick_time_indices(times, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    u_true = true_rollout["U_full"]
    u_pred = pred_rollout["U_full"]
    vis_max = compute_visual_max(u_true, u_pred)
    vis_scale = 0.75 / vis_max

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.4), constrained_layout=True)
    row_info = [
        ("True / data", u_true, MODEL_COLORS["true"]),
        ("Regressed / model", u_pred, MODEL_COLORS["pred"]),
    ]
    for row_idx, (row_name, data, color) in enumerate(row_info):
        for col_idx, frame_idx in enumerate(idxs):
            ax = axes[row_idx, col_idx]
            draw_lattice_background(ax, edge_specs, positions)
            ax.scatter(positions[:, 0], positions[:, 1], s=10, c="#334155", zorder=2)
            vals = data[frame_idx]
            ax.quiver(
                positions[:, 0],
                positions[:, 1],
                np.zeros_like(vals),
                vals * vis_scale,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.005,
                headwidth=3.8,
                headlength=4.8,
                headaxislength=4.5,
                color=color,
                zorder=3,
            )
            ax.set_aspect("equal")
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(-0.5, n - 0.5)
            ax.set_title(f"{row_name}, t = {times[frame_idx]:.2f}")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.grid(alpha=0.12)
    fig.suptitle(
        f"Figure 2A: scalar-deflection quiver snapshots, {model_name}\n"
        f"Arrows render out-of-plane displacement as vertical screen-space arrows; "
        f"visual scale = {vis_scale:.2f} per unit displacement.",
        fontsize=12,
    )
    return save_static_figure(fig, output_stem)


def make_quiver_animation(
    rollout: dict,
    n: int,
    title: str,
    output_stem: Path,
    vis_scale: float,
) -> dict[str, Path]:
    positions = grid_positions(n)
    edge_specs = build_physical_edge_specs(n)
    frame_indices = sample_frame_indices(len(rollout["times"]))
    times = rollout["times"][frame_indices]
    values = rollout["U_full"][frame_indices]

    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    draw_lattice_background(ax, edge_specs, positions)
    ax.scatter(positions[:, 0], positions[:, 1], s=10, c="#334155", zorder=2)
    quiv = ax.quiver(
        positions[:, 0],
        positions[:, 1],
        np.zeros(positions.shape[0]),
        values[0] * vis_scale,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.005,
        headwidth=3.8,
        headlength=4.8,
        headaxislength=4.5,
        color="#2563eb" if "true" in title.lower() else "#d97706",
        zorder=3,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(alpha=0.12)
    title_text = ax.set_title(f"{title}, t = {times[0]:.2f}")

    def update(frame_id):
        vals = values[frame_id]
        quiv.set_UVC(np.zeros_like(vals), vals * vis_scale)
        title_text.set_text(f"{title}, t = {times[frame_id]:.2f}")
        return quiv, title_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=45,
        blit=False,
    )
    return save_animation(anim, output_stem, fps=20)


def make_heatmap_snapshot_figure(
    true_rollout: dict,
    pred_rollout: dict,
    n: int,
    model_name: str,
    output_stem: Path,
) -> dict[str, Path]:
    times = true_rollout["times"]
    idxs = pick_time_indices(times, [1.0 / 8.0, 3.0 / 8.0, 5.0 / 8.0, 7.0 / 8.0])
    u_true = np.abs(true_rollout["U_full"])
    u_pred = np.abs(pred_rollout["U_full"])
    vmax = compute_visual_max(u_true, u_pred)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4), constrained_layout=True)
    row_info = [
        ("True / data", u_true),
        ("Regressed / model", u_pred),
    ]
    im = None
    for row_idx, (row_name, data) in enumerate(row_info):
        for col_idx, frame_idx in enumerate(idxs):
            ax = axes[row_idx, col_idx]
            grid = reshape_node_values(data[frame_idx], n)
            im = ax.imshow(
                grid,
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_title(f"{row_name}, t = {times[frame_idx]:.2f}")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.94)
        cbar.set_label("Displacement magnitude")
    fig.suptitle(f"Figure 2C: displacement-magnitude heatmaps, {model_name}", fontsize=12)
    return save_static_figure(fig, output_stem)


def make_heatmap_animation(
    rollout: dict,
    n: int,
    title: str,
    output_stem: Path,
    vmax: float,
) -> dict[str, Path]:
    frame_indices = sample_frame_indices(len(rollout["times"]))
    times = rollout["times"][frame_indices]
    values = np.abs(rollout["U_full"][frame_indices])

    fig, ax = plt.subplots(figsize=(6.4, 5.9), constrained_layout=True)
    image = ax.imshow(
        reshape_node_values(values[0], n),
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    cb = fig.colorbar(image, ax=ax, shrink=0.88)
    cb.set_label("Displacement magnitude")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    title_text = ax.set_title(f"{title}, t = {times[0]:.2f}")

    def update(frame_id):
        image.set_data(reshape_node_values(values[frame_id], n))
        title_text.set_text(f"{title}, t = {times[frame_id]:.2f}")
        return image, title_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=45,
        blit=False,
    )
    return save_animation(anim, output_stem, fps=20)


def make_node_trace_plots(
    true_rollout: dict,
    pred_rollout: dict,
    selected_nodes: OrderedDict[str, int],
    n: int,
    model_name: str,
    output_stem: Path,
) -> dict[str, Path]:
    times = true_rollout["times"]
    quantities = [
        ("Displacement u", "U_full"),
        ("Velocity v", "V_full"),
        ("Acceleration a", "A_full"),
    ]
    fig, axes = plt.subplots(
        len(selected_nodes),
        len(quantities),
        figsize=(15.8, 2.7 * len(selected_nodes)),
        sharex=True,
        constrained_layout=True,
    )
    if len(selected_nodes) == 1:
        axes = np.asarray([axes])

    for row_idx, (node_label, node) in enumerate(selected_nodes.items()):
        node_desc = describe_node(node, n)
        for col_idx, (quantity_label, key) in enumerate(quantities):
            ax = axes[row_idx, col_idx]
            ax.plot(times, true_rollout[key][:, node], color=MODEL_COLORS["true"], lw=1.8, label="True")
            ax.plot(times, pred_rollout[key][:, node], color=MODEL_COLORS["pred"], lw=1.4, linestyle="--", label="Predicted")
            if row_idx == 0:
                ax.set_title(quantity_label)
            if col_idx == 0:
                ax.set_ylabel(f"{node_label}\n{node_desc}")
            ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Figure 3: node time traces, {model_name}", fontsize=12)
    return save_static_figure(fig, output_stem)


def make_parameter_history_plots(
    model_name: str,
    selected_nodes: OrderedDict[str, int],
    incident_specs: list[dict],
    K_true_full: np.ndarray,
    K_history_full: dict[str, np.ndarray],
    history_epochs: np.ndarray,
    n: int,
    output_stem: Path,
    note: str | None = None,
) -> dict[str, Path]:
    node_to_specs: dict[str, list[dict]] = {label: [] for label in selected_nodes}
    for spec in incident_specs:
        node_to_specs[spec["node_label"]].append(spec)

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.5), constrained_layout=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("tab10")
    for ax, (node_label, node) in zip(axes, selected_nodes.items()):
        specs = node_to_specs[node_label]
        for idx, spec in enumerate(specs):
            color = cmap(idx % 10)
            series = K_history_full[spec["series_key"]]
            true_k = float(-K_true_full[spec["node"], spec["neighbor"]])
            ax.plot(history_epochs, series, color=color, lw=1.7, label=spec["direction"])
            ax.axhline(true_k, color=color, linestyle="--", alpha=0.85)
        ax.set_title(f"{node_label}: {describe_node(node, n)}")
        ax.set_xlabel("Optimization iteration")
        ax.set_ylabel("Incident stiffness")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    subtitle = "Solid = estimate, dashed = true."
    if note:
        subtitle = f"{subtitle} {note}"
    fig.suptitle(f"Figure 5: local stiffness evolution, {model_name}\n{subtitle}", fontsize=12)
    return save_static_figure(fig, output_stem)


def modal_analysis(K_free: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(K_free)
    omegas = np.sqrt(np.clip(evals, 0.0, None))
    return evals, omegas, evecs


def make_modal_comparison(
    model_name: str,
    K_true_free: np.ndarray,
    K_pred_free: np.ndarray,
    output_stem: Path,
) -> dict[str, Path]:
    evals_true, omega_true, _ = modal_analysis(K_true_free)
    evals_pred, omega_pred, _ = modal_analysis(K_pred_free)
    rel_err = np.abs(omega_pred - omega_true) / np.maximum(np.abs(omega_true), 1e-8)
    neg_count = int(np.sum(evals_pred < 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    mode_idx = np.arange(1, len(omega_true) + 1)
    axes[0].plot(mode_idx, omega_true, color=MODEL_COLORS["true"], lw=2.0, label="True")
    axes[0].plot(mode_idx, omega_pred, color=MODEL_COLORS["pred"], lw=1.6, linestyle="--", label="Predicted")
    axes[0].set_xlabel("Mode number")
    axes[0].set_ylabel("Modal frequency")
    axes[0].set_title("Sorted modal frequencies")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].semilogy(mode_idx, rel_err, color="#7c3aed", lw=1.8)
    axes[1].set_xlabel("Mode number")
    axes[1].set_ylabel("Relative frequency error")
    axes[1].set_title("Per-mode relative frequency error")
    axes[1].grid(alpha=0.25)

    note = ""
    if neg_count > 0:
        note = f" Predicted K has {neg_count} negative eigenvalues; frequency plot clips those to omega = 0."
    fig.suptitle(f"Figure 9: modal comparison, {model_name}.{note}", fontsize=12)
    return save_static_figure(fig, output_stem)


def make_mode_shape_examples(
    model_name: str,
    K_true_free: np.ndarray,
    K_pred_free: np.ndarray,
    n: int,
    free: np.ndarray,
    output_stem: Path,
    n_modes: int = 4,
) -> dict[str, Path]:
    _, _, vecs_true = modal_analysis(K_true_free)
    _, _, vecs_pred = modal_analysis(K_pred_free)
    n_total = n * n

    fig, axes = plt.subplots(n_modes, 2, figsize=(8.6, 3.1 * n_modes), constrained_layout=True)
    if n_modes == 1:
        axes = np.asarray([axes])
    for mode_idx in range(n_modes):
        v_true = np.zeros(n_total)
        v_pred = np.zeros(n_total)
        v_true[free] = vecs_true[:, mode_idx]
        v_pred[free] = vecs_pred[:, mode_idx]
        if np.dot(vecs_true[:, mode_idx], vecs_pred[:, mode_idx]) < 0:
            v_pred *= -1.0
        true_grid = reshape_node_values(v_true, n)
        pred_grid = reshape_node_values(v_pred, n)
        vmax = max(float(np.max(np.abs(true_grid))), float(np.max(np.abs(pred_grid))), 1e-8)
        im0 = axes[mode_idx, 0].imshow(true_grid, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        im1 = axes[mode_idx, 1].imshow(pred_grid, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[mode_idx, 0].set_title(f"True mode {mode_idx + 1}")
        axes[mode_idx, 1].set_title(f"Predicted mode {mode_idx + 1}")
        axes[mode_idx, 0].set_ylabel("Row")
        axes[mode_idx, 1].set_ylabel("Row")
        for ax in axes[mode_idx]:
            ax.set_xlabel("Column")
        fig.colorbar(im1, ax=axes[mode_idx], shrink=0.84)
    fig.suptitle(f"Figure 9: representative mode shapes, {model_name}", fontsize=12)
    return save_static_figure(fig, output_stem)


def build_output_dirs(out_path: Path) -> dict[str, Path]:
    if out_path.parent.name == "figures":
        project_root = out_path.parent.parent
    else:
        project_root = out_path.parent
    stem = out_path.stem
    if stem.endswith("_report"):
        asset_slug = stem[: -len("_report")]
    else:
        asset_slug = stem
    if not asset_slug:
        asset_slug = "spring_grid_K_regression"
    figure_dir = project_root / "results" / "figures" / asset_slug
    animation_dir = project_root / "results" / "animations" / asset_slug
    figure_dir.mkdir(parents=True, exist_ok=True)
    animation_dir.mkdir(parents=True, exist_ok=True)
    return {
        "project_root": project_root,
        "figure_dir": figure_dir,
        "animation_dir": animation_dir,
    }


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_gradient_method(
    theta_train,
    target_train,
    theta_test,
    target_test,
    K_true,
    n_features,
    n_states,
    n_epochs=600,
    lr=5e-2,
    l1_lambda=0.0,
    proximal=False,
    snapshot_every=10,
    n_eig=5,
    seed=0,
    xi_mask=None,
    history_specs: list[dict] | None = None,
    history_every: int = 1,
    free: np.ndarray | None = None,
    n_total_dofs: int | None = None,
):
    torch.manual_seed(seed)
    if theta_train.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    xi = nn.Parameter(
        torch.zeros(
            n_features,
            n_states,
            dtype=theta_train.dtype,
            device=theta_train.device,
        )
    )

    if xi_mask is not None:
        # DOF reduction: gradients on disallowed entries are zeroed before
        # the optimizer step, so those entries stay at their initial value
        # of zero throughout training. Effective parameter count is
        # mask.sum(), not n_features * n_states.
        mask_t = xi_mask.to(device=xi.device, dtype=xi.dtype)
        xi.register_hook(lambda grad: grad * mask_t)

    optimizer = sindy_torch.SparseOptimizer(
        xi, l1_lambda=l1_lambda, optimizer_kwargs={"lr": lr},
    )

    train_losses, test_losses = [], []
    snap_epochs, snap_evals, snap_K_err = [], [], []

    true_evals = k_eigenvalues(K_true, n_eig)

    history_epochs = []
    param_history = {spec["series_key"]: [] for spec in history_specs or []}

    for epoch in range(n_epochs):
        info = optimizer.step_derivative_matching(theta_train, target_train)
        if proximal:
            optimizer.proximal_step()
        if xi_mask is not None:
            # Adam's running moments and proximal soft-thresholding can
            # nudge masked entries away from zero; clamp them back.
            with torch.no_grad():
                xi.mul_(mask_t)
        train_losses.append(info["mse"])
        test_losses.append(derivative_mse(theta_test, xi.detach(), target_test))

        if history_specs and free is not None and n_total_dofs is not None:
            if epoch % history_every == 0 or epoch == n_epochs - 1:
                K_pred_full = build_full_K_from_free(xi_to_K(xi), free, n_total_dofs)
                history_epochs.append(epoch)
                for spec in history_specs:
                    k_val = float(-K_pred_full[spec["node"], spec["neighbor"]])
                    param_history[spec["series_key"]].append(k_val)

        if epoch % snapshot_every == 0 or epoch == n_epochs - 1:
            K_pred = xi_to_K(xi)
            snap_epochs.append(epoch)
            snap_evals.append(k_eigenvalues(K_pred, n_eig))
            snap_K_err.append(relative_K_error(K_pred, K_true))

    result = {
        "xi": xi.detach().clone(),
        "train_losses": np.asarray(train_losses),
        "test_losses": np.asarray(test_losses),
        "snap_epochs": np.asarray(snap_epochs),
        "snap_evals": np.asarray(snap_evals),  # (n_snap, n_eig)
        "snap_K_err": np.asarray(snap_K_err),
        "true_evals": true_evals,
    }
    if history_specs:
        result["parameter_history_epochs"] = np.asarray(history_epochs)
        result["parameter_history"] = {
            key: np.asarray(vals) for key, vals in param_history.items()
        }
    return result


def run_stls(
    theta_train, target_train, theta_test, target_test, K_true,
    lam, mask=None,
):
    if mask is None:
        xi = sindy_torch.stls(theta_train, target_train, lam=lam)
    else:
        mask = mask.to(device=theta_train.device)
        xi = sindy_torch.stls_masked(theta_train, target_train, mask, lam=lam)
    K_pred = xi_to_K(xi)
    return {
        "xi": xi,
        "train_mse": derivative_mse(theta_train, xi, target_train),
        "test_mse": derivative_mse(theta_test, xi, target_test),
        "K_err": relative_K_error(K_pred, K_true),
        "K_pred": K_pred,
    }


def summarize_method_metrics(stls_results: dict, grad_results: dict) -> dict[str, dict[str, float]]:
    summary = {}
    for name, res in stls_results.items():
        summary[name] = {
            "train_mse": float(res["train_mse"]),
            "test_mse": float(res["test_mse"]),
            "K_err": float(res["K_err"]),
        }
    for name, res in grad_results.items():
        summary[name] = {
            "train_mse": float(res["train_losses"][-1]),
            "test_mse": float(res["test_losses"][-1]),
            "K_err": float(res["snap_K_err"][-1]),
        }
    return summary


def benchmark_focus_rollouts(
    data: dict,
    stls_results: dict,
    grad_results: dict,
    *,
    device: torch.device | None,
    simulation_backend: str,
    dtype: torch.dtype = torch.float64,
) -> dict[str, dict[str, float]]:
    rollout_metrics = {}
    base_traj = data["trajectories"][0]
    free = data["free"]
    n_total = data["N"]

    for model_name in FOCUS_MODEL_NAMES:
        if model_name in stls_results:
            K_pred_free = stls_results[model_name]["K_pred"].detach()
        else:
            K_pred_free = xi_to_K(grad_results[model_name]["xi"]).detach()

        pred_rollout = rollout_linear_model(
            K_pred_free,
            free,
            data["n"],
            dt=data["dt"],
            t_end=data["t_end"],
            u0_full=base_traj["u0_full"],
            v0_full=base_traj["v0_full"],
            device=device,
            simulation_backend=simulation_backend,
            dtype=dtype,
        )
        mse = float(np.mean((pred_rollout["U_full"] - base_traj["U_full"]) ** 2))
        rel_k = float(
            np.linalg.norm(pred_rollout["K_full"] - data["K_full"])
            / np.linalg.norm(data["K_full"])
        )
        rollout_metrics[model_name] = {
            "rollout_mse": mse,
            "full_K_rel_error": rel_k,
            "n_total_dofs": float(n_total),
        }
    return rollout_metrics


# ---------------------------------------------------------------------------
# Plotting helpers (current HTML diagnostics)
# ---------------------------------------------------------------------------

def plot_loss_curves(results: dict, output_stem: Path | None = None) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, res in results.items():
        axes[0].plot(res["train_losses"], label=name)
        axes[1].plot(res["test_losses"], label=name)
    for ax, title in zip(axes, ("Train MSE", "Test MSE")):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


def plot_eigenvalues(name: str, res: dict, output_stem: Path | None = None) -> str:
    snap_epochs = res["snap_epochs"]
    snap_evals = res["snap_evals"]  # (n_snap, n_eig)
    true_evals = res["true_evals"]
    n_eig = snap_evals.shape[1]

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for i in range(n_eig):
        color = cmap(i)
        ax.plot(snap_epochs, snap_evals[:, i],
                color=color, label=f"eig {i + 1} (pred)")
        ax.axhline(true_evals[i], color=color, linestyle="--", alpha=0.7,
                   label=f"eig {i + 1} (true)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eigenvalue of K")
    ax.set_title(f"{name}: lowest {n_eig} eigenvalues vs epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    return maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


def plot_K_error(grad_results: dict, output_stem: Path | None = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for name, res in grad_results.items():
        ax.plot(res["snap_epochs"], res["snap_K_err"], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|K_pred - K_true| / |K_true|")
    ax.set_title("Relative K error vs epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


def plot_K_matrices(K_true: torch.Tensor, K_preds: dict, output_stem: Path | None = None) -> str:
    n_methods = len(K_preds)
    fig, axes = plt.subplots(
        1,
        n_methods + 1,
        figsize=(3.2 * (n_methods + 1), 3.2),
        constrained_layout=True,
    )
    K_true_np = K_true.cpu().numpy()
    vmax = float(np.abs(K_true_np).max())

    im = axes[0].imshow(K_true_np, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("K_true")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    for ax, (name, K) in zip(axes[1:], K_preds.items()):
        K_np = K.cpu().numpy()
        im = ax.imshow(K_np, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(name)
        plt.colorbar(im, ax=ax, shrink=0.8)

    return maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>{report_title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1180px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
h3 {{ margin-top: 1.6em; }}
h4 {{ margin-bottom: 0.4em; }}
h5 {{ margin: 0 0 0.4em 0; font-size: 0.95rem; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.4em 0.8em; text-align: right; }}
th {{ background: #f3f3f3; }}
img {{ max-width: 100%; height: auto; border: 1px solid #e5e7eb; }}
video {{ width: 100%; height: auto; border: 1px solid #e5e7eb; background: #000; }}
.method-block {{ margin-bottom: 2.5em; }}
code {{ background: #f5f5f5; padding: 0.1em 0.3em; border-radius: 3px; }}
.media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
               gap: 1rem; align-items: start; }}
.video-card {{ display: block; }}
.note-box {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.9em 1em; border-radius: 6px; }}
</style>
</head>
<body>
<h1>{report_title}</h1>

<h2>Setup</h2>
<ul>
<li>Grid: <code>{n}x{n}</code> ({n_dofs} free DOFs after pinning the bottom row)</li>
<li>Springs: {spring_description}</li>
<li>M = I, C = 0, no external forcing</li>
<li>Trajectories: {n_traj}, dt = {dt}, t_end = {t_end}, total samples = {n_total}</li>
<li>Train/test split: {n_train} / {n_test} samples (80/20 over pooled samples)</li>
<li>Library: linear (theta = u), recovered K = -Xi<sup>T</sup>, symmetrised for physical diagnostics</li>
<li>Regression locality assumption: 8-connected neighbours (H + V + both diagonals) + self.
    The data generator only places springs on H, V, and the NW-SE diagonal,
    so the regressor is told nothing about the anti-diagonal being zero --
    it should discover that from the data.</li>
<li>DOF reduction: {n_active} / {n_full} entries ({pct_active:.1f}%) of Xi
    are free parameters; the rest are clamped to zero by mask.</li>
</ul>

<h2>Summary table</h2>
{summary_table}

<h2>Train / test loss curves</h2>
<img src="data:image/png;base64,{loss_curve_b64}" />

<h2>Relative K error vs epoch</h2>
<img src="data:image/png;base64,{k_err_b64}" />

<h2>Recovered K matrices (final)</h2>
<img src="data:image/png;base64,{k_matrix_b64}" />

<h2>Eigenvalue trajectories</h2>
{eigenvalue_blocks}

<h2>Presentation diagnostics</h2>
<div class="note-box">
<p>Detailed rollout / local-parameter / modal diagnostics are generated for
the selected presentation models only:
{presentation_model_summary}</p>
<p>This codebase's spring-grid model is scalar-valued: each node carries one
out-of-plane displacement. The requested "quiver" visualisation is therefore
adapted as a vertical arrow rendering of scalar deflection on the lattice.</p>
{objective_note}
</div>

{presentation_blocks}

<h2>Output locations and assumptions</h2>
{summary_notes}

<h2>Notes</h2>
<p>The "locality assumption" (only neighbour pairs and self-couplings have
non-zero K entries) is encoded by the sparse-regression objective: STLS and
L1/proximal methods drive non-neighbour entries to zero. The dense SINDy
Adam (no L1) baseline is included for comparison and is expected to be more
accurate per coefficient but less sparse. For the new physical diagnostics,
the estimated operator is interpreted through the symmetrised stiffness
matrix <code>K_pred = 0.5 * ((-Xi^T) + (-Xi^T)^T)</code>.</p>
</body>
</html>
"""


def build_summary_table(grad_results: dict, stls_results: dict) -> str:
    rows = []
    rows.append(
        "<tr><th>Method</th><th>Train MSE</th><th>Test MSE</th>"
        "<th>Rel. K error</th><th>Final sparsity (%)</th></tr>"
    )

    for name, res in stls_results.items():
        sparsity = (res["xi"].abs() < 1e-10).float().mean().item() * 100
        rows.append(
            f"<tr><td>{name}</td><td>{res['train_mse']:.3e}</td>"
            f"<td>{res['test_mse']:.3e}</td>"
            f"<td>{res['K_err']:.3e}</td>"
            f"<td>{sparsity:.1f}</td></tr>"
        )

    for name, res in grad_results.items():
        sparsity = (res["xi"].abs() < 1e-10).float().mean().item() * 100
        rows.append(
            f"<tr><td>{name}</td><td>{res['train_losses'][-1]:.3e}</td>"
            f"<td>{res['test_losses'][-1]:.3e}</td>"
            f"<td>{res['snap_K_err'][-1]:.3e}</td>"
            f"<td>{sparsity:.1f}</td></tr>"
        )

    return "<table>" + "".join(rows) + "</table>"


def build_node_summary_list(selected_nodes: OrderedDict[str, int], n: int) -> str:
    items = []
    for label, node in selected_nodes.items():
        items.append(f"<li><code>{label}</code>: {describe_node(node, n)}</li>")
    return "<ul>" + "".join(items) + "</ul>"


def generate_presentation_figures(
    data: dict,
    stls_results: dict,
    grad_results: dict,
    out_path: Path,
    device: torch.device | None = None,
    simulation_backend: str = "auto",
    dtype: torch.dtype = torch.float64,
    focus_model_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, str]:
    output_dirs = build_output_dirs(out_path)
    figure_dir = output_dirs["figure_dir"]
    animation_dir = output_dirs["animation_dir"]
    n = data["n"]
    free = data["free"]
    n_total = data["N"]
    K_true_full = data["K_full"]
    K_true_free = data["K_free"]
    base_traj = data["trajectories"][0]
    selected_nodes = select_representative_nodes(n, free, base_traj["U_full"])
    incident_specs = build_incident_specs(selected_nodes, n)

    figure1_assets = plot_lattice_stiffness(
        K_true_full,
        n,
        figure_dir / "figure_1_true_stiffness_lattice",
    )

    blocks = [
        "<div class=\"method-block\">",
        "<h3>Figure 1: lattice schematic</h3>",
        render_image_tag(out_path.parent, figure1_assets["png"], "True stiffness lattice"),
        "</div>",
    ]

    model_names = tuple(focus_model_names or FOCUS_MODEL_NAMES)

    for model_name in model_names:
        if model_name in stls_results:
            K_pred_free = stls_results[model_name]["K_pred"].detach().cpu().numpy()
            K_pred_full = build_full_K_from_free(K_pred_free, free, n_total)
            pseudo_epochs = np.array([0.0, 1.0])
            pseudo_history = {}
            for spec in incident_specs:
                val = float(-K_pred_full[spec["node"], spec["neighbor"]])
                pseudo_history[spec["series_key"]] = np.array([val, val])
            history_epochs = pseudo_epochs
            history_values = pseudo_history
            param_note = "STLS is a direct sparse solve, so this panel shows the final estimate as a constant pseudo-history."
        else:
            res = grad_results[model_name]
            K_pred_free = xi_to_K(res["xi"]).detach().cpu().numpy()
            history_epochs = res["parameter_history_epochs"]
            history_values = res["parameter_history"]
            param_note = None

        pred_rollout = rollout_linear_model(
            K_pred_free,
            free,
            n,
            dt=data["dt"],
            t_end=data["t_end"],
            u0_full=base_traj["u0_full"],
            v0_full=base_traj["v0_full"],
            device=device,
            simulation_backend=simulation_backend,
            dtype=dtype,
        )

        model_slug = slugify(model_name)
        quiver_assets = make_quiver_snapshot_figure(
            base_traj,
            pred_rollout,
            n,
            model_name,
            figure_dir / f"figure_2A_quiver_snapshots_true_vs_regressed_{model_slug}",
        )
        quiver_vmax = compute_visual_max(base_traj["U_full"], pred_rollout["U_full"])
        quiver_vis_scale = 0.75 / quiver_vmax
        anim_true_quiver = make_quiver_animation(
            base_traj,
            n,
            f"True scalar-deflection quiver, {model_name}",
            animation_dir / f"animation_2B_true_quiver_{model_slug}",
            vis_scale=quiver_vis_scale,
        )
        anim_pred_quiver = make_quiver_animation(
            pred_rollout,
            n,
            f"Regressed scalar-deflection quiver, {model_name}",
            animation_dir / f"animation_2B_regressed_quiver_{model_slug}",
            vis_scale=quiver_vis_scale,
        )

        heatmap_assets = make_heatmap_snapshot_figure(
            base_traj,
            pred_rollout,
            n,
            model_name,
            figure_dir / f"figure_2C_heatmap_snapshots_true_vs_regressed_{model_slug}",
        )
        heatmap_vmax = compute_visual_max(np.abs(base_traj["U_full"]), np.abs(pred_rollout["U_full"]))
        anim_true_heat = make_heatmap_animation(
            base_traj,
            n,
            f"True displacement heatmap, {model_name}",
            animation_dir / f"animation_2D_true_heatmap_{model_slug}",
            vmax=heatmap_vmax,
        )
        anim_pred_heat = make_heatmap_animation(
            pred_rollout,
            n,
            f"Regressed displacement heatmap, {model_name}",
            animation_dir / f"animation_2D_regressed_heatmap_{model_slug}",
            vmax=heatmap_vmax,
        )

        traces_assets = make_node_trace_plots(
            base_traj,
            pred_rollout,
            selected_nodes,
            n,
            model_name,
            figure_dir / f"figure_3_node_traces_{model_slug}",
        )
        param_assets = make_parameter_history_plots(
            model_name,
            selected_nodes,
            incident_specs,
            K_true_full,
            history_values,
            history_epochs,
            n,
            figure_dir / f"figure_5_parameter_history_{model_slug}",
            note=param_note,
        )
        modal_assets = make_modal_comparison(
            model_name,
            K_true_free,
            K_pred_free,
            figure_dir / f"figure_9_modes_true_vs_regressed_{model_slug}",
        )
        mode_shape_assets = make_mode_shape_examples(
            model_name,
            K_true_free,
            K_pred_free,
            n,
            free,
            figure_dir / f"figure_9_mode_shapes_example_{model_slug}",
        )

        blocks.extend([
            "<div class=\"method-block\">",
            f"<h3>{model_name}</h3>",
            "<h4>Figure 2A: static quiver snapshots</h4>",
            render_image_tag(out_path.parent, quiver_assets["png"], f"Quiver snapshots for {model_name}"),
            "<h4>Figure 2B: quiver animations</h4>",
            "<div class=\"media-grid\">",
            render_video_tag(out_path.parent, anim_true_quiver, "True / data quiver"),
            render_video_tag(out_path.parent, anim_pred_quiver, "Regressed / model quiver"),
            "</div>",
            "<h4>Figure 2C: static heatmap snapshots</h4>",
            render_image_tag(out_path.parent, heatmap_assets["png"], f"Heatmap snapshots for {model_name}"),
            "<h4>Figure 2D: heatmap animations</h4>",
            "<div class=\"media-grid\">",
            render_video_tag(out_path.parent, anim_true_heat, "True / data heatmap"),
            render_video_tag(out_path.parent, anim_pred_heat, "Regressed / model heatmap"),
            "</div>",
            "<h4>Figure 3: selected node traces</h4>",
            render_image_tag(out_path.parent, traces_assets["png"], f"Node traces for {model_name}"),
            "<h4>Figure 5: local stiffness evolution</h4>",
            render_image_tag(out_path.parent, param_assets["png"], f"Parameter history for {model_name}"),
            "<h4>Figure 9: modal comparison</h4>",
            render_image_tag(out_path.parent, modal_assets["png"], f"Modal comparison for {model_name}"),
            "<h5>Representative mode shapes</h5>",
            render_image_tag(out_path.parent, mode_shape_assets["png"], f"Mode shapes for {model_name}"),
            "</div>",
        ])

    notes = [
        "<ul>",
        "<li>Static presentation figures saved under "
        f"<code>{figure_dir}</code>.</li>",
        "<li>Animations saved under "
        f"<code>{animation_dir}</code>.</li>",
        "<li>Trajectory diagnostics use the first simulated rollout from "
        f"<code>generate_dataset(...)</code> and compare it against a rollout of the symmetrised estimated stiffness model from the same initial condition.</li>",
        "<li>Node selection used for Figure 3 / Figure 5:"
        f"{build_node_summary_list(selected_nodes, n)}</li>",
        "<li>Figure 5 tracks local 8-neighbour couplings around each selected node. "
        "The true system only contains H / V / NW-SE diagonal springs, so the anti-diagonal true reference is zero where applicable.</li>",
        "<li>Figure 9 uses modal comparison rather than a dispersion relation because the current codebase uses a finite, heterogeneous lattice rather than a periodic homogeneous medium.</li>",
        "</ul>",
    ]

    return "\n".join(blocks), "\n".join(notes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_workflow(
    out_path: str | Path | None = "figures/spring_grid_K_regression_report.html",
    n_epochs: int = 600,
    *,
    device_arg: str = "auto",
    simulation_backend: str = "auto",
    execution_mode: str = "auto",
    build_report: bool = True,
    run_rollout_benchmark: bool = True,
) -> dict:
    t0 = time.perf_counter()
    dtype = torch.float64
    execution_plan = resolve_execution_plan(device_arg, execution_mode)
    dataset_device = execution_plan["dataset_device"]
    stls_device = execution_plan["stls_device"]
    training_device = execution_plan["training_device"]
    rollout_device = execution_plan["rollout_device"]
    report_device = execution_plan["report_device"]
    dataset_backend = resolve_simulation_backend(simulation_backend, dataset_device)
    rollout_backend = resolve_simulation_backend(simulation_backend, rollout_device)
    timings: dict[str, float | dict[str, float]] = {}

    print(f"Execution mode: {execution_plan['mode']}")
    print(
        "Stage devices: "
        f"dataset={dataset_device}, stls={stls_device}, "
        f"training={training_device}, rollout={rollout_device}, report={report_device}"
    )
    print(
        "Simulation backends: "
        f"dataset={dataset_backend}, rollout={rollout_backend}"
    )

    sync_device(dataset_device)
    phase_t0 = time.perf_counter()
    print("Generating spring-grid dataset...")
    data = generate_dataset(
        n=10,
        n_traj=4,
        t_end=20.0,
        dt=0.01,
        seed=0,
        device=dataset_device,
        simulation_backend=dataset_backend,
        dtype=dtype,
    )
    sync_device(dataset_device)
    timings["dataset_generation_seconds"] = time.perf_counter() - phase_t0
    print(f"  free DOFs = {data['free'].size}, total samples = {data['U'].shape[0]}")

    n_active = int(data["locality_free"].sum())
    n_full = data["U"].shape[1] * data["U"].shape[1]
    print(
        f"Locality DOF reduction: {n_active} / {n_full} active entries "
        f"({100 * n_active / n_full:.1f}%)"
    )

    selected_nodes = select_representative_nodes(
        data["n"], data["free"], data["trajectories"][0]["U_full"]
    )
    incident_specs = build_incident_specs(selected_nodes, data["n"])

    sync_device(stls_device)
    phase_t0 = time.perf_counter()
    print("Running STLS variants (locality-constrained)...")
    stls_problem = build_regression_problem(data, device=stls_device, dtype=dtype)
    stls_results = {
        "STLS local (lam=0.01)": run_stls(
            stls_problem["theta_train"],
            stls_problem["target_train"],
            stls_problem["theta_test"],
            stls_problem["target_test"],
            stls_problem["K_true"],
            lam=0.01,
            mask=stls_problem["xi_mask"],
        ),
        "STLS local (lam=0.05)": run_stls(
            stls_problem["theta_train"],
            stls_problem["target_train"],
            stls_problem["theta_test"],
            stls_problem["target_test"],
            stls_problem["K_true"],
            lam=0.05,
            mask=stls_problem["xi_mask"],
        ),
        "STLS dense (lam=0.05)": run_stls(
            stls_problem["theta_train"],
            stls_problem["target_train"],
            stls_problem["theta_test"],
            stls_problem["target_test"],
            stls_problem["K_true"],
            lam=0.05,
            mask=None,
        ),
    }
    sync_device(stls_device)
    timings["stls_seconds"] = time.perf_counter() - phase_t0
    for name, res in stls_results.items():
        print(
            f"  {name}: train={res['train_mse']:.3e} "
            f"test={res['test_mse']:.3e} K_err={res['K_err']:.3e}"
        )

    grad_methods = {
        "SINDy Adam local (no L1)":
            dict(l1_lambda=0.0, proximal=False, lr=5e-2, mask="local"),
        "SINDy Adam local (L1=1e-3)":
            dict(l1_lambda=1e-3, proximal=False, lr=5e-2, mask="local"),
        "SINDy ISTA local (L1=1e-3)":
            dict(l1_lambda=1e-3, proximal=True, lr=5e-2, mask="local"),
        "SINDy Adam dense (no L1)":
            dict(l1_lambda=0.0, proximal=False, lr=5e-2, mask=None),
    }

    grad_results = {}
    grad_timings = {}
    grad_phase_total = 0.0
    sync_device(training_device)
    phase_t0 = time.perf_counter()
    grad_problem = build_regression_problem(data, device=training_device, dtype=dtype)
    sync_device(training_device)
    timings["tensor_setup_seconds"] = time.perf_counter() - phase_t0
    for name, kwargs in grad_methods.items():
        print(f"Training {name}...")
        method_kwargs = dict(kwargs)
        method_mask = method_kwargs.pop("mask")
        history_specs = incident_specs if name in FOCUS_MODEL_NAMES else None
        if method_mask == "local":
            method_mask = grad_problem["xi_mask"]
        sync_device(training_device)
        method_t0 = time.perf_counter()
        res = train_gradient_method(
            grad_problem["theta_train"],
            grad_problem["target_train"],
            grad_problem["theta_test"],
            grad_problem["target_test"],
            grad_problem["K_true"],
            n_features=grad_problem["n_features"],
            n_states=grad_problem["n_states"],
            n_epochs=n_epochs, snapshot_every=max(1, n_epochs // 60),
            n_eig=5, seed=0, xi_mask=method_mask,
            history_specs=history_specs,
            history_every=1,
            free=data["free"],
            n_total_dofs=data["N"],
            **method_kwargs,
        )
        sync_device(training_device)
        method_seconds = time.perf_counter() - method_t0
        grad_phase_total += method_seconds
        grad_timings[name] = method_seconds
        grad_results[name] = res
        print(
            f"  final train={res['train_losses'][-1]:.3e} "
            f"test={res['test_losses'][-1]:.3e} "
            f"K_err={res['snap_K_err'][-1]:.3e} "
            f"time={method_seconds:.1f}s"
        )
    timings["gradient_training_seconds"] = grad_phase_total
    timings["gradient_method_seconds"] = grad_timings

    rollout_metrics = {}
    if run_rollout_benchmark:
        sync_device(rollout_device)
        phase_t0 = time.perf_counter()
        rollout_metrics = benchmark_focus_rollouts(
            data,
            stls_results,
            grad_results,
            device=rollout_device,
            simulation_backend=rollout_backend,
            dtype=dtype,
        )
        sync_device(rollout_device)
        timings["focus_rollout_seconds"] = time.perf_counter() - phase_t0
    else:
        timings["focus_rollout_seconds"] = 0.0

    output_dirs = None
    report_path = None
    K_true_plot = stls_problem["K_true"]
    if build_report and out_path is not None:
        out_path_p = Path(out_path)
        output_dirs = build_output_dirs(out_path_p)

        sync_device(report_device)
        phase_t0 = time.perf_counter()
        print("Building current regression plots...")
        loss_curve_b64 = plot_loss_curves(
            grad_results,
            output_stem=output_dirs["figure_dir"] / "current_loss_curves",
        )
        k_err_b64 = plot_K_error(
            grad_results,
            output_stem=output_dirs["figure_dir"] / "current_relative_k_error",
        )

        K_preds = {name: xi_to_K(res["xi"]) for name, res in stls_results.items()}
        K_preds.update({name: xi_to_K(res["xi"]) for name, res in grad_results.items()})
        k_matrix_b64 = plot_K_matrices(
            K_true_plot,
            K_preds,
            output_stem=output_dirs["figure_dir"] / "current_recovered_k_matrices",
        )

        eig_blocks = []
        for name, res in grad_results.items():
            b64 = plot_eigenvalues(
                name,
                res,
                output_stem=output_dirs["figure_dir"] / f"current_eigenvalues_{slugify(name)}",
            )
            eig_blocks.append(
                f'<div class="method-block"><h3>{name}</h3>'
                f'<img src="data:image/png;base64,{b64}" /></div>'
            )

        summary_table = build_summary_table(grad_results, stls_results)

        print("Generating presentation diagnostics...")
        presentation_blocks, summary_notes = generate_presentation_figures(
            data,
            stls_results,
            grad_results,
            out_path_p,
            device=rollout_device,
            simulation_backend=rollout_backend,
            dtype=dtype,
        )

        html = HTML_TEMPLATE.format(
            report_title="Spring-Grid K Regression Report",
            n=data["n"],
            n_dofs=data["free"].size,
            n_traj=data["n_traj"],
            dt=data["dt"],
            t_end=data["t_end"],
            spring_description="horizontal, vertical, and diagonal pairs with constants ~ Uniform[0.5, 1.5]",
            n_total=grad_problem["n_total"],
            n_train=grad_problem["n_train"],
            n_test=grad_problem["n_total"] - grad_problem["n_train"],
            n_active=n_active,
            n_full=n_full,
            pct_active=100 * n_active / n_full,
            summary_table=summary_table,
            loss_curve_b64=loss_curve_b64,
            k_err_b64=k_err_b64,
            k_matrix_b64=k_matrix_b64,
            eigenvalue_blocks="\n".join(eig_blocks),
            presentation_model_summary=", ".join(
                f"<code>{name}</code>" for name in FOCUS_MODEL_NAMES[:-1]
            ) + f", and <code>{FOCUS_MODEL_NAMES[-1]}</code>",
            objective_note="",
            presentation_blocks=presentation_blocks,
            summary_notes=summary_notes,
        )

        out_path_p.parent.mkdir(parents=True, exist_ok=True)
        out_path_p.write_text(html, encoding="utf-8")
        sync_device(report_device)
        timings["report_build_seconds"] = time.perf_counter() - phase_t0
        report_path = str(out_path_p.resolve())
        print(f"\nWrote report to: {out_path_p.resolve()}")
        print(f"Static figure assets: {output_dirs['figure_dir']}")
        print(f"Animation assets:    {output_dirs['animation_dir']}")
    else:
        timings["report_build_seconds"] = 0.0

    sync_device(training_device)
    total_seconds = time.perf_counter() - t0
    timings["total_seconds"] = total_seconds
    print(f"Total wall time: {total_seconds:.1f}s")

    return {
        "device": str(training_device),
        "execution_mode": execution_plan["mode"],
        "stage_devices": {
            "dataset": str(dataset_device),
            "stls": str(stls_device),
            "training": str(training_device),
            "rollout": str(rollout_device),
            "report": str(report_device),
        },
        "simulation_backend": {
            "dataset": dataset_backend,
            "rollout": rollout_backend,
        },
        "n_epochs": int(n_epochs),
        "data": data,
        "stls_results": stls_results,
        "grad_results": grad_results,
        "method_metrics": summarize_method_metrics(stls_results, grad_results),
        "rollout_metrics": rollout_metrics,
        "timings": timings,
        "report_path": report_path,
        "output_dirs": output_dirs,
    }


def main(
    out_path: str,
    n_epochs: int,
    *,
    device_arg: str = "auto",
    simulation_backend: str = "auto",
    execution_mode: str = "auto",
) -> dict:
    return run_workflow(
        out_path=out_path,
        n_epochs=n_epochs,
        device_arg=device_arg,
        simulation_backend=simulation_backend,
        execution_mode=execution_mode,
        build_report=True,
        run_rollout_benchmark=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="figures/spring_grid_K_regression_report.html",
        help="Path to write the HTML report",
    )
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument(
        "--simulation-backend",
        choices=("auto", "scipy", "torch"),
        default="auto",
        help="Spring-grid rollout backend. Defaults to torch on CUDA and scipy on CPU.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("auto", "single", "hybrid"),
        default="auto",
        help=(
            "Workflow device strategy. "
            "'single' uses one device for all stages, "
            "'hybrid' uses CPU for simulation/STLS/rollouts and CUDA for gradient training, "
            "and 'auto' enables hybrid only when --device auto and CUDA is available."
        ),
    )
    sindy_torch.add_device_arg(parser)
    args = parser.parse_args()
    main(
        args.out,
        args.epochs,
        device_arg=args.device,
        simulation_backend=args.simulation_backend,
        execution_mode=args.execution_mode,
    )
