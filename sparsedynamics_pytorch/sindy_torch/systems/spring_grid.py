"""
Scalar-displacement spring-grid ODE.

Port of the lattice assembly from data_generation/spring_grid_dynamic.py.
Each node carries a single scalar displacement u_i (out-of-plane drum
deflection); the governing equation is

    M u_tt + C u_t + K u = f(t)

with K assembled from horizontal, vertical, and diagonal springs on an
n x n grid. This is intentionally simpler than systems/spring_lattice.py
(which has 2D node positions and nonlinear spring forces); this module
exists for the locality-aware linear K regression task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve
import torch
from torch import Tensor


def node_index(i: int, j: int, n: int) -> int:
    return i * n + j


def build_stiffness_matrix(
    n: int,
    seed: Optional[int] = None,
    k_low: float = 0.5,
    k_high: float = 1.5,
) -> sp.csr_matrix:
    """Assemble K for an n x n grid with springs on H, V, and diagonal pairs.

    Spring constants are drawn uniformly from [k_low, k_high].
    """
    rng = np.random.default_rng(seed)
    N = n * n
    rows, cols, vals = [], [], []

    def add_pairs(a, b, k):
        rows.extend([a, b, a, b])
        cols.extend([a, b, b, a])
        vals.extend([k, k, -k, -k])

    i_h, j_h = np.meshgrid(np.arange(n), np.arange(n - 1), indexing="ij")
    a = (i_h * n + j_h).ravel()
    b = (i_h * n + j_h + 1).ravel()
    add_pairs(a, b, k_low + (k_high - k_low) * rng.random(a.size))

    i_v, j_v = np.meshgrid(np.arange(n - 1), np.arange(n), indexing="ij")
    a = (i_v * n + j_v).ravel()
    b = ((i_v + 1) * n + j_v).ravel()
    add_pairs(a, b, k_low + (k_high - k_low) * rng.random(a.size))

    i_d, j_d = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = (i_d * n + j_d).ravel()
    b = ((i_d + 1) * n + j_d + 1).ravel()
    add_pairs(a, b, k_low + (k_high - k_low) * rng.random(a.size))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    K.sum_duplicates()
    return K


def build_locality_mask(
    n: int,
    include_anti_diagonal: bool = True,
) -> np.ndarray:
    """Boolean (N, N) mask: True where a K_ij entry is allowed under the
    locality assumption.

    By default this is the full 8-connected stencil (horizontal, vertical,
    both diagonals) plus the self entry on the diagonal. Use
    `include_anti_diagonal=False` for the strict 7-stencil that mirrors the
    `build_stiffness_matrix` data-generation pattern.

    The 8-connected variant is the right choice for *regression*: physically
    nothing forbids an anti-diagonal spring, and we want the regressor to
    recover (near-)zero coefficients there from the data rather than baking
    that prior in.
    """
    N = n * n
    mask = np.eye(N, dtype=bool)

    def mark(a, b):
        mask[a, b] = True
        mask[b, a] = True

    i_h, j_h = np.meshgrid(np.arange(n), np.arange(n - 1), indexing="ij")
    mark((i_h * n + j_h).ravel(), (i_h * n + j_h + 1).ravel())

    i_v, j_v = np.meshgrid(np.arange(n - 1), np.arange(n), indexing="ij")
    mark((i_v * n + j_v).ravel(), ((i_v + 1) * n + j_v).ravel())

    # Main diagonal (NW-SE): (i, j) -- (i+1, j+1)
    i_d, j_d = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    mark((i_d * n + j_d).ravel(), ((i_d + 1) * n + j_d + 1).ravel())

    if include_anti_diagonal:
        # Anti-diagonal (NE-SW): (i, j+1) -- (i+1, j)
        i_a, j_a = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
        mark((i_a * n + j_a + 1).ravel(), ((i_a + 1) * n + j_a).ravel())

    return mask


def get_free_dofs(n: int, pinned_rows: Sequence[int] = (0,)) -> Tuple[np.ndarray, np.ndarray]:
    pinned = np.array(
        [node_index(i, j, n) for i in pinned_rows for j in range(n)], dtype=int
    )
    free = np.setdiff1d(np.arange(n * n), pinned)
    return pinned, free


@dataclass
class SimulationResult:
    times: np.ndarray | Tensor
    displacements: np.ndarray | Tensor  # (n_steps + 1, N)
    velocities: np.ndarray | Tensor
    accelerations: np.ndarray | Tensor
    n: int


def _to_dense_tensor(
    x: sp.spmatrix | np.ndarray | Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    if sp.issparse(x):
        x = x.toarray()
    return torch.as_tensor(x, device=device, dtype=dtype)


def newmark_beta_simulation(
    K: sp.spmatrix,
    M: sp.spmatrix,
    C: sp.spmatrix,
    force_fn: Callable[[float], np.ndarray],
    free: np.ndarray,
    n: int,
    t_end: float = 20.0,
    dt: float = 0.01,
    gamma: float = 0.5,
    beta: float = 0.25,
    u0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
) -> SimulationResult:
    N = K.shape[0]
    n_steps = int(np.round(t_end / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)

    Kf = K[free, :][:, free].tocsc()
    Mf = M[free, :][:, free].tocsc()
    Cf = C[free, :][:, free].tocsc()

    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = 0.5 * dt * (gamma / beta - 2.0)
    a6 = dt * (1.0 - gamma)
    a7 = dt * gamma

    Keff = (Kf + a0 * Mf + a1 * Cf).tocsc()
    lu = splu(Keff)

    U = np.zeros((n_steps + 1, N))
    V = np.zeros_like(U)
    A = np.zeros_like(U)
    if u0 is not None:
        U[0] = u0
    if v0 is not None:
        V[0] = v0

    rhs0 = force_fn(0.0)[free] - Cf @ V[0, free] - Kf @ U[0, free]
    A[0, free] = spsolve(Mf, rhs0)

    for k in range(n_steps):
        u_n = U[k, free]
        v_n = V[k, free]
        a_n = A[k, free]
        rhs = (
            force_fn(times[k + 1])[free]
            + Mf @ (a0 * u_n + a2 * v_n + a3 * a_n)
            + Cf @ (a1 * u_n + a4 * v_n + a5 * a_n)
        )
        u_next = lu.solve(rhs)
        a_next = a0 * (u_next - u_n) - a2 * v_n - a3 * a_n
        v_next = v_n + a6 * a_n + a7 * a_next
        U[k + 1, free] = u_next
        V[k + 1, free] = v_next
        A[k + 1, free] = a_next

    return SimulationResult(times, U, V, A, n)


def newmark_beta_simulation_torch(
    K: sp.spmatrix | np.ndarray | Tensor,
    M: sp.spmatrix | np.ndarray | Tensor,
    C: sp.spmatrix | np.ndarray | Tensor,
    force_fn: Callable[[float], np.ndarray | Tensor],
    free: np.ndarray | Tensor,
    n: int,
    t_end: float = 20.0,
    dt: float = 0.01,
    gamma: float = 0.5,
    beta: float = 0.25,
    u0: Optional[np.ndarray | Tensor] = None,
    v0: Optional[np.ndarray | Tensor] = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> SimulationResult:
    if device is None:
        if isinstance(K, Tensor):
            device = K.device
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    K_t = _to_dense_tensor(K, device=device, dtype=dtype)
    M_t = _to_dense_tensor(M, device=device, dtype=dtype)
    C_t = _to_dense_tensor(C, device=device, dtype=dtype)
    free_t = torch.as_tensor(free, device=device, dtype=torch.long)

    N = K_t.shape[0]
    n_steps = int(np.round(t_end / dt))
    times = torch.linspace(0.0, n_steps * dt, n_steps + 1, device=device, dtype=dtype)

    Kf = K_t.index_select(0, free_t).index_select(1, free_t)
    Mf = M_t.index_select(0, free_t).index_select(1, free_t)
    Cf = C_t.index_select(0, free_t).index_select(1, free_t)

    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = 0.5 * dt * (gamma / beta - 2.0)
    a6 = dt * (1.0 - gamma)
    a7 = dt * gamma

    Keff = Kf + a0 * Mf + a1 * Cf
    try:
        chol = torch.linalg.cholesky(Keff)
    except RuntimeError:
        chol = None

    def solve_linear(system_matrix: Tensor, rhs: Tensor) -> Tensor:
        if chol is not None and system_matrix.data_ptr() == Keff.data_ptr():
            return torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
        return torch.linalg.solve(system_matrix, rhs.unsqueeze(-1)).squeeze(-1)

    def force_vec(t_value: float) -> Tensor:
        values = force_fn(t_value)
        return _to_dense_tensor(values, device=device, dtype=dtype).reshape(N)

    U = torch.zeros((n_steps + 1, N), device=device, dtype=dtype)
    V = torch.zeros_like(U)
    A = torch.zeros_like(U)
    if u0 is not None:
        U[0] = _to_dense_tensor(u0, device=device, dtype=dtype).reshape(N)
    if v0 is not None:
        V[0] = _to_dense_tensor(v0, device=device, dtype=dtype).reshape(N)

    rhs0 = force_vec(0.0).index_select(0, free_t) - Cf @ V[0, free_t] - Kf @ U[0, free_t]
    A[0, free_t] = torch.linalg.solve(Mf, rhs0.unsqueeze(-1)).squeeze(-1)

    for k in range(n_steps):
        u_n = U[k, free_t]
        v_n = V[k, free_t]
        a_n = A[k, free_t]
        rhs = (
            force_vec(float(times[k + 1].item())).index_select(0, free_t)
            + Mf @ (a0 * u_n + a2 * v_n + a3 * a_n)
            + Cf @ (a1 * u_n + a4 * v_n + a5 * a_n)
        )
        u_next = solve_linear(Keff, rhs)
        a_next = a0 * (u_next - u_n) - a2 * v_n - a3 * a_n
        v_next = v_n + a6 * a_n + a7 * a_next
        U[k + 1, free_t] = u_next
        V[k + 1, free_t] = v_next
        A[k + 1, free_t] = a_next

    return SimulationResult(times, U, V, A, n)


def zero_force(N: int) -> Callable[[float], np.ndarray]:
    z = np.zeros(N)

    def f(t: float) -> np.ndarray:
        return z

    return f


def zero_force_torch(
    N: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Callable[[float], Tensor]:
    z = torch.zeros(N, device=device, dtype=dtype)

    def f(t: float) -> Tensor:
        return z

    return f
