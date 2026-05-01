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


@dataclass(frozen=True)
class BlockAverageReduction:
    """Block-averaging reduction operators for a scalar spring grid.

    The restriction maps fine-grid free DOFs to coarse block averages, while
    the prolongation maps coarse block values back to a block-constant fine
    representation.
    """

    fine_grid_size: int
    block_size: int
    coarse_grid_size: int
    free_dofs: np.ndarray
    active_counts: np.ndarray
    block_index_by_free_dof: np.ndarray
    restriction: np.ndarray  # (N_coarse, n_free)
    prolongation: np.ndarray  # (n_free, N_coarse)


def build_block_average_reduction(
    n: int,
    block_size: int,
    *,
    free_dofs: Optional[Sequence[int]] = None,
) -> BlockAverageReduction:
    """Build block-averaging reduction operators for an ``n x n`` grid.

    Parameters
    ----------
    n : int
        Fine-grid side length.
    block_size : int
        Side length of each square averaging block. Must divide ``n``.
    free_dofs : sequence of int, optional
        Fine-grid DOFs retained in the state vector. When omitted, all
        ``n * n`` DOFs are treated as active. When provided, the returned
        operators are sized for that free-DOF ordering.

    Returns
    -------
    BlockAverageReduction
        Restriction/prolongation operators and block metadata.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if n % block_size != 0:
        raise ValueError(
            f"grid size n={n} must be an integer multiple of block_size={block_size}"
        )

    N = n * n
    if free_dofs is None:
        free = np.arange(N, dtype=int)
    else:
        free = np.asarray(free_dofs, dtype=int)
        if free.ndim != 1:
            raise ValueError("free_dofs must be a 1D sequence of fine-grid indices")
        if free.size == 0:
            raise ValueError("free_dofs must contain at least one DOF")
        if np.any((free < 0) | (free >= N)):
            raise ValueError(
                f"free_dofs entries must lie in [0, {N - 1}] for an {n}x{n} grid"
            )
        if np.unique(free).size != free.size:
            raise ValueError("free_dofs must not contain duplicates")

    coarse_n = n // block_size
    coarse_N = coarse_n * coarse_n
    rows = free // n
    cols = free % n
    block_rows = rows // block_size
    block_cols = cols // block_size
    block_index = block_rows * coarse_n + block_cols
    active_counts = np.bincount(block_index, minlength=coarse_N)
    if np.any(active_counts == 0):
        empty = np.flatnonzero(active_counts == 0)
        raise ValueError(
            "Each coarse block must contain at least one active fine DOF; "
            f"empty coarse block indices: {empty.tolist()}"
        )

    restriction = np.zeros((coarse_N, free.size), dtype=float)
    restriction[block_index, np.arange(free.size)] = 1.0 / active_counts[block_index]

    prolongation = np.zeros((free.size, coarse_N), dtype=float)
    prolongation[np.arange(free.size), block_index] = 1.0

    return BlockAverageReduction(
        fine_grid_size=int(n),
        block_size=int(block_size),
        coarse_grid_size=int(coarse_n),
        free_dofs=free.copy(),
        active_counts=active_counts.astype(int, copy=False),
        block_index_by_free_dof=block_index.astype(int, copy=False),
        restriction=restriction,
        prolongation=prolongation,
    )


def apply_block_average_reduction(
    values: np.ndarray | Tensor,
    reduction: BlockAverageReduction,
) -> np.ndarray | Tensor:
    """Apply a block-averaging restriction to vectors or time series.

    ``values`` must use the same free-DOF ordering as ``reduction.free_dofs``.
    Accepted shapes are ``(n_free,)`` and ``(..., n_free)``.
    """
    if isinstance(values, Tensor):
        R_t = torch.as_tensor(
            reduction.restriction,
            device=values.device,
            dtype=values.dtype,
        )
        if values.shape[-1] != R_t.shape[1]:
            raise ValueError(
                f"Expected last dimension {R_t.shape[1]}, got {values.shape[-1]}"
            )
        if values.ndim == 1:
            return R_t @ values
        return values @ R_t.T

    arr = np.asarray(values, dtype=float)
    if arr.shape[-1] != reduction.restriction.shape[1]:
        raise ValueError(
            f"Expected last dimension {reduction.restriction.shape[1]}, got {arr.shape[-1]}"
        )
    if arr.ndim == 1:
        return reduction.restriction @ arr
    return arr @ reduction.restriction.T


def project_block_average_operator(
    operator: np.ndarray | Tensor,
    reduction: BlockAverageReduction,
) -> np.ndarray | Tensor:
    """Project a fine-grid free-DOF operator into the block-averaged space."""
    if isinstance(operator, Tensor):
        R_t = torch.as_tensor(
            reduction.restriction,
            device=operator.device,
            dtype=operator.dtype,
        )
        P_t = torch.as_tensor(
            reduction.prolongation,
            device=operator.device,
            dtype=operator.dtype,
        )
        if operator.shape != (P_t.shape[0], P_t.shape[0]):
            raise ValueError(
                "operator must have shape "
                f"({P_t.shape[0]}, {P_t.shape[0]}), got {tuple(operator.shape)}"
            )
        return R_t @ operator @ P_t

    arr = np.asarray(operator, dtype=float)
    n_free = reduction.prolongation.shape[0]
    if arr.shape != (n_free, n_free):
        raise ValueError(
            f"operator must have shape ({n_free}, {n_free}), got {arr.shape}"
        )
    return reduction.restriction @ arr @ reduction.prolongation


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
