"""Frequency response analysis for linear second-order systems.

Computes the steady-state response of `M u_tt + C u_t + K u = F0 cos(omega t)`
across a sweep of forcing frequencies, by directly solving
`(K - omega**2 M + i omega C) U = F0` per frequency.

Designed for the spring-grid SINDy regression pipeline, but the math is
agnostic: any dense or sparse matrices of compatible shapes will work.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def bottom_row_uniform_force(
    n: int,
    free: np.ndarray,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Build a real force vector on free DOFs.

    Bottom row is row index ``i = 0`` with flat indexing ``i*n + j``, so the
    bottom-row full-grid DOFs are ``0, 1, ..., n-1``. The result is masked to
    the free-DOF ordering supplied in ``free``.
    """
    full = np.zeros(n * n, dtype=float)
    full[:n] = float(amplitude)
    return full[np.asarray(free, dtype=int)]


def compute_frequency_response(
    K: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
    F0: np.ndarray,
    omegas: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the complex steady-state displacement at each forcing frequency.

    Solves `(K - omega**2 M + i omega C) U = F0` for each omega.

    Returns an array of shape ``(len(omegas), n_dof)`` with complex entries.
    The amplitude at DOF j is ``abs(U[k, j])`` and the phase is ``angle(U[k, j])``
    relative to the forcing ``F0 cos(omega t)``.
    """
    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)
    C = np.asarray(C, dtype=float)
    F0_c = np.asarray(F0, dtype=float).astype(complex)
    omegas = np.asarray(omegas, dtype=float)
    n_dof = K.shape[0]
    U = np.empty((omegas.size, n_dof), dtype=complex)
    for k, w in enumerate(omegas):
        A = (K - (w * w) * M).astype(complex) + (1j * w) * C
        U[k] = np.linalg.solve(A, F0_c)
    return U


def amplitude_metrics(
    U_omega: np.ndarray,
    *,
    top_row_indices: Iterable[int] | None = None,
) -> dict[str, np.ndarray]:
    """Reduce a complex response array to scalar amplitude metrics per omega."""
    abs_U = np.abs(U_omega)
    metrics = {
        "l2": np.linalg.norm(abs_U, axis=1),
        "max": abs_U.max(axis=1),
        "mean": abs_U.mean(axis=1),
    }
    if top_row_indices is not None:
        idx = np.asarray(list(top_row_indices), dtype=int)
        if idx.size > 0:
            metrics["top_row_mean"] = abs_U[:, idx].mean(axis=1)
    return metrics


def pick_animation_frequencies(
    omegas: np.ndarray,
    amplitude: np.ndarray,
    *,
    boundary_skip_frac: float = 0.05,
) -> dict[str, dict]:
    """Select small, medium, and big response frequencies from a sweep curve.

    Avoids the lowest and highest few percent of the omega range so the
    "small" pick is a real interior anti-resonance rather than a sweep edge.
    Returns a mapping like ``{"small": {"index": k, "omega": w, "amp": a}, ...}``.
    """
    omegas = np.asarray(omegas, dtype=float)
    amp = np.asarray(amplitude, dtype=float)
    n = omegas.size
    skip = max(1, int(round(boundary_skip_frac * n)))
    interior = np.arange(skip, n - skip)
    if interior.size == 0:
        interior = np.arange(n)

    interior_amp = amp[interior]
    big_local = int(np.argmax(interior_amp))
    small_local = int(np.argmin(interior_amp))
    big_idx = int(interior[big_local])
    small_idx = int(interior[small_local])

    sorted_idx = interior[np.argsort(interior_amp)]
    medium_idx = int(sorted_idx[sorted_idx.size // 2])

    return {
        "small": {"index": small_idx, "omega": float(omegas[small_idx]), "amp": float(amp[small_idx])},
        "medium": {"index": medium_idx, "omega": float(omegas[medium_idx]), "amp": float(amp[medium_idx])},
        "big": {"index": big_idx, "omega": float(omegas[big_idx]), "amp": float(amp[big_idx])},
    }


def steady_state_rollout(
    U_complex: np.ndarray,
    omega: float,
    *,
    n_periods: float = 2.0,
    n_frames: int = 120,
    free: np.ndarray | None = None,
    n_total: int | None = None,
) -> dict:
    """Build a rollout-style dict for one steady-state period.

    Given the complex displacement vector ``U(omega)`` (shape ``(n_dof,)``),
    sample ``n_periods`` of the real signal ``Re(U exp(i omega t))`` over
    ``n_frames`` evenly spaced time points. If ``free`` and ``n_total`` are
    provided, the per-frame displacement is expanded onto the full
    ``n_total`` grid (zeros at pinned DOFs) so the result is compatible with
    ``make_quiver_animation`` / ``make_heatmap_animation``.
    """
    period = 2.0 * np.pi / max(float(omega), 1e-12)
    times = np.linspace(0.0, n_periods * period, n_frames)
    phasor = np.exp(1j * omega * times)
    U_t_free = np.real(np.outer(phasor, U_complex))

    if free is not None and n_total is not None:
        free = np.asarray(free, dtype=int)
        U_t_full = np.zeros((n_frames, int(n_total)), dtype=float)
        U_t_full[:, free] = U_t_free
    else:
        U_t_full = U_t_free

    return {
        "times": times,
        "U_full": U_t_full,
        "V_full": np.zeros_like(U_t_full),
        "A_full": np.zeros_like(U_t_full),
        "u0_full": U_t_full[0].copy(),
        "v0_full": np.zeros(U_t_full.shape[1]),
    }
