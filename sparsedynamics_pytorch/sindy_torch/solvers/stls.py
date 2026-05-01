"""
Sequential Thresholded Least Squares (STLS) for sparse regression.

Port of sparsify_dynamics.py using torch.linalg.lstsq.
This is a non-differentiable solver — it computes Xi in a single pass
and is used for quick identification or warm-starting gradient-based methods.
"""

import torch
from torch import Tensor


def stls(
    theta: Tensor,
    dxdt: Tensor,
    lam: float,
    n_iter: int = 10,
    return_history: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """Find sparse coefficient matrix Xi such that dxdt ≈ Theta @ Xi.

    Uses sequential thresholded least squares: solve least-squares,
    threshold small coefficients to zero, re-solve on remaining terms,
    repeat.

    Parameters
    ----------
    theta : Tensor, shape (n_samples, n_features)
        Library matrix.
    dxdt : Tensor, shape (n_samples, n_states)
        Time derivatives of the state.
    lam : float
        Sparsification threshold.
    n_iter : int
        Number of thresholding iterations (default 10).

    return_history : bool
        When True, also return a list of Xi snapshots. Entry 0 is the initial
        least-squares solve before thresholding; entries 1..n_iter are the
        sequential threshold/refit iterates.

    Returns
    -------
    xi : Tensor, shape (n_features, n_states)
        Sparse coefficient matrix (detached, no grad).
    """
    # Initial least-squares solution
    xi = torch.linalg.lstsq(theta, dxdt).solution
    history = [xi.detach().clone()] if return_history else None

    n_states = dxdt.shape[1]

    for _ in range(n_iter):
        small_inds = xi.abs() < lam
        xi = xi.clone()
        xi[small_inds] = 0.0

        for col in range(n_states):
            big_inds = ~small_inds[:, col]
            if big_inds.any():
                xi_col = torch.linalg.lstsq(
                    theta[:, big_inds], dxdt[:, col : col + 1]
                ).solution
                xi[big_inds, col] = xi_col.squeeze(-1)

        if return_history:
            history.append(xi.detach().clone())

    xi_out = xi.detach()
    if return_history:
        return xi_out, history
    return xi_out


def stls_masked(
    theta: Tensor,
    dxdt: Tensor,
    mask: Tensor,
    lam: float = 0.0,
    n_iter: int = 10,
    return_history: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """STLS with a hard support mask (locality / structural prior).

    The recovered Xi is constrained to be zero everywhere `mask` is False;
    only the True entries are estimated by least squares, with optional
    sequential thresholding at level `lam`.

    Parameters
    ----------
    theta : Tensor, shape (n_samples, n_features)
    dxdt : Tensor, shape (n_samples, n_states)
    mask : Tensor (bool), shape (n_features, n_states)
        Allowed support pattern. Per-column least-squares fits use only the
        rows where `mask[:, col]` is True.
    lam : float
        Sparsification threshold (set to 0 to keep every allowed entry).
    n_iter : int
        Refit iterations.
    return_history : bool
        When True, also return a list of Xi snapshots. Entry 0 is the initial
        masked least-squares solve before thresholding; entries 1..n_iter are
        the sequential threshold/refit iterates.
    """
    n_features, n_states = mask.shape
    xi = torch.zeros(n_features, n_states, dtype=theta.dtype, device=theta.device)

    for col in range(n_states):
        support = mask[:, col]
        if support.any():
            sol = torch.linalg.lstsq(
                theta[:, support], dxdt[:, col : col + 1]
            ).solution
            xi[support, col] = sol.squeeze(-1)
    history = [xi.detach().clone()] if return_history else None

    for _ in range(n_iter):
        keep = (xi.abs() >= lam) & mask
        xi = torch.zeros_like(xi)
        for col in range(n_states):
            support = keep[:, col]
            if support.any():
                sol = torch.linalg.lstsq(
                    theta[:, support], dxdt[:, col : col + 1]
                ).solution
                xi[support, col] = sol.squeeze(-1)
        if return_history:
            history.append(xi.detach().clone())

    xi_out = xi.detach()
    if return_history:
        return xi_out, history
    return xi_out
