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
) -> Tensor:
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

    Returns
    -------
    xi : Tensor, shape (n_features, n_states)
        Sparse coefficient matrix (detached, no grad).
    """
    # Initial least-squares solution
    xi = torch.linalg.lstsq(theta, dxdt).solution

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

    return xi.detach()
