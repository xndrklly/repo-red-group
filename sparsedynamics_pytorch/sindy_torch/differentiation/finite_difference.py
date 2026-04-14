"""
4th-order central finite difference for derivative estimation.

Port of utils.compute_derivative_4th_order, operating on tensors.
"""

from typing import Tuple

from torch import Tensor


def finite_difference_4th(x: Tensor, dt: float) -> Tuple[Tensor, Tensor]:
    """Compute derivatives using 4th-order central finite differences.

    Formula: dx[i] = (1/(12*dt)) * (-x[i+2] + 8*x[i+1] - 8*x[i-1] + x[i-2])

    Parameters
    ----------
    x : Tensor, shape (n_samples, n_vars)
        State trajectory.
    dt : float
        Time step.

    Returns
    -------
    dx : Tensor, shape (n_samples - 4, n_vars)
        Derivatives at interior points (indices 2..n_samples-3).
    x_trimmed : Tensor, shape (n_samples - 4, n_vars)
        State data at matching interior indices.
    """
    dx = (1.0 / (12 * dt)) * (
        -x[4:] + 8 * x[3:-1] - 8 * x[1:-3] + x[:-4]
    )
    x_trimmed = x[2:-2]
    return dx, x_trimmed
