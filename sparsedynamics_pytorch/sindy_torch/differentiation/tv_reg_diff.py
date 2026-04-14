"""
Total Variation Regularized Differentiation — tensor wrapper.

The core algorithm uses SciPy sparse matrices and conjugate gradient solvers,
which are best served by NumPy. This module wraps the NumPy implementation
with tensor I/O for seamless integration into the PyTorch SINDy pipeline.

This is intentionally NOT differentiable — it is a preprocessing step.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

import torch
from torch import Tensor


def tv_reg_diff(
    data: Tensor,
    n_iter: int,
    alpha: float,
    u0: Tensor = None,
    scale: str = "small",
    ep: float = 1e-6,
    dx: float = None,
    diag_flag: bool = False,
) -> Tensor:
    """Estimate derivative of noisy data using total variation regularization.

    Thin wrapper: converts to NumPy, runs the SciPy-based solver,
    returns a Tensor on the same device as the input.

    Parameters
    ----------
    data : Tensor, shape (n,)
        Noisy data to differentiate.
    n_iter : int
        Number of iterations.
    alpha : float
        Regularization strength (higher = smoother).
    u0 : Tensor, optional
        Initialization.
    scale : str
        'small' or 'large'.
    ep : float
        Epsilon to avoid division by zero.
    dx : float, optional
        Grid spacing (default 1/n).
    diag_flag : bool
        Print convergence diagnostics.

    Returns
    -------
    Tensor
        Estimated derivative. Length n+1 if scale='small', n if 'large'.
    """
    device = data.device
    dtype = data.dtype
    data_np = data.detach().cpu().numpy()
    u0_np = u0.detach().cpu().numpy() if u0 is not None else None

    result = _tv_reg_diff_numpy(
        data_np, n_iter, alpha, u0=u0_np, scale=scale, ep=ep, dx=dx,
        diag_flag=diag_flag,
    )
    return torch.tensor(result, device=device, dtype=dtype)


# ─── NumPy implementation (copied from sparsedynamics_python) ───


def _tv_reg_diff_numpy(data, n_iter, alpha, u0=None, scale="small", ep=1e-6,
                       dx=None, diag_flag=False):
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    if dx is None:
        dx = 1.0 / n
    scale = scale.lower()
    if scale == "small":
        return _small(data, n, n_iter, alpha, u0, ep, dx, diag_flag)
    elif scale == "large":
        return _large(data, n, n_iter, alpha, u0, ep, dx, diag_flag)
    else:
        raise ValueError(f"scale must be 'small' or 'large', got '{scale}'")


def _small(data, n, n_iter, alpha, u0, ep, dx, diag_flag):
    e = np.ones(n + 1) / dx
    D = sparse.spdiags([-e, e], [0, 1], n, n + 1).tocsc()
    DT = D.T.tocsc()

    def A(x):
        cs = np.cumsum(x)
        return (cs[1:] - 0.5 * (x[1:] + x[0])) * dx

    def AT(w):
        sw = np.sum(w)
        cs = np.cumsum(w)
        result = np.empty(n + 1)
        result[0] = sw / 2.0
        result[1:] = sw - cs + w / 2.0
        return result * dx

    if u0 is None:
        u0 = np.zeros(n + 1)
        u0[1:-1] = np.diff(data)
    u = u0.copy()

    ofst = data[0]
    ATb = AT(ofst - data)

    for ii in range(n_iter):
        Du = D @ u
        q = 1.0 / np.sqrt(Du ** 2 + ep)
        Q = sparse.spdiags(q, 0, n, n).tocsc()
        L = dx * (DT @ Q @ D)
        g = AT(A(u)) + ATb + alpha * (L @ u)
        P_diag = alpha * L.diagonal() + 1.0

        def matvec(v, L=L, alpha=alpha):
            return alpha * (L @ v) + AT(A(v))

        LinOp = sparse.linalg.LinearOperator((n + 1, n + 1), matvec=matvec)
        s, _ = cg(
            LinOp, g, tol=1e-4, maxiter=100,
            M=sparse.linalg.LinearOperator(
                (n + 1, n + 1), matvec=lambda v: v / P_diag
            ),
        )
        if diag_flag:
            print(f"  iteration {ii+1:4d}: relative change = "
                  f"{np.linalg.norm(s)/np.linalg.norm(u):.3e}, "
                  f"gradient norm = {np.linalg.norm(g):.3e}")
        u = u - s
    return u


def _large(data, n, n_iter, alpha, u0, ep, dx, diag_flag):
    def A(v):
        return np.cumsum(v)

    def AT(w):
        sw = np.sum(w)
        result = np.empty(len(w))
        result[0] = sw
        result[1:] = sw - np.cumsum(w[:-1])
        return result

    e = np.ones(n)
    D = sparse.spdiags([-e, e], [0, 1], n, n).tocsc() / dx
    D = D.tolil()
    D[n - 1, n - 1] = 0
    D = D.tocsc()
    DT = D.T.tocsc()

    data = data - data[0]
    if u0 is None:
        u0 = np.zeros(n)
        u0[1:] = np.diff(data)
    u = u0.copy()
    ATd = AT(data)

    for ii in range(n_iter):
        Du = D @ u
        q = 1.0 / np.sqrt(Du ** 2 + ep)
        Q = sparse.spdiags(q, 0, n, n).tocsc()
        L = DT @ Q @ D
        g = AT(A(u)) - ATd + alpha * (L @ u)

        c = np.cumsum(np.arange(n, 0, -1, dtype=float))
        B = alpha * L + sparse.spdiags(c[::-1], 0, n, n).tocsc()
        B_diag = np.abs(B.diagonal()) + 1e-10
        P_inv_diag = 1.0 / B_diag

        def matvec(v, L=L, alpha=alpha):
            return alpha * (L @ v) + AT(A(v))

        LinOp = sparse.linalg.LinearOperator((n, n), matvec=matvec)
        s, _ = cg(
            LinOp, -g, tol=1e-4, maxiter=100,
            M=sparse.linalg.LinearOperator(
                (n, n), matvec=lambda v: v * P_inv_diag
            ),
        )
        if diag_flag:
            print(f"  iteration {ii+1:2d}: relative change = "
                  f"{np.linalg.norm(s)/np.linalg.norm(u):.3e}, "
                  f"gradient norm = {np.linalg.norm(g):.3e}")
        u = u + s
    return u
