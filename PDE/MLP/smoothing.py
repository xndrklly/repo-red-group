"""
smoothing.py
------------
Gaussian kernel smoothing of discrete lattice displacement data to a
continuously-differentiable macroscopic field, plus analytical computation
of the symmetric strain tensor.

All spatial derivatives are obtained by differentiating the Gaussian kernel
analytically — no finite-differencing of the displacement data anywhere.

Usage
-----
    from smoothing import GaussianSmoother, make_eval_grid, strain_invariants

    smoother  = GaussianSmoother(node_pos, u, h=2.0*a)
    eval_grid = make_eval_grid(node_pos, downsample=1, margin=2)
    u_hat, eps = smoother.compute_vectorised(eval_grid)
    inv        = strain_invariants(eps)   # I1, I2 for NN input

Shapes
------
    node_pos  : (n, n, 2)    -- lattice node positions (x, y)
    u         : (n, n, T, 2) -- nodal displacements [x-comp, y-comp]
    h         : float         -- Gaussian bandwidth (physical units)

    u_hat     : (M, M, T, 2)  -- smoothed displacement field on eval grid
    eps       : (M, M, T, 3)  -- strain tensor [eps11, eps22, eps12]
    inv       : (M, M, T, 2)  -- strain invariants [I1, I2]
"""

import numpy as np


# ---------------------------------------------------------------------------
# Gaussian kernel and its spatial derivatives
# ---------------------------------------------------------------------------

def _gaussian(r2: np.ndarray, h: float) -> np.ndarray:
    """Isotropic 2-D Gaussian kernel:  G(r, h) = exp(-|r|^2 / (2h^2)).

    Parameters
    ----------
    r2 : array of squared distances |x - x_i|^2
    h  : bandwidth (physical units)

    Returns
    -------
    G  : same shape as r2
    """
    return np.exp(-r2 / (2.0 * h * h))


def _gaussian_grad(dx: np.ndarray, dy: np.ndarray,
                   G: np.ndarray, h: float):
    """First spatial derivatives of the isotropic Gaussian kernel.

    dG/dx_alpha = G(r, h) * (-(x_alpha - x_i_alpha) / h^2)

    This follows directly from differentiating G = exp(-|r|^2 / 2h^2).

    Parameters
    ----------
    dx, dy : arrays of shape (...) -- (eval_x - node_x), (eval_y - node_y)
    G      : array of same shape   -- kernel values (pre-computed)
    h      : bandwidth

    Returns
    -------
    dGdx, dGdy : same shape as G
    """
    inv_h2 = 1.0 / (h * h)
    dGdx = -G * dx * inv_h2
    dGdy = -G * dy * inv_h2
    return dGdx, dGdy


# ---------------------------------------------------------------------------
# Main smoother class
# ---------------------------------------------------------------------------

class GaussianSmoother:
    """
    Nadaraya-Watson Gaussian kernel smoother for 2-D lattice displacement data.

    The smoothed field is defined as the weighted average:
        u_hat(x, t) = [sum_k u(x_k, t) * G(x - x_k, h)]
                    / [sum_k G(x - x_k, h)]

    Spatial derivatives are computed analytically via the quotient rule:
        d(u_hat)/dx = [sum_k u_k * dG/dx(x - x_k)  * W
                       - sum_k u_k * G(x - x_k)     * dW/dx] / W^2

    where W(x) = sum_k G(x - x_k, h) is the normalisation.

    Parameters
    ----------
    node_pos : (n, n, 2)   -- (x, y) positions of every lattice node
    u        : (n, n, T, 2)-- displacement data; d=0 is x, d=1 is y
    h        : float        -- Gaussian bandwidth (recommend 2a to 3a,
                               where a is the lattice spacing)
    cutoff   : float        -- support truncation in units of h (default 3).
                               Nodes with |x - x_k| > cutoff*h are ignored,
                               giving O(n^2 * (cutoff*h/a)^2) cost per point.
    """

    def __init__(self,
                 node_pos: np.ndarray,
                 u: np.ndarray,
                 h: float,
                 cutoff: float = 3.0):

        assert node_pos.ndim == 3 and node_pos.shape[2] == 2, \
            "node_pos must be shape (n, n, 2)"
        assert u.ndim == 4 and u.shape[3] == 2, \
            "u must be shape (n, n, T, 2)"
        assert node_pos.shape[:2] == u.shape[:2], \
            "node_pos and u must share the same (n, n) leading dimensions"

        self.node_pos = node_pos
        self.u        = u
        self.h        = float(h)
        self.cutoff   = float(cutoff)
        self.cutoff2  = (cutoff * h) ** 2    # squared cutoff distance

        # Flattened views for vectorised computation
        self._nodes_flat = node_pos.reshape(-1, 2)              # (N, 2)
        self._u_flat     = u.reshape(-1, u.shape[2], u.shape[3])  # (N, T, 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_vectorised(self, eval_grid: np.ndarray):
        """
        Evaluate the smoothed displacement and strain tensor on a grid.

        This is the recommended entry point.  All eval points are processed
        in a single batch of NumPy operations.  Memory use is O(P * N) where
        P = M1*M2 eval points and N = n^2 nodes; for very large grids use
        compute() instead.

        Parameters
        ----------
        eval_grid : (M1, M2, 2) -- (x, y) coordinates of evaluation points

        Returns
        -------
        u_hat : (M1, M2, T, 2) -- smoothed displacement [u_x, u_y]
        eps   : (M1, M2, T, 3) -- strain tensor [eps11, eps22, eps12]
                                   eps12 = 0.5*(du_x/dy + du_y/dx)
        """
        assert eval_grid.ndim == 3 and eval_grid.shape[2] == 2, \
            "eval_grid must be (M1, M2, 2)"

        M1, M2, _ = eval_grid.shape
        T = self.u.shape[2]
        P = M1 * M2
        N = self._nodes_flat.shape[0]

        # Reshape to (P, 2) for broadcasting
        X = eval_grid.reshape(P, 2)           # (P, 2)

        # Displacement vectors: (eval_pt - node)
        # Broadcasting: X[:,0] is (P,), nodes[:,0] is (N,)
        # dx[p, k] = X[p, 0] - nodes[k, 0]
        dx = X[:, 0:1] - self._nodes_flat[:, 0]   # (P, N)
        dy = X[:, 1:2] - self._nodes_flat[:, 1]   # (P, N)
        r2 = dx**2 + dy**2                         # (P, N)

        # Cutoff mask
        mask = r2 <= self.cutoff2                   # (P, N) bool

        # Kernel values — zero outside support
        G = np.where(mask, _gaussian(r2, self.h), 0.0)    # (P, N)

        # Normalisation: W(x) = sum_k G_k(x)
        W = G.sum(axis=1, keepdims=True)           # (P, 1)
        # Guard against degenerate points (no nodes in support)
        W = np.where(W == 0.0, 1e-30, W)

        # Kernel derivatives — zero outside support
        inv_h2 = 1.0 / (self.h ** 2)
        dGdx = np.where(mask, -G * dx * inv_h2, 0.0)      # (P, N)
        dGdy = np.where(mask, -G * dy * inv_h2, 0.0)      # (P, N)

        # Normalisation derivatives
        dWdx = dGdx.sum(axis=1, keepdims=True)    # (P, 1)
        dWdy = dGdy.sum(axis=1, keepdims=True)    # (P, 1)

        # ------------------------------------------------------------------
        # Smoothed displacement field
        #   u_hat[p, t, d] = sum_k G[p,k] * u[k,t,d] / W[p]
        # einsum: 'pN, NTd -> pTd'
        # ------------------------------------------------------------------
        u_flat = self._u_flat                      # (N, T, 2)
        GU = np.einsum('pN,NTd->pTd', G, u_flat)  # (P, T, 2)
        u_hat_flat = GU / W[:, :, np.newaxis]      # (P, T, 2)

        # ------------------------------------------------------------------
        # Spatial derivatives via quotient rule
        #
        #   d(u_hat)/dx = [dGdx_U * W - GU * dWdx] / W^2
        #
        # where  dGdx_U[p,t,d] = sum_k dG/dx[p,k] * u[k,t,d]
        # ------------------------------------------------------------------
        dGdx_U = np.einsum('pN,NTd->pTd', dGdx, u_flat)  # (P, T, 2)
        dGdy_U = np.einsum('pN,NTd->pTd', dGdy, u_flat)  # (P, T, 2)

        W_    = W[:, :, np.newaxis]          # (P, 1, 1)  broadcast over T, d
        dWdx_ = dWdx[:, :, np.newaxis]      # (P, 1, 1)
        dWdy_ = dWdy[:, :, np.newaxis]      # (P, 1, 1)

        du_dx = (dGdx_U * W_ - GU * dWdx_) / (W_ ** 2)   # (P, T, 2)
        du_dy = (dGdy_U * W_ - GU * dWdy_) / (W_ ** 2)   # (P, T, 2)

        # ------------------------------------------------------------------
        # Symmetric linearised strain tensor
        #   eps11 = du_x/dx          (d=0, deriv w.r.t. x)
        #   eps22 = du_y/dy          (d=1, deriv w.r.t. y)
        #   eps12 = 0.5*(du_x/dy + du_y/dx)
        # ------------------------------------------------------------------
        eps11_flat = du_dx[:, :, 0]                              # (P, T)
        eps22_flat = du_dy[:, :, 1]                              # (P, T)
        eps12_flat = 0.5 * (du_dy[:, :, 0] + du_dx[:, :, 1])   # (P, T)

        eps_flat = np.stack([eps11_flat, eps22_flat, eps12_flat], axis=-1)
        # eps_flat : (P, T, 3)

        # Reshape to (M1, M2, T, ...)
        u_hat = u_hat_flat.reshape(M1, M2, T, 2)
        eps   = eps_flat.reshape(M1, M2, T, 3)

        return u_hat, eps

    def compute(self, eval_grid: np.ndarray):
        """
        Loop-based version of compute_vectorised.  Lower memory footprint at
        the cost of speed.  Useful when M*N would be very large.

        Parameters / Returns: same as compute_vectorised.
        """
        assert eval_grid.ndim == 3 and eval_grid.shape[2] == 2

        M1, M2, _ = eval_grid.shape
        T = self.u.shape[2]

        u_hat = np.zeros((M1, M2, T, 2))
        eps   = np.zeros((M1, M2, T, 3))

        for i in range(M1):
            for j in range(M2):
                uh, e        = self._eval_point(eval_grid[i, j])
                u_hat[i, j] = uh
                eps[i, j]   = e

        return u_hat, eps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_point(self, x_eval: np.ndarray):
        """Evaluate smoothed displacement and strain at a single point.

        Parameters
        ----------
        x_eval : (2,) -- evaluation coordinate (x, y)

        Returns
        -------
        u_hat : (T, 2)
        eps   : (T, 3) -- [eps11, eps22, eps12]
        """
        nodes  = self._nodes_flat   # (N, 2)
        u_flat = self._u_flat       # (N, T, 2)
        T      = u_flat.shape[1]

        dx = x_eval[0] - nodes[:, 0]   # (N,)
        dy = x_eval[1] - nodes[:, 1]   # (N,)
        r2 = dx**2 + dy**2             # (N,)

        in_support = r2 <= self.cutoff2
        if not np.any(in_support):
            return np.zeros((T, 2)), np.zeros((T, 3))

        dx_ = dx[in_support]
        dy_ = dy[in_support]
        r2_ = r2[in_support]
        u_  = u_flat[in_support]       # (K, T, 2)

        G_             = _gaussian(r2_, self.h)
        dGdx_, dGdy_   = _gaussian_grad(dx_, dy_, G_, self.h)

        W    = G_.sum()
        dWdx = dGdx_.sum()
        dWdy = dGdy_.sum()

        GU     = np.einsum('k,kTd->Td', G_,    u_)   # (T, 2)
        dGdx_U = np.einsum('k,kTd->Td', dGdx_, u_)   # (T, 2)
        dGdy_U = np.einsum('k,kTd->Td', dGdy_, u_)   # (T, 2)

        u_hat  = GU / W
        du_dx  = (dGdx_U * W - GU * dWdx) / (W**2)
        du_dy  = (dGdy_U * W - GU * dWdy) / (W**2)

        eps11 = du_dx[:, 0]
        eps22 = du_dy[:, 1]
        eps12 = 0.5 * (du_dy[:, 0] + du_dx[:, 1])
        eps   = np.stack([eps11, eps22, eps12], axis=-1)   # (T, 3)

        return u_hat, eps


# ---------------------------------------------------------------------------
# Convenience: build a regular evaluation grid from the lattice
# ---------------------------------------------------------------------------

def make_eval_grid(node_pos: np.ndarray,
                   downsample: int = 1,
                   margin: int = 0) -> np.ndarray:
    """
    Build a regular evaluation grid by selecting a subset of lattice nodes.

    Parameters
    ----------
    node_pos   : (n, n, 2) -- lattice node positions
    downsample : int        -- stride (1 = same resolution as lattice)
    margin     : int        -- number of boundary nodes to skip on each side.
                               Use margin >= ceil(cutoff*h/a) to keep all
                               evaluation points well inside the kernel support.

    Returns
    -------
    eval_grid : (M, M, 2)
    """
    n   = node_pos.shape[0]
    idx = np.arange(margin, n - margin, downsample)
    return node_pos[np.ix_(idx, idx)]    # (M, M, 2)


# ---------------------------------------------------------------------------
# Strain invariants for NN input
# ---------------------------------------------------------------------------

def strain_invariants(eps: np.ndarray) -> np.ndarray:
    """
    Compute the two 2-D strain invariants used as NN inputs.

    For a 2-D isotropic material these two invariants fully characterise the
    strain state under the assumption of frame indifference:

        I1 = tr(eps)    = eps11 + eps22
        I2 = tr(eps^2)  = eps11^2 + 2*eps12^2 + eps22^2

    Parameters
    ----------
    eps : (..., 3) -- last axis is [eps11, eps22, eps12]

    Returns
    -------
    inv : (..., 2) -- last axis is [I1, I2]
    """
    eps11 = eps[..., 0]
    eps22 = eps[..., 1]
    eps12 = eps[..., 2]

    I1 = eps11 + eps22
    I2 = eps11**2 + 2.0 * eps12**2 + eps22**2

    return np.stack([I1, I2], axis=-1)
