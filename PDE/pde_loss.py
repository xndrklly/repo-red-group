"""
pde_loss.py
-----------
Weak-form loss for static equilibrium:  div(sigma) = 0

Weak form: integrate by parts once in space:
    R_k = integral[ grad(psi_k) : sigma(eps) ] dA = 0

Test functions psi_k are raised-cosine bumps with compact support.
No time derivatives needed (static case).

Expected inputs (all numpy arrays, converted to torch internally):
    eps : (M, M, 3)   strain field  [e11, e22, e12]
    X   : (M, M, 2)   evaluation grid coordinates
    DX  : float       grid spacing
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def raised_cosine(s, l):
    """Scalar raised cosine bump.  Returns 0 outside |s| < l."""
    inside = np.abs(s) < l
    out = np.where(inside, 0.5 * (1.0 + np.cos(np.pi * s / l)), 0.0)
    return out


def raised_cosine_grad(s, l):
    """Derivative of raised cosine w.r.t. s."""
    inside = np.abs(s) < l
    out = np.where(inside, -0.5 * (np.pi / l) * np.sin(np.pi * s / l), 0.0)
    return out


def make_test_functions(X, K=50, lx=None, ly=None, seed=0):
    """
    Build K raised-cosine test functions on the evaluation grid X.

    Parameters
    ----------
    X   : (M, M, 2)  grid coordinates
    K   : int         number of test functions
    lx  : float       spatial support in x (default L/4)
    ly  : float       spatial support in y (default L/4)
    seed: int         random seed for center placement

    Returns
    -------
    list of dicts, each with:
        'grad_psi' : (M, M, 2)  [dpsi/dx, dpsi/dy]
    """
    rng = np.random.default_rng(seed)

    xmin, xmax = X[:, :, 0].min(), X[:, :, 0].max()
    ymin, ymax = X[:, :, 1].min(), X[:, :, 1].max()
    L = max(xmax - xmin, ymax - ymin)

    if lx is None:
        lx = L / 4.0
    if ly is None:
        ly = L / 4.0

    test_fns = []
    for _ in range(K):
        # Random center, kept away from boundary so support fits inside
        cx = rng.uniform(xmin + lx, xmax - lx)
        cy = rng.uniform(ymin + ly, ymax - ly)

        sx = X[:, :, 0] - cx   # (M, M)
        sy = X[:, :, 1] - cy   # (M, M)

        phi_x  = raised_cosine(sx, lx)        # (M, M)
        phi_y  = raised_cosine(sy, ly)        # (M, M)
        dphi_x = raised_cosine_grad(sx, lx)   # (M, M)
        dphi_y = raised_cosine_grad(sy, ly)   # (M, M)

        # psi(x,y) = phi_x(x) * phi_y(y)
        # dpsi/dx  = dphi_x(x) * phi_y(y)
        # dpsi/dy  = phi_x(x)  * dphi_y(y)
        dpsi_dx = dphi_x * phi_y   # (M, M)
        dpsi_dy = phi_x * dphi_y   # (M, M)

        grad_psi = np.stack([dpsi_dx, dpsi_dy], axis=-1)  # (M, M, 2)
        test_fns.append({'grad_psi': grad_psi})

    return test_fns


# ---------------------------------------------------------------------------
# Weak-form residual
# ---------------------------------------------------------------------------

def weak_residual(model, eps_t, grad_psi_t, DX):
    """
    Compute the weak-form residual for one test function (static case).

    R_k = sum_{I,J} grad_psi_k(X_IJ) . sigma(eps_IJ) * DX^2

    In index notation the integrand is:
        (dpsi/dx_i) * sigma_ij   summed over i, giving a vector in j

    For 2D with sigma = [s11, s22, s12]:
        component 1: dpsi/dx * s11 + dpsi/dy * s12
        component 2: dpsi/dx * s12 + dpsi/dy * s22

    Parameters
    ----------
    model      : ConstitutiveNN
    eps_t      : (M, M, 3) torch tensor  -- strain field
    grad_psi_t : (M, M, 2) torch tensor  -- [dpsi/dx, dpsi/dy]
    DX         : float                   -- grid spacing

    Returns
    -------
    R : (2,) torch tensor  -- residual vector (should be ~0)
    """
    # Flatten spatial dims for batched NN forward pass
    M1, M2, _ = eps_t.shape
    eps_flat = eps_t.reshape(-1, 3)                 # (M*M, 3)

    # Strain invariants: I1 = e11+e22, I2 = e11^2 + 2*e12^2 + e22^2
    e11, e22, e12 = eps_flat[:, 0], eps_flat[:, 1], eps_flat[:, 2]
    I1 = e11 + e22
    I2 = e11**2 + 2.0 * e12**2 + e22**2
    inv = torch.stack([I1, I2], dim=-1)             # (M*M, 2)

    sigma_flat = model(inv)                         # (M*M, 3)  [s11, s22, s12]
    sigma = sigma_flat.reshape(M1, M2, 3)           # (M, M, 3)

    gx = grad_psi_t[:, :, 0]   # (M, M)  dpsi/dx
    gy = grad_psi_t[:, :, 1]   # (M, M)  dpsi/dy

    s11 = sigma[:, :, 0]
    s22 = sigma[:, :, 1]
    s12 = sigma[:, :, 2]

    # Weak form integrand: (grad psi)^T sigma
    #   R_1 = dpsi/dx * s11 + dpsi/dy * s12
    #   R_2 = dpsi/dx * s12 + dpsi/dy * s22
    integrand_1 = gx * s11 + gy * s12   # (M, M)
    integrand_2 = gx * s12 + gy * s22   # (M, M)

    R1 = integrand_1.sum() * DX**2
    R2 = integrand_2.sum() * DX**2

    return torch.stack([R1, R2])         # (2,)


# ---------------------------------------------------------------------------
# Total loss
# ---------------------------------------------------------------------------

def pde_loss(model, eps, grad_psis, DX):
    """
    Sum of squared weak-form residuals over all test functions.

    Parameters
    ----------
    model     : ConstitutiveNN
    eps       : (M, M, 3) numpy array or torch tensor -- strain field
    grad_psis : list of (M, M, 2) numpy arrays        -- precomputed gradients
    DX        : float                                 -- grid spacing

    Returns
    -------
    loss : scalar torch tensor
    """
    if not isinstance(eps, torch.Tensor):
        eps = torch.tensor(eps, dtype=torch.float32)

    loss = torch.tensor(0.0, requires_grad=False)
    for gp in grad_psis:
        gp_t = torch.tensor(gp, dtype=torch.float32)
        R = weak_residual(model, eps, gp_t, DX)
        loss = loss + (R**2).sum()

    return loss
