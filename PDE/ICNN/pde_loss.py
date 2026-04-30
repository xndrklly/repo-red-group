"""
pde_loss.py  (ICNN)
-------------------
Weak-form loss for static equilibrium:  div(sigma) = 0

Identical to MLP/pde_loss.py in structure (WSINDy-style weak form):
  - raised-cosine test functions with compact support at random centers
  - integration by parts: R_k = ∫ grad(ψ_k) : σ(ε) dA = 0
  - loss = Σ_k |R_k|²

Only difference from MLP version: the model takes raw strains eps_flat
(shape M*M, 3) directly instead of pre-computed invariants. The ICNN
handles the invariant computation and autograd chain rule internally.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Test functions  (unchanged from MLP)
# ---------------------------------------------------------------------------

def raised_cosine(s, l):
    inside = np.abs(s) < l
    return np.where(inside, 0.5 * (1.0 + np.cos(np.pi * s / l)), 0.0)


def raised_cosine_grad(s, l):
    inside = np.abs(s) < l
    return np.where(inside, -0.5 * (np.pi / l) * np.sin(np.pi * s / l), 0.0)


def make_test_functions(X, K=50, lx=None, ly=None, seed=0):
    """
    Build K raised-cosine test functions on the evaluation grid X.

    Parameters
    ----------
    X    : (M, M, 2)  grid coordinates
    K    : int        number of test functions
    lx   : float      spatial support in x (default L/4)
    ly   : float      spatial support in y (default L/4)
    seed : int        random seed for center placement

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
        lx = L / 8.0
    if ly is None:
        ly = L / 8.0

    test_fns = []
    for _ in range(K):
        cx = rng.uniform(xmin + lx, xmax - lx)
        cy = rng.uniform(ymin + ly, ymax - ly)

        sx = X[:, :, 0] - cx
        sy = X[:, :, 1] - cy

        phi_x  = raised_cosine(sx, lx)
        phi_y  = raised_cosine(sy, ly)
        dphi_x = raised_cosine_grad(sx, lx)
        dphi_y = raised_cosine_grad(sy, ly)

        dpsi_dx = dphi_x * phi_y
        dpsi_dy = phi_x  * dphi_y

        grad_psi = np.stack([dpsi_dx, dpsi_dy], axis=-1)   # (M, M, 2)
        test_fns.append({'grad_psi': grad_psi})

    return test_fns


# ---------------------------------------------------------------------------
# Weak-form residual
# ---------------------------------------------------------------------------

def weak_residual(model, eps_t, grad_psi_t, DX):
    """
    Compute the weak-form residual for one test function (static case).

    R_k = sum_{I,J} grad_psi_k(X_IJ) · sigma(eps_IJ) * DX²

    Parameters
    ----------
    model      : ICNN
    eps_t      : (M, M, 3) torch tensor  -- strain field [e11, e22, e12]
    grad_psi_t : (M, M, 2) torch tensor  -- [dpsi/dx, dpsi/dy]
    DX         : float                   -- grid spacing

    Returns
    -------
    R : (2,) torch tensor  -- residual vector (should be ~0)
    """
    M1, M2, _ = eps_t.shape
    eps_flat = eps_t.reshape(-1, 3)             # (M*M, 3)

    # ICNN takes raw strains — invariants and chain rule handled inside model
    sigma_flat = model(eps_flat)                # (M*M, 3)  [s11, s22, s12]
    sigma = sigma_flat.reshape(M1, M2, 3)       # (M, M, 3)

    gx = grad_psi_t[:, :, 0]   # dpsi/dx
    gy = grad_psi_t[:, :, 1]   # dpsi/dy

    s11 = sigma[:, :, 0]
    s22 = sigma[:, :, 1]
    s12 = sigma[:, :, 2]

    # Weak form: (grad ψ)ᵀ σ
    #   R₁ = ∫ (∂ψ/∂x · σ₁₁ + ∂ψ/∂y · σ₁₂) dA
    #   R₂ = ∫ (∂ψ/∂x · σ₁₂ + ∂ψ/∂y · σ₂₂) dA
    R1 = (gx * s11 + gy * s12).sum() * DX ** 2
    R2 = (gx * s12 + gy * s22).sum() * DX ** 2

    return torch.stack([R1, R2])    # (2,)


# ---------------------------------------------------------------------------
# Total loss
# ---------------------------------------------------------------------------

def pde_loss(model, eps, grad_psis, DX):
    """
    Sum of squared weak-form residuals over all test functions.

    Parameters
    ----------
    model     : ICNN
    eps       : (M, M, 3) numpy array or torch tensor -- strain field
    grad_psis : list of (M, M, 2) numpy arrays        -- precomputed gradients
    DX        : float                                 -- grid spacing

    Returns
    -------
    loss : scalar torch tensor
    """
    device = next(model.parameters()).device
    if not isinstance(eps, torch.Tensor):
        eps = torch.tensor(eps, dtype=torch.float32, device=device)
    else:
        eps = eps.to(device)

    loss = torch.tensor(0.0, device=device)
    for gp in grad_psis:
        gp_t = torch.tensor(gp, dtype=torch.float32, device=device)
        R = weak_residual(model, eps, gp_t, DX)
        loss = loss + (R ** 2).sum()

    return loss


# ---------------------------------------------------------------------------
# Boundary reaction force loss  (inspired by nn-EUCLID train.py)
# ---------------------------------------------------------------------------

def boundary_reaction_loss(model, bdry_eps_t, F_target, DX):
    """
    Constrain the absolute stress scale by matching predicted boundary
    traction to the measured total reaction force.

    By global equilibrium, ∫ σ_yy(y) dx = const at every horizontal level,
    equal to the total vertical reaction at the bottom boundary.

    Parameters
    ----------
    model      : ICNN (in train mode)
    bdry_eps_t : (M, 3) torch tensor  -- strain at one horizontal row
    F_target   : float                -- total vertical reaction force
                                         (from spring solver, bottom nodes)
    DX         : float                -- grid spacing

    Returns
    -------
    scalar torch tensor
    """
    sigma   = model(bdry_eps_t)           # (M, 3)
    F_pred  = sigma[:, 1].sum() * DX      # ∫ σ_yy dx  (σ_22 = s22)
    return (F_pred - F_target) ** 2
