"""
pde_loss.py  (ICNN_weak)
------------------------
Weak-form (WSINDy-style) loss for static equilibrium: div(sigma) = 0

Uses K random raised-cosine test functions. The reaction force is a
separate term weighted by rxn_factor — this is the known scaling issue.
See ../ICNN/ for the cleaner nodal-force formulation.
"""

import numpy as np
import torch


def raised_cosine(s, l):
    inside = np.abs(s) < l
    return np.where(inside, 0.5 * (1.0 + np.cos(np.pi * s / l)), 0.0)


def raised_cosine_grad(s, l):
    inside = np.abs(s) < l
    return np.where(inside, -0.5 * (np.pi / l) * np.sin(np.pi * s / l), 0.0)


def make_test_functions(X, K=50, lx=None, ly=None, seed=0):
    rng = np.random.default_rng(seed)
    xmin, xmax = X[:, :, 0].min(), X[:, :, 0].max()
    ymin, ymax = X[:, :, 1].min(), X[:, :, 1].max()
    L = max(xmax - xmin, ymax - ymin)
    if lx is None: lx = L / 8.0
    if ly is None: ly = L / 8.0

    test_fns = []
    for _ in range(K):
        cx = rng.uniform(xmin + lx, xmax - lx)
        cy = rng.uniform(ymin + ly, ymax - ly)
        sx = X[:, :, 0] - cx
        sy = X[:, :, 1] - cy
        phi_x  = raised_cosine(sx, lx);    phi_y  = raised_cosine(sy, ly)
        dphi_x = raised_cosine_grad(sx, lx); dphi_y = raised_cosine_grad(sy, ly)
        grad_psi = np.stack([dphi_x * phi_y, phi_x * dphi_y], axis=-1)
        test_fns.append({'grad_psi': grad_psi})
    return test_fns


def weak_residual(model, eps_t, grad_psi_t, DX):
    M1, M2, _ = eps_t.shape
    sigma = model(eps_t.reshape(-1, 3)).reshape(M1, M2, 3)
    gx = grad_psi_t[:, :, 0];  gy = grad_psi_t[:, :, 1]
    s11, s22, s12 = sigma[:,:,0], sigma[:,:,1], sigma[:,:,2]
    R1 = (gx * s11 + gy * s12).sum() * DX**2
    R2 = (gx * s12 + gy * s22).sum() * DX**2
    return torch.stack([R1, R2])


def pde_loss(model, eps, grad_psis, DX):
    device = next(model.parameters()).device
    if not isinstance(eps, torch.Tensor):
        eps = torch.tensor(eps, dtype=torch.float32, device=device)
    loss = torch.tensor(0.0, device=device)
    for gp in grad_psis:
        gp_t = gp if isinstance(gp, torch.Tensor) else torch.tensor(gp, dtype=torch.float32, device=device)
        R = weak_residual(model, eps, gp_t, DX)
        loss = loss + (R**2).sum()
    return loss


def boundary_reaction_loss(model, bdry_eps_t, F_target, DX):
    sigma  = model(bdry_eps_t)
    F_pred = sigma[:, 1].sum() * DX
    return (F_pred - F_target)**2
