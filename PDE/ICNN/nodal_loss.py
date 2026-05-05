"""
nodal_loss.py  —  NN-EUCLID loss  (eq. 22, Thakolkaran et al. 2022)
--------------------------------------------------------------------
Loss = sum over FREE nodes of (f_i^a)^2
     + sum over FIXED boundaries of (R_beta - r_beta)^2

where f_i^a is the nodal force residual at node a in direction i,
and r_beta is the predicted reaction force on boundary beta.

This is DIFFERENT from our weak-form approach:
  - No random test functions, no rxn_factor scaling issue
  - Free-node residuals and reaction residuals are both in (force)^2
    so they're naturally on the same scale — no weighting needed
  - Nodal forces computed via finite-difference of stress field
    (mimics the FEM internal force assembly: f = integral sigma:grad(N) dV)

Grid layout (matches smoothed output):
  eval_grid : (M, M, 2)  — eval point coordinates
  eps       : (M, M, 3)  — strain field [e11, e22, e12]
  DX        : float       — grid spacing

Boundary conventions (matching data generation in simple_data_gen.py):
  bottom row (i=0)  : fixed in y  (Dirichlet, reaction measured here)
  top row    (i=-1) : prescribed displacement (Dirichlet)
  left/right cols   : free in x (could be free or periodic — treated as free)

The nodal force at interior node (i,j) in direction d is approximated as:
  f_x^{i,j} = (s11_{i,j+1} - s11_{i,j-1})/(2*DX)
             + (s12_{i+1,j} - s12_{i-1,j})/(2*DX)
  f_y^{i,j} = (s12_{i,j+1} - s12_{i,j-1})/(2*DX)
             + (s22_{i+1,j} - s22_{i-1,j})/(2*DX)

This is the strong-form divergence evaluated at each node, multiplied by
DX^2 to convert to a nodal force (integral over the Voronoi cell).

Reaction force (bottom boundary, y-direction):
  r_y = sum_j s22[0, j] * DX   (resultant from bottom row stresses)
"""

import torch


def nodal_loss(model, eps_t, DX, F_reaction_y):
    """
    Compute the NN-EUCLID loss (eq. 22).

    Parameters
    ----------
    model        : ICNN
    eps_t        : (M, M, 3) torch tensor — strain field
    DX           : float — grid spacing
    F_reaction_y : float — measured total vertical reaction force

    Returns
    -------
    loss         : scalar torch tensor
    loss_free    : scalar (detached) — interior equilibrium contribution
    loss_rxn     : scalar (detached) — reaction force contribution
    """
    M = eps_t.shape[0]
    device = eps_t.device

    # Forward pass: get all stresses
    eps_flat = eps_t.reshape(-1, 3)
    sigma_flat = model(eps_flat)                  # (M*M, 3)
    sigma = sigma_flat.reshape(M, M, 3)

    s11 = sigma[:, :, 0]   # (M, M)
    s22 = sigma[:, :, 1]
    s12 = sigma[:, :, 2]

    # ----------------------------------------------------------------
    # Interior nodal force residuals  (div sigma = 0 strongly)
    # Use central differences on interior nodes i=1..M-2, j=1..M-2
    # f_x = ds11/dx + ds12/dy,  f_y = ds12/dx + ds22/dy
    # Multiply by DX^2 to get force units (area per node = DX^2)
    # ----------------------------------------------------------------
    # Interior slice: rows 1..M-2, cols 1..M-2
    s11_c = s11[1:-1, 1:-1];  s11_r = s11[1:-1, 2:];  s11_l = s11[1:-1, :-2]
    s22_c = s22[1:-1, 1:-1];  s22_u = s22[2:, 1:-1];  s22_d = s22[:-2, 1:-1]
    s12_r = s12[1:-1, 2:];    s12_l = s12[1:-1, :-2]
    s12_u = s12[2:, 1:-1];    s12_d = s12[:-2, 1:-1]

    # Central difference divergence * DX^2  (= nodal force at each interior node)
    f_x = ((s11_r - s11_l) / (2*DX) + (s12_u - s12_d) / (2*DX)) * DX**2
    f_y = ((s12_r - s12_l) / (2*DX) + (s22_u - s22_d) / (2*DX)) * DX**2

    loss_free = (f_x**2 + f_y**2).sum()

    # ----------------------------------------------------------------
    # Reaction force residual (bottom boundary, y-direction)
    # r_y = sum_j sigma_22[0, j] * DX  (traction integrated over bottom edge)
    # Target: F_reaction_y from the simulation
    # ----------------------------------------------------------------
    r_y = s22[0, :].sum() * DX
    loss_rxn = (r_y - F_reaction_y) ** 2

    # Total loss — no weighting factor needed, both terms are in force^2
    loss = loss_free + loss_rxn

    return loss, loss_free.detach(), loss_rxn.detach()
