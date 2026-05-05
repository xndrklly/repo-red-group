"""
nodal_loss.py  —  NN-EUCLID loss  (eq. 22, Thakolkaran et al. 2022)
--------------------------------------------------------------------
Loss = mean_ij[ (div sigma)^2 ]
       + w_rxn * mean_i[ ( <sigma_yy>_i(row i) - sigma_target )^2 ]

``sigma_target = F / (M * DX)`` is the mean ``sigma_yy`` that gives slice
resultant ``∫ sigma_yy dx = F`` on a row of ``M`` points spaced by ``DX``.

Averaging the squared error over **all** rows ``i`` enforces (approximately)
the same vertical slice resultant at every height — a flat ``R_y(y)`` —
not only at the bottom. That aligns stress magnitude with ``F`` through
the depth, which bottom-only does not guarantee when ``div sigma ≈ 0``
alone is underdetermined.

Training uses ``initial_w_rxn`` so ``w_rxn * loss_rxn ~ loss_free`` at
init. Legacy checkpoints use ``DEFAULT_W_RXN``.

**Caveat:** constant ``R_y(y)=F`` at every height is exact for a vertical
``y``-column with no body force; bending-dominated 2D states can have
``y``-varying slice resultants. Use this loss when a **flat** resultant
matches the intended physics (e.g. predominantly axial transfer).
"""

import torch

DEFAULT_W_RXN = 1e6


def initial_w_rxn(model, eps_t, DX, F_reaction_y,
                  min_w=1e-12, max_w=1e10, lr_floor_frac=1e-12):
    """
    Set w_rxn ≈ loss_free / loss_rxn at init so ``w_rxn * loss_rxn`` is on
    the same order as ``loss_free`` (both terms contribute to gradients).

    When loss_rxn ≫ loss_free, w_rxn is **small** (< 1); a floor of 1 would
    break that balance (reaction dominates the whole run).

    Returns
    -------
    w_rxn : float
    lf0, lr0 : initial unweighted loss_free and loss_rxn
    """
    model.eval()
    with torch.no_grad():
        _, lf, lr = nodal_loss(model, eps_t, DX, F_reaction_y, w_rxn=0.0)
    lf0 = float(lf.item())
    lr0 = float(lr.item())
    # Avoid division by zero / astronomical w_rxn when lr0 is tiny.
    denom = max(lr0, abs(lf0) * lr_floor_frac, 1e-30)
    w = lf0 / denom
    # min_w: numerical floor only (not 1.0 — that over-weights rxn when lr0 ≫ lf0)
    w = float(min(max(w, min_w), max_w))
    return w, lf0, lr0


def nodal_loss(model, eps_t, DX, F_reaction_y, w_rxn=DEFAULT_W_RXN):
    """
    Parameters
    ----------
    model        : ICNN
    eps_t        : (M, M, 3) torch tensor on device
    DX           : float
    F_reaction_y : float — total vertical reaction force
    w_rxn        : float — weight on slice-resultant / mean-stress term

    Returns
    -------
    loss, loss_free (detached), loss_rxn (detached, unweighted)
        ``loss_rxn`` is mean over rows of
        ``(row_mean(sigma_yy) - sigma_target)^2``.
    """
    M = eps_t.shape[0]

    sigma_flat = model(eps_t.reshape(-1, 3))
    sigma      = sigma_flat.reshape(M, M, 3)
    s11, s22, s12 = sigma[:, :, 0], sigma[:, :, 1], sigma[:, :, 2]

    # Interior equilibrium: mean squared divergence
    div_x = ((s11[1:-1, 2:] - s11[1:-1, :-2])
            + (s12[2:, 1:-1] - s12[:-2, 1:-1])) / (2 * DX)
    div_y = ((s12[1:-1, 2:] - s12[1:-1, :-2])
            + (s22[2:, 1:-1] - s22[:-2, 1:-1])) / (2 * DX)
    loss_free = (div_x**2 + div_y**2).mean()

    # Slice resultant: at each row i, mean σ_yy should match σ_target so
    # ∫ σ_yy dx = M*DX*<σ_yy>_i = F (flat R_y at all heights on this grid).
    sigma_target = F_reaction_y / (M * DX)
    sigma_pred_rows = s22.mean(dim=1)
    loss_rxn = ((sigma_pred_rows - sigma_target) ** 2).mean()

    w = float(w_rxn)
    loss = loss_free + w * loss_rxn

    return loss, loss_free.detach(), loss_rxn.detach()
