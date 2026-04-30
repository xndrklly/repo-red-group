"""
train.py
--------
Training script for static equilibrium case.

Expected inputs:
    eps : (M, M, 3)  strain field from smoothing.py
    X   : (M, M, 2)  evaluation grid coordinates
    DX  : float      grid spacing

Example usage:
    from smoothing import GaussianSmoother, make_eval_grid, strain_invariants
    from train import train

    smoother  = GaussianSmoother(node_pos, u, h=2.0*a)
    eval_grid = make_eval_grid(node_pos, margin=3)
    u_hat, eps = smoother.compute_vectorised(eval_grid)
    # Static: squeeze out time dim (T=1) or pick one snapshot
    eps_static = eps[:, :, 0, :]   # (M, M, 3)

    model, losses = train(eps_static, eval_grid, DX=a)
"""

import math
import time
import torch
import torch.optim as optim

from nn_model import ConstitutiveNN
from pde_loss import make_test_functions, pde_loss, boundary_reaction_loss


def train(eps, X, DX,
          K=50, epochs=2000, lr=1e-3,
          hidden=64, layers=3,
          reaction_force=None, bdry_eps=None, rxn_factor=1.0,
          seed=0):
    """
    Train the constitutive NN on a single static strain field.

    Parameters
    ----------
    eps    : (M, M, 3) numpy array  -- strain field [e11, e22, e12]
    X      : (M, M, 2) numpy array  -- evaluation grid coordinates
    DX     : float                  -- grid spacing
    K      : int                    -- number of test functions
    epochs : int
    lr     : float                  -- Adam learning rate
    hidden : int                    -- hidden units per layer
    layers : int                    -- number of hidden layers
    seed   : int

    Returns
    -------
    model  : trained ConstitutiveNN
    losses : list of float (one per epoch)
    """
    def _fmt_sec(sec):
        sec = max(0, int(sec))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    # Precompute test function gradients (done once)
    print(f"Building {K} test functions...")
    test_fns = make_test_functions(X, K=K, seed=seed)
    grad_psis = [tf['grad_psi'] for tf in test_fns]

    # Convert strain to torch (fixed, not a parameter)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

    # Model and optimizer
    torch.manual_seed(seed)
    model = ConstitutiveNN(hidden=hidden, layers=layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bdry_t = (torch.tensor(bdry_eps, dtype=torch.float32, device=device)
              if bdry_eps is not None else None)

    losses = []
    t_start = time.perf_counter()
    print(f"Training for {epochs} epochs...")
    print(
        "Loss reporting: raw = full objective; "
        "loss/K ≈ average squared weak residual per test function (rough scale); "
        "log10(raw) for exponent-style reading."
    )
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = pde_loss(model, eps_t, grad_psis, DX)
        if reaction_force is not None and bdry_t is not None:
            loss = loss + rxn_factor * boundary_reaction_loss(
                model, bdry_t, reaction_force, DX
            )
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if epoch % 50 == 0 or epoch == epochs - 1:
            elapsed = time.perf_counter() - t_start
            done = epoch + 1
            avg_epoch = elapsed / done
            eta = avg_epoch * (epochs - done)
            pct = 100.0 * done / max(epochs, 1)
            loss_per_k = loss_val / max(K, 1)
            log10_loss = math.log10(max(loss_val, 1e-300))
            print(
                f"  Epoch {epoch:4d}/{epochs-1:4d} ({pct:5.1f}%)"
                f"  loss = {loss_val:.4e}  loss/K = {loss_per_k:.4e}"
                f"  log10(loss) = {log10_loss:6.3f}"
                f"  elapsed = {_fmt_sec(elapsed)}  ETA = {_fmt_sec(eta)}"
            )

    print("Done.")
    model = model.to('cpu')
    return model, losses
