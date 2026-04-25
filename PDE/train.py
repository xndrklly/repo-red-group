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

import numpy as np
import torch
import torch.optim as optim

from nn_model import ConstitutiveNN
from pde_loss import make_test_functions, pde_loss


def train(eps, X, DX,
          K=50, epochs=2000, lr=1e-3,
          hidden=64, layers=3, seed=0):
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
    # Precompute test function gradients (done once)
    print(f"Building {K} test functions...")
    test_fns = make_test_functions(X, K=K, seed=seed)
    grad_psis = [tf['grad_psi'] for tf in test_fns]

    # Convert strain to torch (fixed, not a parameter)
    eps_t = torch.tensor(eps, dtype=torch.float32)

    # Model and optimizer
    model = ConstitutiveNN(hidden=hidden, layers=layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = pde_loss(model, eps_t, grad_psis, DX)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d}  loss = {loss_val:.4e}")

    print("Done.")
    return model, losses
