"""
train.py  —  NN-EUCLID training protocol  (Thakolkaran et al. 2022)
--------------------------------------------------------------------
Matches paper Appendix A exactly:
  - Adam optimizer
  - Learning rate linearly cycled 0.001 → 0.1 → 0.001 every 100 epochs
  - 500 epochs total
  - Dropout 0.2 during training, off at eval
  - Ensemble of n_ensemble independent runs; keep best by final loss

Loss: nodal force balance (eq. 22) — no rxn_factor needed.
"""

import math
import os
import time
import torch
import torch.optim as optim
import numpy as np

from nn_model import ICNN
from nodal_loss import nodal_loss


def _fmt(sec):
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def train_single(eps_t, DX, F_reaction_y,
                 hidden=64, layers=3, act_scale=1/12, dropout=0.2,
                 epochs=500, base_lr=0.001, max_lr=0.1, cycle_epochs=100,
                 seed=0, verbose=True):
    """
    One training run — matches nn-EUCLID Appendix A exactly.

    hidden=64, 3 hidden layers, dropout=0.2 per Table A.1 of paper.
    """
    device = eps_t.device
    torch.manual_seed(seed)

    model = ICNN(hidden=hidden, layers=layers,
                 act_scale=act_scale, dropout=dropout).to(device)

    opt = optim.Adam(model.parameters(), lr=base_lr)

    # Paper: "linearly cycled from 0.001 to 0.1 and back every 100 epochs"
    # step_size_up=50, step_size_down=50 → full cycle = 100 epochs
    scheduler = optim.lr_scheduler.CyclicLR(
        opt,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=cycle_epochs // 2,
        step_size_down=cycle_epochs // 2,
        cycle_momentum=False,
        mode='triangular',
    )

    losses = []
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()

        loss, lf, lr_ = nodal_loss(model, eps_t, DX, F_reaction_y)
        loss.backward()
        opt.step()
        scheduler.step()

        lv = loss.item()
        losses.append(lv)

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            elapsed = time.perf_counter() - t0
            eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
            lr_now = opt.param_groups[0]['lr']
            print(
                f"    ep {epoch:4d}/{epochs-1}  "
                f"loss={lv:.4e}  free={lf.item():.4e}  rxn={lr_.item():.4e}  "
                f"lr={lr_now:.4f}  {_fmt(elapsed)}<{_fmt(eta)}"
            )

    model.eval()
    model = model.to('cpu')
    return model, losses


def train_ensemble(eps, DX, F_reaction_y,
                   hidden=64, layers=3, act_scale=1/12, dropout=0.2,
                   epochs=500, base_lr=0.001, max_lr=0.1, cycle_epochs=100,
                   n_ensemble=5,
                   checkpoint_dir=None):
    """
    Train n_ensemble independent models, return the best one.

    Paper uses n_e=30; we default to 5 for practical GPU training.
    Each run uses a different random seed for weight initialisation.
    Acceptance criterion: within 20% of best final loss (per paper).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running {n_ensemble} ensemble members, {epochs} epochs each")
    print(f"Architecture: hidden={hidden}, layers={layers}, dropout={dropout}")
    print(f"LR: {base_lr} -> {max_lr} -> {base_lr}, cycle={cycle_epochs} epochs")
    print(f"Loss: nodal force balance (nn-EUCLID eq. 22, no rxn_factor)")
    print()

    eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    all_losses = []
    final_losses = []
    models = []

    for run in range(n_ensemble):
        print(f"--- Ensemble run {run+1}/{n_ensemble}  (seed={run}) ---")
        model, losses = train_single(
            eps_t, DX, F_reaction_y,
            hidden=hidden, layers=layers,
            act_scale=act_scale, dropout=dropout,
            epochs=epochs,
            base_lr=base_lr, max_lr=max_lr, cycle_epochs=cycle_epochs,
            seed=run,
            verbose=True,
        )
        all_losses.append(losses)
        final_losses.append(losses[-1])
        models.append(model)

        if checkpoint_dir:
            path = os.path.join(checkpoint_dir, f'run_{run:02d}.pt')
            _save(model, losses, hidden, layers, act_scale, dropout, run, path)
            print(f"    saved -> {path}")

        print(f"    final loss = {losses[-1]:.4e}\n")

    best_idx = int(np.argmin(final_losses))
    best_model = models[best_idx]
    print(f"Best run: {best_idx+1}/{n_ensemble}  "
          f"(loss={final_losses[best_idx]:.4e})")

    threshold = final_losses[best_idx] * 1.20
    accepted = [i for i, l in enumerate(final_losses) if l <= threshold]
    print(f"Accepted (within 20% of best): runs {[i+1 for i in accepted]}  "
          f"losses: {[f'{final_losses[i]:.3e}' for i in accepted]}")

    return best_model, all_losses, best_idx


def _save(model, losses, hidden, layers, act_scale, dropout, seed, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        torch.save({
            'state_dict': model.state_dict(),
            'hidden': hidden, 'layers': layers,
            'act_scale': act_scale, 'dropout': dropout,
            'losses': losses, 'seed': seed,
        }, path)
    except Exception as e:
        print(f"WARNING: save failed for {path}: {e}")
