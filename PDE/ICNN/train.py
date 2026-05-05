"""
train.py  —  NN-EUCLID training protocol  (Thakolkaran et al. 2022)
--------------------------------------------------------------------
Phase A — Adam + CyclicLR (paper-style exploration; max LR reduced
          from 0.1 when that destabilises stiff lattices).
Phase B — L-BFGS refinement on the best Adam weights (eval mode,
          dropout off).

Reaction weight ``w_rxn`` is set once per run via ``initial_w_rxn`` so
``w_rxn * loss_rxn`` matches ``loss_free`` at random init (see nodal_loss).

Saves BEST checkpoint across both phases (not just final step).
"""

import math
import os
import copy
import time
import torch
import torch.optim as optim
import numpy as np

from nn_model import ICNN
from nodal_loss import nodal_loss, initial_w_rxn


def _fmt(sec):
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def train_single(eps_t, DX, F_reaction_y,
                 hidden=64, layers=3, act_scale=1/12, dropout=0.2,
                 adam_epochs=600, base_lr=0.001, max_lr=0.01, cycle_steps=50,
                 lbfgs_epochs=400, lbfgs_lr=0.1,
                 lbfgs_max_iter=30,
                 lbfgs_tol_grad=1e-10, lbfgs_tol_change=1e-12,
                 seed=0, verbose=True):
    """
    One training run: cyclic Adam, then optional L-BFGS.

    Returns
    -------
    model, losses, w_rxn
        ``losses`` concatenates Adam epochs then L-BFGS steps (one scalar
        per outer iteration). ``w_rxn`` is fixed from init scaling.
    """
    device = eps_t.device
    torch.manual_seed(seed)

    model = ICNN(hidden=hidden, layers=layers,
                 act_scale=act_scale, dropout=dropout).to(device)

    w_rxn, lf0, lr0 = initial_w_rxn(model, eps_t, DX, F_reaction_y)
    if verbose:
        print(f"    init scales: loss_free={lf0:.4e}  loss_rxn={lr0:.4e}  "
              f"-> w_rxn={w_rxn:.4e}")

    losses = []
    best_loss = float('inf')
    best_state = None

    def maybe_best(lv):
        nonlocal best_loss, best_state
        if math.isfinite(lv) and lv < best_loss:
            best_loss = lv
            best_state = copy.deepcopy(model.state_dict())

    # ----- Phase A: Adam + CyclicLR -----
    opt = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = optim.lr_scheduler.CyclicLR(
        opt, base_lr=base_lr, max_lr=max_lr,
        step_size_up=cycle_steps, step_size_down=cycle_steps,
        cycle_momentum=False)

    t0 = time.perf_counter()
    total_steps = adam_epochs + max(lbfgs_epochs, 0)

    for epoch in range(adam_epochs):
        model.train()
        opt.zero_grad()
        loss, lf, lr_ = nodal_loss(model, eps_t, DX, F_reaction_y, w_rxn=w_rxn)
        loss.backward()
        opt.step()
        scheduler.step()

        lv = loss.item()
        losses.append(lv)
        maybe_best(lv)

        if verbose and (epoch % 100 == 0 or epoch == adam_epochs - 1):
            elapsed = time.perf_counter() - t0
            done = epoch + 1
            eta = elapsed / done * (total_steps - done) if done else 0.0
            lr_now = opt.param_groups[0]['lr']
            print(
                f"    adam {epoch:4d}/{adam_epochs-1}  "
                f"loss={lv:.4e}  free={lf.item():.4e}  rxn={lr_.item():.4e}  "
                f"best={best_loss:.4e}  lr={lr_now:.4f}  "
                f"{_fmt(int(elapsed))}<{_fmt(int(eta))}"
            )

    model.load_state_dict(best_state)

    # ----- Phase B: L-BFGS -----
    if lbfgs_epochs > 0:
        model.eval()
        ls_fn = None if device.type == 'cuda' else 'strong_wolfe'
        lbfgs_aborted = False

        for attempt in range(2):
            lr_cur = lbfgs_lr * (0.5 ** attempt)
            if attempt > 0:
                model.load_state_dict(best_state)
                if verbose:
                    print(f"    --- L-BFGS retry (attempt {attempt+1}/2) "
                          f"lr={lr_cur} ---")
            elif verbose:
                print(f"    --- L-BFGS ({lbfgs_epochs} steps, lr={lr_cur}, "
                      f"max_iter={lbfgs_max_iter}, tol_grad={lbfgs_tol_grad:g}, "
                      f"tol_change={lbfgs_tol_change:g}) ---")

            opt_bfgs = optim.LBFGS(
                model.parameters(),
                lr=lr_cur,
                max_iter=lbfgs_max_iter,
                tolerance_grad=lbfgs_tol_grad,
                tolerance_change=lbfgs_tol_change,
                line_search_fn=ls_fn,
            )
            lbfgs_aborted = False

            for k in range(lbfgs_epochs):
                def closure():
                    opt_bfgs.zero_grad()
                    loss, _, _ = nodal_loss(model, eps_t, DX, F_reaction_y,
                                            w_rxn=w_rxn)
                    loss.backward()
                    return loss

                opt_bfgs.step(closure)
                with torch.no_grad():
                    lv = nodal_loss(model, eps_t, DX, F_reaction_y,
                                    w_rxn=w_rxn)[0].item()

                if not math.isfinite(lv):
                    if verbose:
                        print(
                            f"    lbfgs {k:4d}/{lbfgs_epochs-1}  "
                            f"non-finite loss — reverting to best weights"
                            + ("; retrying at half lr."
                               if attempt == 0 else
                               ". Stopping L-BFGS (try --lbfgs-lr 0.05-0.1)."))
                    model.load_state_dict(best_state)
                    losses.append(best_loss)
                    lbfgs_aborted = True
                    break

                losses.append(lv)
                maybe_best(lv)

                if verbose and (k % 20 == 0 or k == lbfgs_epochs - 1):
                    elapsed = time.perf_counter() - t0
                    done = adam_epochs + k + 1
                    eta = elapsed / done * (total_steps - done) if done else 0.0
                    print(
                        f"    lbfgs {k:4d}/{lbfgs_epochs-1}  "
                        f"loss={lv:.4e}  best={best_loss:.4e}  "
                        f"{_fmt(int(elapsed))}<{_fmt(int(eta))}"
                    )

            if not lbfgs_aborted:
                break

    model.load_state_dict(best_state)
    model.eval()
    model = model.to('cpu')
    return model, losses, w_rxn


def train_ensemble(eps, DX, F_reaction_y,
                   hidden=64, layers=3, act_scale=1/12, dropout=0.2,
                   adam_epochs=600, base_lr=0.001, max_lr=0.01, cycle_steps=50,
                   lbfgs_epochs=400, lbfgs_lr=0.1,
                   lbfgs_max_iter=30,
                   lbfgs_tol_grad=1e-10, lbfgs_tol_change=1e-12,
                   n_ensemble=5,
                   checkpoint_dir=None):
    """
    Train n_ensemble independent models, return the one with lowest
    best-seen loss across all runs.

    Returns
    -------
    best_model, all_losses, best_idx, w_rxn_best
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running {n_ensemble} ensemble members")
    print(f"  Phase A: Adam {adam_epochs} ep, CyclicLR {base_lr:g}->{max_lr:g}, "
          f"cycle step up/down={cycle_steps}")
    if lbfgs_epochs > 0:
        print(f"  Phase B: L-BFGS {lbfgs_epochs} steps, lr={lbfgs_lr}, "
              f"max_iter={lbfgs_max_iter}, tol_grad={lbfgs_tol_grad:g}, "
              f"tol_change={lbfgs_tol_change:g}")
    else:
        print("  Phase B: (disabled)")
    print(f"Architecture: hidden={hidden}, layers={layers}, dropout={dropout}")
    print(f"Loss: nodal force balance + slice resultant (all rows); "
          f"w_rxn from init loss ratio (per run)")
    print()

    eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    all_losses = []
    best_losses = []
    models = []
    all_w_rxn = []

    for run in range(n_ensemble):
        print(f"--- Ensemble run {run+1}/{n_ensemble}  (seed={run}) ---")
        model, losses, w_rxn = train_single(
            eps_t, DX, F_reaction_y,
            hidden=hidden, layers=layers,
            act_scale=act_scale, dropout=dropout,
            adam_epochs=adam_epochs,
            base_lr=base_lr, max_lr=max_lr, cycle_steps=cycle_steps,
            lbfgs_epochs=lbfgs_epochs, lbfgs_lr=lbfgs_lr,
            lbfgs_max_iter=lbfgs_max_iter,
            lbfgs_tol_grad=lbfgs_tol_grad,
            lbfgs_tol_change=lbfgs_tol_change,
            seed=run,
            verbose=True,
        )
        all_losses.append(losses)
        best_losses.append(min(losses))
        models.append(model)
        all_w_rxn.append(w_rxn)

        if checkpoint_dir:
            path = os.path.join(checkpoint_dir, f'run_{run:02d}.pt')
            _save(model, losses, hidden, layers, act_scale, dropout, run, path,
                  w_rxn=w_rxn)
            print(f"    saved -> {path}")

        print(f"    best loss = {min(losses):.4e}  "
              f"final loss = {losses[-1]:.4e}  w_rxn = {w_rxn:.4e}\n")

    best_idx = int(np.argmin(best_losses))
    best_model = models[best_idx]
    w_rxn_best = all_w_rxn[best_idx]
    print(f"Best run: {best_idx+1}/{n_ensemble}  "
          f"(best loss={best_losses[best_idx]:.4e}, w_rxn={w_rxn_best:.4e})")

    threshold = best_losses[best_idx] * 1.20
    accepted = [i for i, l in enumerate(best_losses) if l <= threshold]
    print(f"Accepted (within 20% of best): runs {[i+1 for i in accepted]}  "
          f"losses: {[f'{best_losses[i]:.3e}' for i in accepted]}")

    return best_model, all_losses, best_idx, w_rxn_best


def _save(model, losses, hidden, layers, act_scale, dropout, seed, path,
          w_rxn=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        'state_dict': model.state_dict(),
        'hidden': hidden,
        'layers': layers,
        'act_scale': act_scale,
        'dropout': dropout,
        'losses': losses,
        'seed': seed,
    }
    if w_rxn is not None:
        payload['w_rxn'] = w_rxn
    try:
        torch.save(payload, path)
    except Exception as e:
        print(f"WARNING: save failed for {path}: {e}")
