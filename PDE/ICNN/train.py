"""
train.py  (ICNN)
----------------
Training script for static equilibrium using the ICNN constitutive model.

Improvements over MLP/train.py (inspired by nn-EUCLID):
  optimizer    : 'adam' or 'lbfgs' (L-BFGS with strong-Wolfe line search)
  lr_schedule  : 'cyclic' (CyclicLR) or 'constant'
  reaction_force / bdry_eps : optional boundary traction loss that pins the
                              absolute stress scale (nn-EUCLID: reaction_loss)
"""

import math
import numpy as np
import time
import torch
import torch.optim as optim

from nn_model import ICNN
from pde_loss import make_test_functions, pde_loss, boundary_reaction_loss


def train(eps, X, DX,
          K=50, epochs=2000,
          hidden=32, layers=3,
          act_scale=1/12, dropout=0.0,
          optimizer='adam',
          base_lr=1e-3, max_lr=1e-2,
          lr_schedule='cyclic', cycle_steps=200,
          reaction_force=None, bdry_eps=None, rxn_factor='auto',
          checkpoint_path=None, checkpoint_every=200,
          seed=0):
    """
    Parameters
    ----------
    eps           : (M, M, 3)   strain field [e11, e22, e12]
    X             : (M, M, 2)   evaluation grid coordinates
    DX            : float       grid spacing
    K             : int         number of test functions
    epochs        : int
    hidden        : int         hidden units per layer
    layers        : int         number of hidden layers
    act_scale     : float       squared-softplus scaling (nn-EUCLID: 1/12)
    dropout       : float       dropout probability (0 = off)
    optimizer     : 'adam' | 'lbfgs'
    base_lr       : float       base / initial learning rate
    max_lr        : float       peak LR for cyclic schedule (Adam only)
    lr_schedule   : 'cyclic' | 'constant'  (ignored for lbfgs)
    cycle_steps   : int         half-cycle length for CyclicLR
    reaction_force: float|None  total vertical reaction at bottom boundary
    bdry_eps      : (M,3)|None  strain at top row of eval grid
    rxn_factor    : float|'auto'  weight on the reaction force loss term.
                                  'auto' uses fixed 1e5 — chosen so that the
                                  reaction loss (~O(1e3) at start) is comparable
                                  to the weak form loss (~O(1e8) at start).
    seed          : int

    Returns
    -------
    model  : trained ICNN
    losses : list of float (one per epoch)
    """
    def _fmt_sec(sec):
        sec = max(0, int(sec))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    print(f"Building {K} test functions...")
    test_fns  = make_test_functions(X, K=K, seed=seed)
    grad_psis = [tf['grad_psi'] for tf in test_fns]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

    # Pre-convert boundary strain once if provided
    bdry_t = (torch.tensor(bdry_eps, dtype=torch.float32, device=device)
              if bdry_eps is not None else None)

    torch.manual_seed(seed)
    model = ICNN(hidden=hidden, layers=layers,
                 act_scale=act_scale, dropout=dropout).to(device)

    # ----- optimizer -----
    if optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=base_lr)
    elif optimizer == 'lbfgs':
        opt = optim.LBFGS(model.parameters(), lr=base_lr,
                          line_search_fn='strong_wolfe')
    else:
        raise ValueError(f"optimizer must be 'adam' or 'lbfgs', got '{optimizer}'")

    # ----- LR scheduler (Adam only; L-BFGS has its own line search) -----
    scheduler = None
    if optimizer == 'adam' and lr_schedule == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=cycle_steps,
            step_size_down=cycle_steps,
            cycle_momentum=False,
        )

    # Fixed rxn_factor = 1e5.
    # Rationale: weak form loss starts at ~O(1e8), reaction loss starts at
    # ~O(1e3) for a unit-scale force. Multiplying by 1e5 brings reaction loss
    # to ~O(1e8), making both terms comparable from the first epoch.
    # Pass rxn_factor explicitly to override.
    if rxn_factor == 'auto':
        rxn_factor = 1e5
        print(f"rxn_factor = {rxn_factor:.3e}  (fixed default)")

    def compute_loss():
        loss = pde_loss(model, eps_t, grad_psis, DX)
        if reaction_force is not None and bdry_t is not None:
            rxn_loss = boundary_reaction_loss(model, bdry_t, reaction_force, DX)
            loss = loss + rxn_factor * rxn_loss
        return loss

    losses = []
    t_start = time.perf_counter()
    print(f"Training for {epochs} epochs  [{optimizer}"
          + (f", {lr_schedule} LR" if optimizer == 'adam' else "")
          + (f", reaction loss w/ rxn_factor={rxn_factor:.1e}"
             if reaction_force is not None else "")
          + "] ...")
    print(
        "Loss reporting: raw = full objective; "
        "loss/K ≈ average squared weak residual per test function (rough scale); "
        "log10(raw) for exponent-style reading."
    )

    for epoch in range(epochs):
        model.train()

        if optimizer == 'lbfgs':
            def closure():
                opt.zero_grad()
                loss = compute_loss()
                loss.backward()
                return loss
            loss_val_t = opt.step(closure)
            loss_val   = loss_val_t.item()
        else:
            opt.zero_grad()
            loss_val_t = compute_loss()
            loss_val_t.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            loss_val = loss_val_t.item()

        losses.append(loss_val)

        if epoch % 50 == 0 or epoch == epochs - 1:
            lr_now = opt.param_groups[0]['lr']
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
                f"  lr = {lr_now:.2e}"
                f"  elapsed = {_fmt_sec(elapsed)}  ETA = {_fmt_sec(eta)}"
            )

        if (checkpoint_path is not None
                and checkpoint_every > 0
                and (epoch + 1) % checkpoint_every == 0):
            _cpu = model.to('cpu')
            torch.save({'state_dict': _cpu.state_dict(),
                        'epoch': epoch + 1,
                        'losses': losses}, checkpoint_path)
            model = _cpu.to(device)
            print(f"  [checkpoint saved → {checkpoint_path}  epoch {epoch+1}]")

    print("Done.")
    model = model.to('cpu')
    return model, losses
