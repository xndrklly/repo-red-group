"""
train.py  (ICNN_weak)
---------------------
Two-phase training: Adam warm-start -> L-BFGS refinement.
Known issues documented in README.md.

See ../ICNN/train.py for the cleaner nn-EUCLID implementation.
"""

import math, os, time
import torch
import torch.optim as optim
from nn_model import ICNN
from pde_loss import make_test_functions, pde_loss, boundary_reaction_loss


def _fmt(sec):
    sec = max(0, int(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _save_ckpt(model, losses, hidden, layers, act_scale, dropout, epoch, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        torch.save({'state_dict': model.state_dict(), 'hidden': hidden,
                    'layers': layers, 'act_scale': act_scale,
                    'dropout': dropout, 'epoch': epoch, 'losses': losses}, path)
        print(f"  [checkpoint -> {path}  epoch {epoch}]")
    except Exception as e:
        print(f"  [WARNING: checkpoint save FAILED: {e}]")


def train(eps, X, DX,
          K=50, epochs=2000,
          hidden=32, layers=3, act_scale=1/12, dropout=0.0,
          optimizer='adam', base_lr=1e-3, max_lr=1e-2,
          lr_schedule='cyclic', cycle_steps=200,
          reaction_force=None, bdry_eps=None, rxn_factor='auto',
          checkpoint_path=None, checkpoint_every=200,
          init_state_dict=None, seed=0):

    def compute_loss():
        loss = pde_loss(model, eps_t, grad_psis_t, DX)
        if reaction_force is not None and bdry_t is not None:
            loss = loss + rxn_factor * boundary_reaction_loss(model, bdry_t, reaction_force, DX)
        return loss

    print(f"Building {K} test functions...")
    grad_psis = [tf['grad_psi'] for tf in make_test_functions(X, K=K, seed=seed)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    eps_t = torch.tensor(eps, dtype=torch.float32, device=device)
    grad_psis_t = [torch.tensor(gp, dtype=torch.float32, device=device) for gp in grad_psis]
    bdry_t = torch.tensor(bdry_eps, dtype=torch.float32, device=device) if bdry_eps is not None else None

    torch.manual_seed(seed)
    model = ICNN(hidden=hidden, layers=layers, act_scale=act_scale, dropout=dropout)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
        print("  [loaded warm-start weights]")
    model = model.to(device)

    if optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = optim.lr_scheduler.CyclicLR(
            opt, base_lr=base_lr, max_lr=max_lr,
            step_size_up=cycle_steps, step_size_down=cycle_steps,
            cycle_momentum=False) if lr_schedule == 'cyclic' else None
    elif optimizer == 'lbfgs':
        opt = optim.LBFGS(model.parameters(), lr=base_lr, line_search_fn='strong_wolfe')
        scheduler = None
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    if rxn_factor == 'auto':
        rxn_factor = 1e5
        print(f"rxn_factor = {rxn_factor:.3e}  (fixed default)")

    losses = []; t0 = time.perf_counter()
    print(f"Training {epochs} epochs [{optimizer}"
          + (f", reaction loss rxn_factor={rxn_factor:.1e}" if reaction_force is not None else "")
          + "] ...")

    for epoch in range(epochs):
        model.train()
        if optimizer == 'lbfgs':
            def closure():
                opt.zero_grad(); loss = compute_loss(); loss.backward(); return loss
            loss_val = opt.step(closure).item()
        else:
            opt.zero_grad(); lv = compute_loss(); lv.backward(); opt.step()
            if scheduler: scheduler.step()
            loss_val = lv.item()

        losses.append(loss_val)
        if epoch % 50 == 0 or epoch == epochs-1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (epoch+1) * (epochs-epoch-1)
            print(f"  Epoch {epoch:4d}/{epochs-1}  loss={loss_val:.4e}"
                  f"  log10={math.log10(max(loss_val,1e-300)):.2f}"
                  f"  lr={opt.param_groups[0]['lr']:.2e}"
                  f"  {_fmt(elapsed)}<{_fmt(eta)}")

        if checkpoint_path and checkpoint_every > 0 and (epoch+1) % checkpoint_every == 0:
            cpu = model.to('cpu')
            _save_ckpt(cpu, losses, hidden, layers, act_scale, dropout, epoch+1, checkpoint_path)
            model = cpu.to(device)

    print("Done.")
    return model.to('cpu'), losses
