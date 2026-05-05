"""
run.py  —  NN-EUCLID pipeline  (Thakolkaran et al. 2022)
---------------------------------------------------------
Training (see train.py):
  - Phase A: Adam + CyclicLR (defaults 0.001 <-> 0.01, triangular cycle)
  - Phase B: L-BFGS on best Adam weights (dropout off)
  - w_rxn from initial loss_free / loss_rxn (no fixed 1e6)
  - Reaction loss: mean σ_yy vs F/(M·DX) on **every** row (flat R_y(y))
  - Ensemble of 5 runs by default, keep best (within 20% acceptance)

Usage (from PDE/ICNN/ or any directory):
    python run.py
    python run.py --condition static --grid-size 100
    python run.py --data /abs/path/to/lattice_static.npz
    python run.py --data .../data/static/100_1N/lattice_static.npz   # spring_grid_static naming
    python run.py --load-model   # skip training, load saved best model
    python run.py --force-train  # retrain even if checkpoint exists

All paths resolved relative to script location — works from any cwd.
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from smoothing import GaussianSmoother, make_eval_grid
from train import train_ensemble, _save
from nodal_loss import nodal_loss, DEFAULT_W_RXN
from nn_model import ICNN


# ------------------------------------------------------------------ helpers --

def load_data(path):
    d = np.load(path)
    if 'F_reaction_y' not in d:
        raise ValueError(f"Dataset {path} missing F_reaction_y")
    return (d['node_pos'], d['u'],
            float(d['DX']), float(d['F_reaction_y']))


def smooth(node_pos, u, DX, margin=3):
    smoother  = GaussianSmoother(node_pos, u, h=2.0 * DX)
    eval_grid = make_eval_grid(node_pos, downsample=1, margin=margin)
    u_hat, eps = smoother.compute_vectorised(eval_grid)
    return eval_grid, u_hat, eps


def resolve_data_path(data_arg, condition, grid_size):
    if data_arg:
        return os.path.abspath(data_arg)
    rel = os.path.join(SCRIPT_DIR, '..', '..', 'data',
                       condition, str(grid_size), f'lattice_{condition}.npz')
    return os.path.abspath(rel)


def infer_tags(data_path):
    parts = os.path.normpath(data_path).split(os.sep)
    if len(parts) >= 3 and parts[-3] == 'static':
        return 'lattice_static', parts[-2]
    stem = os.path.splitext(os.path.basename(data_path))[0]
    return stem, (parts[-2] if len(parts) >= 2 else 'unknown')


def prompt(msg, default):
    v = input(f"{msg} [{default}]: ").strip()
    return v if v else str(default)


def opath(results_dir, fname):
    return os.path.join(results_dir, fname)


def _eval_grid_spacing(eval_grid):
    """Mean Δx along a row and mean Δy along a column (uniform grid)."""
    dx = float(np.mean(np.diff(eval_grid[0, :, 0])))
    dy = float(np.mean(np.diff(eval_grid[:, 0, 1])))
    return dx, dy


def _check_dx_dy_vs_dataset(DX, eval_grid, rtol=1e-4):
    """
    ``nodal_loss`` uses ``DX`` from the dataset for σ_target and finite
    differences. Diagnostics should use the same spacing for ∫σ_yy dx.
    """
    dx_g, dy_g = _eval_grid_spacing(eval_grid)
    print(f'  Grid spacing: Δx={dx_g:.6g}  Δy={dy_g:.6g}  (dataset DX={DX:.6g})')
    for name, v in ('Δx', dx_g), ('Δy', dy_g):
        if abs(v - DX) > rtol * max(abs(DX), 1e-30):
            print(f'  WARNING: {name} differs from DX by '
                  f'{abs(v - DX) / max(abs(DX), 1e-30):.2e} (rtol={rtol}); '
                  f'vertical resultant uses DX to match the loss.')
    return dx_g, dy_g


# -------------------------------------------------------------------- plots --

def plot_loss(all_losses, best_idx, out):
    """All ensemble runs on one plot, best highlighted."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, losses in enumerate(all_losses):
        lw    = 2.5 if i == best_idx else 0.8
        alpha = 1.0 if i == best_idx else 0.4
        label = f'run {i+1} (BEST)' if i == best_idx else f'run {i+1}'
        ax.semilogy(losses, lw=lw, alpha=alpha, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total loss (log scale)')
    ax.set_title('Ensemble training — nodal force balance loss')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved  {out}')


def plot_loss_breakdown(model, eps_static, DX, F_reaction_y, all_losses, best_idx, out,
                        w_rxn=None):
    """
    Two-panel plot showing the free-node and reaction contributions separately
    for the best run, plus the final numerical values.

    This is the PRIMARY diagnostic for whether training worked:
      - loss_free should be driving toward zero (equilibrium satisfied)
      - loss_rxn  should be near zero (mean σ_yy on every row vs F/(M·DX))
      - If loss_free is low but loss_rxn is high: stress scale / flat R_y off
      - If loss_rxn is low but loss_free is high: stress is non-equilibrated
      - If both are low: training succeeded
    """
    device = torch.device('cpu')
    eps_t  = torch.tensor(eps_static, dtype=torch.float32, device=device)

    w = float(DEFAULT_W_RXN if w_rxn is None else w_rxn)
    model.eval()
    with torch.no_grad():
        _, lf, lr = nodal_loss(model, eps_t, DX, F_reaction_y, w_rxn=w)
    lf_val = lf.item()
    lr_val = lr.item()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: loss breakdown bar chart
    ax = axes[0]
    bars = ax.bar(['Interior\nequilibrium\n(loss_free)',
                   'Flat slice\nresultant\n(loss_rxn)'],
                  [lf_val, lr_val],
                  color=['steelblue', 'tomato'], width=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('Loss contribution (log scale)')
    ax.set_title('Final loss breakdown')
    for bar, val in zip(bars, [lf_val, lr_val]):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.5,
                f'{val:.3e}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # Panel 2: best run loss curve split into free vs rxn if available,
    # otherwise just total
    ax2 = axes[1]
    losses = all_losses[best_idx]
    ax2.semilogy(losses, lw=1.5, color='steelblue', label='total loss (best run)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title(f'Best run (run {best_idx+1}) training curve')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=8)

    total_train = lf_val + w * lr_val
    fig.suptitle(
        f'Training diagnostics  |  '
        f'free={lf_val:.3e}  rxn={lr_val:.3e}  '
        f'w_rxn={w:.3e}  total={total_train:.3e}',
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out}  (free={lf_val:.3e}, rxn={lr_val:.3e})')


def plot_stress_field(model, eps_static, eval_grid, out):
    """
    Spatial maps of all three stress components.
    Good training → smooth fields consistent with the boundary conditions
    (σ_yy roughly uniform in x, varying smoothly in y).
    Bad training → noisy, near-zero, or physically implausible fields.
    """
    M = eval_grid.shape[0]
    model.eval()
    with torch.no_grad():
        sigma_f = model(torch.tensor(
            eps_static.reshape(-1, 3), dtype=torch.float32)).numpy()
    sigma = sigma_f.reshape(M, M, 3)
    titles = [r'$\sigma_{11}$', r'$\sigma_{22}$', r'$\sigma_{12}$']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, (ax, title) in enumerate(zip(axes, titles)):
        field = sigma[:, :, i]
        vmax  = np.abs(field).max()
        im = ax.imshow(field, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('j  (x)'); ax.set_ylabel('i  (y)')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Predicted stress fields', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out}')


def plot_stress_strain(model, eps_static, out):
    """σ-ε curves sweeping each strain component independently."""
    model.eval()
    e_max   = float(np.abs(eps_static).max())
    e_range = np.linspace(-1.5*e_max, 1.5*e_max, 400)
    comp_labels = [r'$\varepsilon_{11}$', r'$\varepsilon_{22}$', r'$\varepsilon_{12}$']
    sig_labels  = [r'$\sigma_{11}$', r'$\sigma_{22}$', r'$\sigma_{12}$']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for comp in range(3):
        ax = axes[comp]
        e  = np.zeros((len(e_range), 3));  e[:, comp] = e_range
        with torch.no_grad():
            sigma = model(torch.tensor(e, dtype=torch.float32)).numpy()
        for s in range(3):
            ax.plot(e_range, sigma[:, s], color=colors[s],
                    label=sig_labels[s], lw=1.8)
        # Small-strain slope of the *driven* stress–strain pair (not always σ11)
        mask = np.abs(e_range) <= 0.1 * e_max
        if mask.sum() > 3:
            slope = np.polyfit(e_range[mask], sigma[mask, comp], 1)[0]
            ax.plot(e_range, slope * e_range, 'k--', lw=1.2,
                    label=fr'small-strain slope $\approx$ {slope:.3f}')
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax.set_xlabel(comp_labels[comp]); ax.set_ylabel('stress')
        ax.set_title(f'Sweep {comp_labels[comp]}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Constitutive law: ICNN  σ–ε  curves', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out}')


def plot_pde_residual(model, eps_static, eval_grid, out):
    """Pointwise div σ — should be near zero everywhere in interior."""
    M  = eval_grid.shape[0]
    dx = float(eval_grid[0, 1, 0] - eval_grid[0, 0, 0])
    dy = float(eval_grid[1, 0, 1] - eval_grid[0, 0, 1])
    model.eval()
    with torch.no_grad():
        sigma_f = model(torch.tensor(
            eps_static.reshape(-1, 3), dtype=torch.float32)).numpy()
    sigma = sigma_f.reshape(M, M, 3)
    s11, s22, s12 = sigma[:,:,0], sigma[:,:,1], sigma[:,:,2]
    div_x   = np.gradient(s11, dx, axis=1) + np.gradient(s12, dy, axis=0)
    div_y   = np.gradient(s12, dx, axis=1) + np.gradient(s22, dy, axis=0)
    div_mag = np.sqrt(div_x**2 + div_y**2)
    vmax_mag = np.percentile(div_mag[2:-2, 2:-2], 99)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field, title, cmap in zip(
        axes, [div_x, div_y, div_mag],
        [r'$(\nabla\!\cdot\!\sigma)_x$',
         r'$(\nabla\!\cdot\!\sigma)_y$',
         r'$|\nabla\!\cdot\!\sigma|$ (99th pct)'],
        ['RdBu_r', 'RdBu_r', 'hot_r']
    ):
        if cmap == 'RdBu_r':
            vmax = np.percentile(np.abs(field[2:-2,2:-2]), 99)
            im = ax.imshow(field, origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(field, origin='lower', cmap=cmap, vmin=0, vmax=vmax_mag)
        ax.set_title(title, fontsize=9); ax.set_xlabel('j'); ax.set_ylabel('i')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(r'Pointwise PDE residual  $\nabla\!\cdot\!\sigma$  (should be $\approx 0$)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out}')


def plot_vertical_resultant(model, eps_static, eval_grid, F_reaction_y, out,
                            row_sum_dx=None):
    """
    ∫ σ_yy dx vs height. At convergence should be flat and equal to F_reaction_y.
    The red dashed line shows the target reaction force.
    CV < 0.05 means the resultant is flat — global equilibrium satisfied.

    ``row_sum_dx`` should match ``DX`` in ``nodal_loss`` (dataset nominal
    spacing). If None, falls back to mean Δx from ``eval_grid``.
    """
    M = eval_grid.shape[0]
    if row_sum_dx is not None:
        dx = float(row_sum_dx)
    else:
        dx, _ = _eval_grid_spacing(eval_grid)
    model.eval()
    with torch.no_grad():
        sigma_f = model(torch.tensor(
            eps_static.reshape(-1, 3), dtype=torch.float32)).numpy()
    sigma = sigma_f.reshape(M, M, 3)
    Ry    = sigma[:,:,1].sum(axis=1) * dx
    y_row = eval_grid[:, 0, 1]

    interior = Ry[1:-1]
    cv  = float(np.std(interior) / max(abs(np.mean(interior)), 1e-12))
    err = float(Ry[0] - F_reaction_y)   # bottom-row error vs target

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(Ry, y_row, 'b.-', lw=1.5, markersize=4,
            label=r'$\int\sigma_{yy}\,dx$ (ICNN)')
    ax.axvline(F_reaction_y, color='r', ls='--', lw=2,
               label=f'target $F_y$ = {F_reaction_y:.4g}')
    ax.set_xlabel(r'$\int\sigma_{yy}\,dx$'); ax.set_ylabel('y')
    ax.set_title('Vertical stress resultant vs height')
    info = (f'CV = {cv:.3f} ({"good" if cv < 0.05 else "not flat"})\n'
            f'bottom-row error = {err:+.4f}')
    ax.annotate(info, xy=(0.98, 0.05), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.7))
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out}  (CV={cv:.3f}, bottom-row err={err:+.4f})')


def plot_smoothed_vs_original(node_pos, u_raw, eval_grid, u_hat, eps, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    def show(ax, data, title, cmap='viridis', sym=False):
        vmax = np.abs(data).max()
        im   = ax.imshow(data, origin='lower', cmap=cmap,
                         vmin=-vmax if sym else data.min(), vmax=vmax)
        ax.set_title(title); ax.set_xlabel('j'); ax.set_ylabel('i')
        fig.colorbar(im, ax=ax, shrink=0.75)
    show(axes[0,0], u_raw[:,:,0,1], 'Original $u_y$')
    show(axes[0,1], u_hat[:,:,0,1], 'Smoothed $u_y$')
    show(axes[1,0], eps[:,:,0,0],   r'$\varepsilon_{11}$', sym=True, cmap='RdBu_r')
    show(axes[1,1], eps[:,:,0,1],   r'$\varepsilon_{22}$')
    fig.suptitle('Displacement and strain fields', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out}')


# -------------------------------------------------------------------  main  --

def main():
    parser = argparse.ArgumentParser(description='NN-EUCLID ICNN pipeline')
    parser.add_argument('--condition', choices=['static','dynamic'], default=None)
    parser.add_argument('--grid-size', type=int, default=None)
    parser.add_argument('--data', default=None)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--force-train', action='store_true')
    parser.add_argument('--n-ensemble', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=600,
                        help='Adam (phase A) epochs')
    parser.add_argument('--max-lr', type=float, default=0.01,
                        help='CyclicLR peak learning rate')
    parser.add_argument('--cycle-steps', type=int, default=50,
                        help='CyclicLR step_size_up and step_size_down')
    parser.add_argument('--lbfgs-epochs', type=int, default=400,
                        help='L-BFGS outer steps (0 to skip phase B)')
    parser.add_argument('--lbfgs-lr', type=float, default=0.1,
                        help='L-BFGS step size (CUDA: try 0.05–0.15 if NaNs)')
    parser.add_argument('--lbfgs-max-iter', type=int, default=30,
                        help='L-BFGS max_iter per outer step')
    parser.add_argument('--lbfgs-tol-grad', type=float, default=1e-10,
                        help='L-BFGS gradient tolerance (inf norm)')
    parser.add_argument('--lbfgs-tol-change', type=float, default=1e-12,
                        help='L-BFGS loss change tolerance')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f'CUDA: yes  |  GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('CUDA: no  |  using CPU')

    # Resolve data path
    condition = args.condition
    grid_size = args.grid_size
    if args.data is None:
        if condition is None:
            condition = prompt('Condition (static/dynamic)', 'static').lower()
        if grid_size is None:
            grid_size = int(prompt('Grid size', 100))
    data_path    = resolve_data_path(args.data, condition, grid_size)
    cond_tag, grid_tag = infer_tags(data_path)

    results_dir = os.path.join(SCRIPT_DIR, 'results', cond_tag, str(grid_tag))
    os.makedirs(results_dir, exist_ok=True)
    print(f'Results dir: {results_dir}')

    model_path = (os.path.abspath(args.model_path) if args.model_path
                  else os.path.join(results_dir, 'icnn_model_best.pt'))

    # Load data
    print(f'Loading {data_path} ...')
    node_pos, u, DX, F_reaction_y = load_data(data_path)
    print(f'  F_reaction_y = {F_reaction_y:.6g}')

    # Smooth
    print('Smoothing ...')
    eval_grid, u_hat, eps = smooth(node_pos, u, DX, margin=3)
    eps_static = eps[:, :, 0, :]
    np.savez(opath(results_dir, 'smoothed_data.npz'),
             eval_grid=eval_grid, u_hat=u_hat, eps=eps)
    print(f'Saved  {opath(results_dir, "smoothed_data.npz")}')
    _check_dx_dy_vs_dataset(DX, eval_grid)

    # Architecture — paper Table A.1: 64 neurons, 3 hidden layers
    hidden, layers, act_scale, dropout = 64, 3, 1/12, 0.2

    # Train or load
    auto_load = os.path.exists(model_path) and not args.force_train
    use_load  = args.load_model or auto_load

    w_rxn_plot = float(DEFAULT_W_RXN)

    if use_load:
        print(f'Loading model from {model_path}')
        ckpt = torch.load(model_path, map_location='cpu')
        model = ICNN(hidden=ckpt['hidden'], layers=ckpt['layers'],
                     act_scale=ckpt.get('act_scale', 1/12),
                     dropout=ckpt.get('dropout', 0.2))
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        all_losses = [ckpt.get('losses', [])]
        best_idx   = 0
        w_rxn_plot = float(ckpt.get('w_rxn', DEFAULT_W_RXN))
        print('Loaded (training skipped)')
    else:
        ckpt_dir = os.path.join(results_dir, 'ensemble_checkpoints')
        best_model, all_losses, best_idx, w_rxn_best = train_ensemble(
            eps_static, DX, F_reaction_y,
            hidden=hidden, layers=layers,
            act_scale=act_scale, dropout=dropout,
            adam_epochs=args.epochs,
            base_lr=0.001,
            max_lr=args.max_lr,
            cycle_steps=args.cycle_steps,
            lbfgs_epochs=args.lbfgs_epochs,
            lbfgs_lr=args.lbfgs_lr,
            lbfgs_max_iter=args.lbfgs_max_iter,
            lbfgs_tol_grad=args.lbfgs_tol_grad,
            lbfgs_tol_change=args.lbfgs_tol_change,
            n_ensemble=args.n_ensemble,
            checkpoint_dir=ckpt_dir,
        )
        model = best_model
        w_rxn_plot = float(w_rxn_best)
        _save(model, all_losses[best_idx], hidden, layers,
              act_scale, dropout, best_idx, model_path, w_rxn=w_rxn_plot)
        print(f'Best model saved -> {model_path}')

    # Plots
    plot_smoothed_vs_original(node_pos, u, eval_grid, u_hat, eps,
                              opath(results_dir, 'smoothed_vs_original.png'))
    plot_loss(all_losses, best_idx,
              opath(results_dir, 'loss_curve.png'))
    plot_loss_breakdown(model, eps_static, DX, F_reaction_y,
                        all_losses, best_idx,
                        opath(results_dir, 'loss_breakdown.png'),
                        w_rxn=w_rxn_plot)
    plot_stress_field(model, eps_static, eval_grid,
                      opath(results_dir, 'stress_fields.png'))
    plot_stress_strain(model, eps_static,
                       opath(results_dir, 'stress_strain.png'))
    plot_pde_residual(model, eps_static, eval_grid,
                      opath(results_dir, 'pde_residual.png'))
    plot_vertical_resultant(model, eps_static, eval_grid, F_reaction_y,
                            opath(results_dir, 'vertical_resultant.png'),
                            row_sum_dx=DX)

    print(f'\nAll done.  Results in {results_dir}')


if __name__ == '__main__':
    main()
