"""
run.py
------
Full pipeline: load spring-lattice data → smooth → train constitutive NN
→ save diagnostic plots.

Usage (run from PDE/MLP/):
    python simple_data_gen.py            # once, creates ../data/lattice_duffing.npz
    python run.py                        # uses ../data/lattice_duffing.npz by default
    python run.py --data ../data/lattice_linear.npz

Outputs in results/<data-stem>/:
    smoothed_vs_original.png  -- raw vs smoothed displacement field
    loss_curve.png            -- training loss (log scale)
    stress_strain.png         -- NN σ-ε curves (sweep each strain component)
    pde_residual.png          -- pointwise |∇·σ| residual colormap
    smoothed_data.npz         -- smoothed displacement and strain arrays
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smoothing import GaussianSmoother, make_eval_grid
from train import train as train_model



# ------------------------------------------------------------------ helpers --

def load_data(path):
    d = np.load(path)
    F_reaction_y = float(d['F_reaction_y']) if 'F_reaction_y' in d else None
    return d['node_pos'], d['u'], float(d['DX']), F_reaction_y


def smooth(node_pos, u, DX, margin=3):
    smoother  = GaussianSmoother(node_pos, u, h=2.0 * DX)
    eval_grid = make_eval_grid(node_pos, downsample=1, margin=margin)
    u_hat, eps = smoother.compute_vectorised(eval_grid)
    return eval_grid, u_hat, eps


# -------------------------------------------------------------------- plots --

def plot_smoothed_vs_original(node_pos, u_raw, eval_grid, u_hat, eps, out):
    """
    2×2 grid:
      top    — uy on raw lattice  |  uy on smoothed eval grid
      bottom — smoothed ε₁₁       |  smoothed ε₂₂
    """
    uy_raw    = u_raw[:, :, 0, 1]
    uy_smooth = u_hat[:, :, 0, 1]
    eps_11    = eps[:, :, 0, 0]
    eps_22    = eps[:, :, 0, 1]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    def show(ax, data, title, cmap='viridis', sym=False):
        vmax = np.abs(data).max()
        vmin = -vmax if sym else data.min()
        im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('j');  ax.set_ylabel('i')
        fig.colorbar(im, ax=ax, shrink=0.75)

    show(axes[0, 0], uy_raw,    'Original lattice  $u_y$')
    show(axes[0, 1], uy_smooth, 'Smoothed field  $u_y$')
    show(axes[1, 0], eps_11,    r'Smoothed  $\varepsilon_{11}$', sym=True, cmap='RdBu_r')
    show(axes[1, 1], eps_22,    r'Smoothed  $\varepsilon_{22}$', cmap='viridis')

    fig.suptitle('Displacement and strain fields', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out}')


def plot_loss(losses, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PDE loss  (log scale)')
    ax.set_title('Training loss')
    ax.grid(True, which='both', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved  {out}')


def plot_stress_strain(model, eps_static, out):
    """
    Three panels: sweep ε₁₁, ε₂₂, ε₁₂ independently (others held at zero).
    Plots all three stress components vs the swept strain, plus a linear-regime
    tangent fitted to the central 20 % of the sweep range.

    Note: I₂ = ε₁₁² + 2ε₁₂² + ε₂₂² is even in ε₁₂, so the NN cannot
    distinguish +ε₁₂ from −ε₁₂.  The σ₁₂ vs ε₁₂ curve will be symmetric
    rather than antisymmetric — a known limitation of the 2-invariant
    parameterisation.
    """
    model.eval()

    e_max   = float(np.abs(eps_static).max())
    e_range = np.linspace(-1.5 * e_max, 1.5 * e_max, 400)

    comp_labels = [r'$\varepsilon_{11}$',
                   r'$\varepsilon_{22}$',
                   r'$\varepsilon_{12}$']
    sig_labels  = [r'$\sigma_{11}$', r'$\sigma_{22}$', r'$\sigma_{12}$']
    colors      = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for comp in range(3):
        ax = axes[comp]

        e = np.zeros((len(e_range), 3))
        e[:, comp] = e_range

        e11, e22, e12 = e[:, 0], e[:, 1], e[:, 2]
        I1 = e11 + e22
        I2 = e11**2 + 2.0 * e12**2 + e22**2

        inv_t = torch.tensor(
            np.stack([I1, I2], axis=-1), dtype=torch.float32)
        with torch.no_grad():
            sigma = model(inv_t).numpy()   # (400, 3)

        for s in range(3):
            ax.plot(e_range, sigma[:, s], color=colors[s],
                    label=sig_labels[s], lw=1.8)

        # Linear tangent: regression on central 20 % of sweep
        mask = np.abs(e_range) <= 0.1 * e_max
        if mask.sum() > 3:
            slope = np.polyfit(e_range[mask], sigma[mask, 0], 1)[0]
            ax.plot(e_range, slope * e_range, 'k--', lw=1.2,
                    label=fr'linear tangent  $\approx$ {slope:.3f}')

        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', lw=0.5)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        ax.set_xlabel(comp_labels[comp])
        ax.set_ylabel('stress')
        ax.set_title(f'Sweep {comp_labels[comp]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Constitutive law: NN  σ–ε  curves', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out}')


def plot_pde_residual(model, eps_static, eval_grid, out):
    """
    Evaluate ∇·σ_NN at every grid point (static: should be ≈ 0).
    Colourmap clipped at 99th percentile of interior to suppress boundary noise.
    """
    M  = eval_grid.shape[0]
    dx = float(eval_grid[0, 1, 0] - eval_grid[0, 0, 0])
    dy = float(eval_grid[1, 0, 1] - eval_grid[0, 0, 1])

    model.eval()
    eps_f = eps_static.reshape(-1, 3)
    e11, e22, e12 = eps_f[:, 0], eps_f[:, 1], eps_f[:, 2]
    I1 = e11 + e22
    I2 = e11**2 + 2.0 * e12**2 + e22**2

    inv_t = torch.tensor(
        np.stack([I1, I2], axis=-1), dtype=torch.float32)
    with torch.no_grad():
        sigma_f = model(inv_t).numpy()

    sigma = sigma_f.reshape(M, M, 3)
    s11, s22, s12 = sigma[:, :, 0], sigma[:, :, 1], sigma[:, :, 2]

    # ∂/∂x → axis 1 (j),  ∂/∂y → axis 0 (i)
    div_x   = np.gradient(s11, dx, axis=1) + np.gradient(s12, dy, axis=0)
    div_y   = np.gradient(s12, dx, axis=1) + np.gradient(s22, dy, axis=0)
    div_mag = np.sqrt(div_x**2 + div_y**2)

    # Clip colourmap to interior 99th percentile to suppress boundary noise
    interior_mag = div_mag[2:-2, 2:-2]
    vmax_mag = np.percentile(interior_mag, 99)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, field, title, cmap in zip(
        axes,
        [div_x, div_y, div_mag],
        [r'$(\nabla\!\cdot\!\sigma)_x$',
         r'$(\nabla\!\cdot\!\sigma)_y$',
         r'$|\nabla\!\cdot\!\sigma|$  (magnitude, clipped at 99th pct)'],
        ['RdBu_r', 'RdBu_r', 'hot_r'],
    ):
        if cmap == 'RdBu_r':
            vmax = np.percentile(np.abs(field[2:-2, 2:-2]), 99)
            vmin = -vmax
        else:
            vmax = vmax_mag
            vmin = 0
        im = ax.imshow(field, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('j');  ax.set_ylabel('i')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(r'Pointwise PDE residual  $\nabla\!\cdot\!\sigma_\mathrm{NN}$'
                 r'  (should be $\approx 0$)', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out}')


# -------------------------------------------------------------------  main  --

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../data/lattice_duffing.npz',
                        help='Path to lattice .npz file')
    args = parser.parse_args()

    data_stem   = os.path.splitext(os.path.basename(args.data))[0]
    results_dir = os.path.join('results', data_stem)
    os.makedirs(results_dir, exist_ok=True)

    def out(fname):
        return os.path.join(results_dir, fname)

    # 1. Load
    print(f'Loading {args.data} ...')
    node_pos, u, DX, F_reaction_y = load_data(args.data)

    # 2. Smooth
    print('Smoothing displacement field ...')
    eval_grid, u_hat, eps = smooth(node_pos, u, DX, margin=3)
    eps_static = eps[:, :, 0, :]    # (M, M, 3)

    np.savez(out('smoothed_data.npz'), eval_grid=eval_grid, u_hat=u_hat, eps=eps)
    print(f'Saved  {out("smoothed_data.npz")}')

    bdry_eps = eps_static[-1, :, :] if F_reaction_y is not None else None

    # 3. Train
    model, losses = train_model(
        eps_static, eval_grid, DX=DX,
        K=50, epochs=2000, lr=1e-3,
        reaction_force=F_reaction_y, bdry_eps=bdry_eps, rxn_factor=1.0
    )

    # 4. Plots
    plot_smoothed_vs_original(node_pos, u, eval_grid, u_hat, eps,
                              out('smoothed_vs_original.png'))
    plot_loss(losses,         out('loss_curve.png'))
    plot_stress_strain(model, eps_static,
                              out('stress_strain.png'))
    plot_pde_residual(model, eps_static, eval_grid,
                              out('pde_residual.png'))

    print(f'\nAll done.  Results in {results_dir}/')


if __name__ == '__main__':
    main()
