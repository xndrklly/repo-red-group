"""
simple_data_gen.py
------------------
Solve a 100×100 spring lattice at static equilibrium.

Modes
-----
  linear  : linear Hookean springs  (direct sparse solve)
  duffing : nonlinear  F = k₁δ + k₃δ³  (Newton-Raphson)

Usage:
    python simple_data_gen.py                # duffing (default)
    python simple_data_gen.py --mode linear
    python simple_data_gen.py --mode duffing

Output saved to data/lattice_{mode}.npz:
    node_pos : (N, N, 2)     node (x, y) positions
    u        : (N, N, 1, 2)  displacements [ux, uy]
    DX       : float          lattice spacing
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# --------------------------------------------------------------- spring list --

def build_springs(N, a, rng, mode):
    """
    Return vectorised spring arrays: na, nb, lx, ly, k1, k3.
    Each entry is a 1-D numpy array of length n_springs.

    k3 = 0 for linear mode; k3 ≈ 40/L³ (±20 %) for duffing mode,
    chosen so the cubic correction is ~10 % at the max spring extension.
    """
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]

    na_l, nb_l = [], []
    lx_l, ly_l = [], []
    k1_l, k3_l = [], []

    for i in range(N):
        for j in range(N):
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if not (0 <= ni < N and 0 <= nj < N):
                    continue
                if not (ni > i or (ni == i and nj > j)):
                    continue   # assemble each spring once only

                sdx = (nj - j) * a
                sdy = (ni - i) * a
                L   = np.sqrt(sdx**2 + sdy**2)
                lx, ly = sdx / L, sdy / L

                k1 = rng.uniform(0.8, 1.2) / L
                k3 = rng.uniform(0.8, 1.2) * 40.0 / L**3 if mode == 'duffing' else 0.0

                na_l.append(i * N + j)
                nb_l.append(ni * N + nj)
                lx_l.append(lx);  ly_l.append(ly)
                k1_l.append(k1);  k3_l.append(k3)

    na = np.array(na_l, dtype=int);  nb = np.array(nb_l, dtype=int)
    lx = np.array(lx_l);             ly = np.array(ly_l)
    k1 = np.array(k1_l);             k3 = np.array(k3_l)

    # DOF index arrays (reused by both solvers)
    ax = 2 * na;  ay = ax + 1
    bx = 2 * nb;  by = bx + 1

    return ax, ay, bx, by, lx, ly, k1, k3


def _assemble_K(ax, ay, bx, by, lx, ly, kt, n_dof):
    """Assemble sparse stiffness from vectorised spring arrays."""
    kll = kt * lx**2;  klm = kt * lx * ly;  kmm = kt * ly**2

    rows = np.r_[ax, ax, ay, ay,  bx, bx, by, by,
                 ax, ax, ay, ay,  bx, bx, by, by]
    cols = np.r_[ax, ay, ax, ay,  bx, by, bx, by,
                 bx, by, bx, by,  ax, ay, ax, ay]
    vals = np.r_[kll, klm, klm, kmm,   kll, klm, klm, kmm,
                -kll,-klm,-klm,-kmm,  -kll,-klm,-klm,-kmm]

    return coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()


# ------------------------------------------------------------------- solvers --

def solve_linear(ax, ay, bx, by, lx, ly, k1, n_dof, prescribed):
    K = _assemble_K(ax, ay, bx, by, lx, ly, k1, n_dof)

    c_dofs = np.array(sorted(prescribed), dtype=int)
    f_dofs = np.setdiff1d(np.arange(n_dof), c_dofs)

    u_bar = np.zeros(n_dof)
    for d, v in prescribed.items():
        u_bar[d] = v

    K_ff  = K[f_dofs, :][:, f_dofs]
    f_rhs = -K[f_dofs, :][:, c_dofs].dot(u_bar[c_dofs])

    print('Solving linear system ...')
    u = u_bar.copy()
    u[f_dofs] = spsolve(K_ff, f_rhs)
    return u


def solve_nonlinear(ax, ay, bx, by, lx, ly, k1, k3, n_dof, prescribed,
                    n_load_steps=4, max_iter=25, tol=1e-10):
    """Newton-Raphson with load stepping for Duffing springs."""
    c_dofs = np.array(sorted(prescribed), dtype=int)
    f_dofs = np.setdiff1d(np.arange(n_dof), c_dofs)

    u_bar = np.zeros(n_dof)
    for d, v in prescribed.items():
        u_bar[d] = v

    u = np.zeros(n_dof)

    for step in range(1, n_load_steps + 1):
        frac = step / n_load_steps
        u[c_dofs] = u_bar[c_dofs] * frac
        print(f'  Load step {step}/{n_load_steps} ...', end='', flush=True)

        for it in range(max_iter):
            # Spring extensions
            d_ext = lx * (u[bx] - u[ax]) + ly * (u[by] - u[ay])

            # Internal forces (vectorised scatter-add)
            F = k1 * d_ext + k3 * d_ext**3
            f_int = np.zeros(n_dof)
            np.add.at(f_int, ax, -F * lx);  np.add.at(f_int, ay, -F * ly)
            np.add.at(f_int, bx,  F * lx);  np.add.at(f_int, by,  F * ly)

            R   = -f_int[f_dofs]
            res = np.linalg.norm(R)

            if res < tol:
                print(f'  converged ({it + 1} iter, |R| = {res:.2e})')
                break

            # Tangent stiffness and Newton step
            kt  = k1 + 3.0 * k3 * d_ext**2
            K_T = _assemble_K(ax, ay, bx, by, lx, ly, kt, n_dof)
            u[f_dofs] += spsolve(K_T[f_dofs, :][:, f_dofs], R)
        else:
            print(f'  WARNING: did not converge in {max_iter} iterations')

    return u


def compute_reaction_force_y(u, ax, ay, bx, by, lx, ly, k1, k3, bottom_y_dofs):
    """Compute total vertical support reaction on bottom boundary."""
    d_ext = lx * (u[bx] - u[ax]) + ly * (u[by] - u[ay])
    F = k1 * d_ext + k3 * d_ext**3

    f_int = np.zeros_like(u)
    np.add.at(f_int, ax, -F * lx);  np.add.at(f_int, ay, -F * ly)
    np.add.at(f_int, bx,  F * lx);  np.add.at(f_int, by,  F * ly)

    return float((-f_int[bottom_y_dofs]).sum())


# ----------------------------------------------------------------- main plot --

def save_displacement_plot(ux, uy, delta, fname):
    mag  = np.sqrt(ux**2 + uy**2)
    vmax_ux = max(np.abs(ux).max(), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    im0 = axes[0].imshow(mag, origin='lower', cmap='viridis')
    axes[0].set_title('Displacement magnitude |u|')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(ux, origin='lower', cmap='RdBu_r',
                         vmin=-vmax_ux, vmax=vmax_ux)
    axes[1].set_title('ux  (horizontal)')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(uy, origin='lower', cmap='viridis',
                         vmin=0, vmax=uy.max())
    axes[2].set_title('uy  (vertical)')
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.set_xlabel('j  (x)');  ax.set_ylabel('i  (y)')

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'Saved  {fname}')


# ---------------------------------------------------------------------- main --

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['linear', 'duffing'],
                        default='duffing',
                        help='Spring type: linear or duffing (default: duffing)')
    args = parser.parse_args()
    mode = args.mode

    N   = 100
    a   = 1.0
    # Duffing needs larger displacement so the cubic term is visible
    delta = 0.05 if mode == 'linear' else 5.0
    rng = np.random.default_rng(42)

    # Node positions: node (i,j) → x = j*a, y = i*a
    jj, ii   = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float))
    node_pos = np.stack([jj * a, ii * a], axis=-1)   # (N, N, 2)

    n_dof = 2 * N * N

    # Boundary conditions: fix bottom row, pull top row up by delta
    prescribed = {}
    bottom_y_dofs = []
    for j in range(N):
        nb_idx = 0 * N + j
        prescribed[2 * nb_idx] = 0.0
        prescribed[2 * nb_idx + 1] = 0.0
        bottom_y_dofs.append(2 * nb_idx + 1)

        nt_idx = (N - 1) * N + j
        prescribed[2 * nt_idx] = 0.0
        prescribed[2 * nt_idx + 1] = delta

    print(f'Building {mode} springs ...')
    ax, ay, bx, by, lx, ly, k1, k3 = build_springs(N, a, rng, mode)
    print(f'  {len(k1)} springs assembled.')

    if mode == 'linear':
        u_full = solve_linear(ax, ay, bx, by, lx, ly, k1, n_dof, prescribed)
    else:
        u_full = solve_nonlinear(ax, ay, bx, by, lx, ly, k1, k3, n_dof, prescribed)

    ux = u_full[0::2].reshape(N, N)
    uy = u_full[1::2].reshape(N, N)
    u  = np.stack([ux, uy], axis=-1)[:, :, np.newaxis, :]   # (N, N, 1, 2)
    F_reaction_y = compute_reaction_force_y(
        u_full, ax, ay, bx, by, lx, ly, k1, k3, np.array(bottom_y_dofs, dtype=int)
    )

    os.makedirs('../data', exist_ok=True)
    out_npz  = f'../data/lattice_{mode}.npz'
    out_plot = f'../data/displacement_{mode}.png'

    np.savez(out_npz, node_pos=node_pos, u=u, DX=a, F_reaction_y=F_reaction_y)
    print(f'Saved  {out_npz}')
    print(f'  node_pos : {node_pos.shape}')
    print(f'  u        : {u.shape}   max|u| = {np.abs(u).max():.4f}')
    print(f'  F_reaction_y (bottom total) : {F_reaction_y:.6e}')

    save_displacement_plot(ux, uy, delta, out_plot)


if __name__ == '__main__':
    main()
