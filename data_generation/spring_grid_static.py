import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def prompt_with_default(prompt, default):
    raw = input(f'{prompt} [{default}]: ').strip()
    return raw if raw else str(default)


def idx(i, j, grid_n):
    """Flat node index for lattice node (i, j)."""
    return i * grid_n + j


def build_springs(n, a, rng):
    """
    Build full 8-neighbor spring list (each spring assembled once).

    Returns DOF index arrays plus spring unit directions and linear stiffness.
    """
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]

    na_l, nb_l = [], []
    lx_l, ly_l = [], []
    k1_l = []

    for i in range(n):
        for j in range(n):
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if not (0 <= ni < n and 0 <= nj < n):
                    continue
                if not (ni > i or (ni == i and nj > j)):
                    continue

                sdx = (nj - j) * a
                sdy = (ni - i) * a
                L = np.sqrt(sdx**2 + sdy**2)
                lx, ly = sdx / L, sdy / L

                k1 = rng.uniform(0.8, 1.2) / L

                na_l.append(i * n + j)
                nb_l.append(ni * n + nj)
                lx_l.append(lx)
                ly_l.append(ly)
                k1_l.append(k1)

    na = np.array(na_l, dtype=int)
    nb = np.array(nb_l, dtype=int)
    lx = np.array(lx_l, dtype=float)
    ly = np.array(ly_l, dtype=float)
    k1 = np.array(k1_l, dtype=float)

    ax = 2 * na
    ay = ax + 1
    bx = 2 * nb
    by = bx + 1
    return ax, ay, bx, by, lx, ly, k1


def assemble_stiffness(ax, ay, bx, by, lx, ly, k1, n_dof):
    """Assemble sparse global stiffness matrix for linear springs."""
    kll = k1 * lx**2
    klm = k1 * lx * ly
    kmm = k1 * ly**2

    rows = np.r_[ax, ax, ay, ay,  bx, bx, by, by,
                 ax, ax, ay, ay,  bx, bx, by, by]
    cols = np.r_[ax, ay, ax, ay,  bx, by, bx, by,
                 bx, by, bx, by,  ax, ay, ax, ay]
    vals = np.r_[kll, klm, klm, kmm,   kll, klm, klm, kmm,
                -kll, -klm, -klm, -kmm, -kll, -klm, -klm, -kmm]

    K = sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    K.sum_duplicates()
    return K


def get_free_dofs(n, pinned_rows=(0,)):
    pinned = []
    for i in pinned_rows:
        for j in range(n):
            node = idx(i, j, n)
            pinned.extend([2 * node, 2 * node + 1])  # pin both ux, uy
    pinned = np.array(sorted(pinned), dtype=int)
    free = np.setdiff1d(np.arange(2 * n * n), pinned)
    return np.array(pinned, dtype=int), free


def solve_static(K, free, f):
    Kf = K[free, :][:, free].tocsc()
    u_free = spsolve(Kf, f[free])
    u = np.zeros(K.shape[0], dtype=float)
    u[free] = u_free
    return u


def compute_reaction_force_y(u_full, K, n):
    """
    Total vertical support reaction on the bottom boundary.

    For constrained DOFs, reaction = -f_int, with f_int = K @ u.
    """
    f_int = K.dot(u_full)
    bottom_y_dofs = np.array([2 * idx(0, j, n) + 1 for j in range(n)], dtype=int)
    return float((-f_int[bottom_y_dofs]).sum())


def save_displacement_plot(ux, uy, force_mag, fname):
    """Save PDE-style displacement diagnostics: |u|, ux, uy."""
    mag = np.sqrt(ux**2 + uy**2)
    vmax_ux = max(np.abs(ux).max(), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    im0 = axes[0].imshow(mag, origin='lower', cmap='viridis')
    axes[0].set_title('Displacement magnitude |u|')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(ux, origin='lower', cmap='RdBu_r',
                         vmin=-vmax_ux, vmax=vmax_ux)
    axes[1].set_title('ux  (horizontal)')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(uy, origin='lower', cmap='viridis')
    axes[2].set_title('uy  (vertical)')
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.set_xlabel('j  (x)')
        ax.set_ylabel('i  (y)')

    fig.suptitle(f'Static spring lattice (point force = {force_mag:.3g})', fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Static spring-grid equilibrium solver')
    parser.add_argument('--grid-size', type=int, default=None,
                        help='Grid size n for an n x n lattice (prompted if omitted)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for spring stiffnesses')
    parser.add_argument('--force', type=float, default=1.0,
                        help='Point load magnitude at top-center node')
    parser.add_argument('--dx', type=float, default=1.0, help='Grid spacing')
    args = parser.parse_args()

    grid_size = args.grid_size
    if grid_size is None:
        raw = prompt_with_default('Grid size', 100)
        while True:
            try:
                grid_size = int(raw)
                if grid_size < 3:
                    raise ValueError
                break
            except ValueError:
                raw = prompt_with_default('Please enter an integer >= 3 for grid size', 100)

    n = int(grid_size)
    if n < 3:
        raise ValueError('grid-size must be >= 3')

    N_nodes = n * n
    n_dof = 2 * N_nodes
    print(f'Grid: {n} x {n}  ->  {N_nodes} nodes, {n_dof} DOFs')

    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    ax, ay, bx, by, lx, ly, k1 = build_springs(n, args.dx, rng)
    K = assemble_stiffness(ax, ay, bx, by, lx, ly, k1, n_dof)
    pinned, free = get_free_dofs(n, pinned_rows=(0,))

    # Point force at top-center in vertical direction only.
    f = np.zeros(n_dof, dtype=float)
    load_node = idx(n - 1, (n - 1) // 2, n)
    f[2 * load_node + 1] = float(args.force)

    u_full = solve_static(K, free, f)
    elapsed = time.perf_counter() - t0
    F_reaction_y = compute_reaction_force_y(u_full, K, n)

    ux = u_full[0::2].reshape(n, n)
    uy = u_full[1::2].reshape(n, n)
    u = np.stack([ux, uy], axis=-1)[:, :, np.newaxis, :]  # (n, n, 1, 2)

    jj, ii = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    node_pos = np.stack([jj * args.dx, ii * args.dx], axis=-1)  # (n, n, 2)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, 'data', 'static', str(n))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'lattice_static.npz')
    out_plot = os.path.join(out_dir, 'displacement_static.png')

    np.savez(
        out_path,
        node_pos=node_pos,
        u=u,
        DX=float(args.dx),
        F_reaction_y=F_reaction_y,
        meta_grid_size=n,
        meta_force=float(args.force),
        meta_seed=int(args.seed),
    )

    print(f'Solved linear static equilibrium in {elapsed:.2f}s')
    print(f'Springs assembled: {len(k1)}')
    print(f'max |u_y| = {np.max(np.abs(uy)):.6e}')
    print(f'Saved {out_path}')
    print(f'  node_pos: {node_pos.shape}')
    print(f'  u       : {u.shape}')
    print(f'  F_reaction_y (bottom total): {F_reaction_y:.6e}')

    save_displacement_plot(ux, uy, float(args.force), out_plot)
    print(f'Saved {out_plot}')


if __name__ == '__main__':
    main()
