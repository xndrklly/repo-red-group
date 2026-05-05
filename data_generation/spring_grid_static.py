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


def default_output_subdir(n, force, spring_model):
    """
    Folder under data/static/, e.g. 100_1N, 100_10N, 100_1N_Duffing.
    Force is formatted without a dot in the name (1.5 -> 1p5).
    """
    f = float(force)
    if abs(f - round(f)) < 1e-9:
        fstr = str(int(round(f)))
    else:
        fstr = str(f).replace('.', 'p')
    name = f'{n}_{fstr}N'
    if spring_model == 'duffing':
        name += '_Duffing'
    return name


def build_springs(n, a, rng, spring_model='linear', duffing_gamma=15.0):
    """
    Build full 8-neighbor spring list (each spring assembled once).

    Returns DOF index arrays, rest length L0, linear stiffness k1, and
    cubic coefficient k3 (zero for linear; for duffing,
    k3 = duffing_gamma * k1 / L0**2 so cubic term competes when |dl| ~ L0/gamma).
    """
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]

    na_l, nb_l = [], []
    lx_l, ly_l = [], []
    k1_l, L0_l, k3_l = [], [], []

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
                L0 = float(np.sqrt(sdx**2 + sdy**2))
                lx, ly = sdx / L0, sdy / L0

                k1 = rng.uniform(0.8, 1.2) / L0

                na_l.append(i * n + j)
                nb_l.append(ni * n + nj)
                lx_l.append(lx)
                ly_l.append(ly)
                k1_l.append(k1)
                L0_l.append(L0)
                if spring_model == 'duffing':
                    k3_l.append(float(duffing_gamma * k1 / (L0 * L0)))
                else:
                    k3_l.append(0.0)

    na = np.array(na_l, dtype=int)
    nb = np.array(nb_l, dtype=int)
    lx = np.array(lx_l, dtype=float)
    ly = np.array(ly_l, dtype=float)
    k1 = np.array(k1_l, dtype=float)
    L0 = np.array(L0_l, dtype=float)
    k3 = np.array(k3_l, dtype=float)

    ax = 2 * na
    ay = ax + 1
    bx = 2 * nb
    by = bx + 1
    return ax, ay, bx, by, lx, ly, k1, L0, k3


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
            pinned.extend([2 * node, 2 * node + 1])
    pinned = np.array(sorted(pinned), dtype=int)
    free = np.setdiff1d(np.arange(2 * n * n), pinned)
    return np.array(pinned, dtype=int), free


def solve_static(K, free, f):
    Kf = K[free, :][:, free].tocsc()
    u_free = spsolve(Kf, f[free])
    u = np.zeros(K.shape[0], dtype=float)
    u[free] = u_free
    return u


def assemble_force_tangent(u, ax, ay, bx, by, lx, ly, k1, k3, n_dof):
    """
    Small-strain / fixed-direction springs with nonlinear axial law
        T = k1 * dl + k3 * dl**3,
    dl = (u_b - u_a) · n0  with n0 = (lx, ly) from the reference lattice.

    Matches the linear ``assemble_stiffness`` model when k3 = 0.
    Returns nodal force F_int with same convention as ``K @ u`` (so
    equilibrium is ``F_int = f`` and Newton uses ``R = F_int - f``).
    """
    F_int = np.zeros(n_dof, dtype=float)
    rows, cols, vals = [], [], []

    for s in range(len(ax)):
        iax, iay, ibx, iby = ax[s], ay[s], bx[s], by[s]
        dux = u[ibx] - u[iax]
        duy = u[iby] - u[iay]
        dl = dux * lx[s] + duy * ly[s]
        T = k1[s] * dl + k3[s] * (dl ** 3)
        Ea = k1[s] + 3.0 * k3[s] * (dl ** 2)

        F_int[iax] -= T * lx[s]
        F_int[iay] -= T * ly[s]
        F_int[ibx] += T * lx[s]
        F_int[iby] += T * ly[s]

        kll = Ea * lx[s] ** 2
        klm = Ea * lx[s] * ly[s]
        kmm = Ea * ly[s] ** 2

        for r, c, v in zip(
            [iax, iax, iay, iay, ibx, ibx, iby, iby,
             iax, iax, iay, iay, ibx, ibx, iby, iby],
            [iax, iay, iax, iay, ibx, iby, ibx, iby,
             ibx, iby, ibx, iby, iax, iay, iax, iay],
            [kll, klm, klm, kmm, kll, klm, klm, kmm,
             -kll, -klm, -klm, -kmm, -kll, -klm, -klm, -kmm],
        ):
            rows.append(r)
            cols.append(c)
            vals.append(v)

    K_t = sp.coo_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                        shape=(n_dof, n_dof)).tocsr()
    K_t.sum_duplicates()
    return F_int, K_t


def solve_static_nonlinear(
    ax, ay, bx, by, lx, ly, k1, k3, f_ext, pinned, free, n_dof,
    max_iter=80, tol=1e-8, verbose=False,
):
    """Newton–Raphson: F_int(u) - f_ext = 0 on free DOFs (same as K u = f)."""
    u = np.zeros(n_dof, dtype=float)
    free = np.asarray(free, dtype=int)

    for it in range(max_iter):
        F_int, K_t = assemble_force_tangent(u, ax, ay, bx, by, lx, ly, k1, k3, n_dof)
        R = F_int - f_ext
        Rf = R[free]
        nrm = float(np.linalg.norm(Rf))
        if verbose and (it % 5 == 0 or it == max_iter - 1):
            print(f'    Newton {it:3d}  |R|_free = {nrm:.4e}')
        if nrm < tol:
            break

        Kff = K_t[free, :][:, free].tocsc()
        try:
            du_f = spsolve(Kff, -Rf)  # K_t du = -R = f - F_int
        except Exception as e:
            if verbose:
                print(f'    Newton: linear solve failed ({e})')
            break
        if not np.all(np.isfinite(du_f)):
            if verbose:
                print('    Newton: non-finite du; stopping')
            break

        # Damped line search (full step often OK; shrink if residual grows)
        alpha = 1.0
        u_trial = u.copy()
        for _ in range(12):
            u_trial[free] = u[free] + alpha * du_f
            F2, _ = assemble_force_tangent(
                u_trial, ax, ay, bx, by, lx, ly, k1, k3, n_dof,
            )
            R2 = F2 - f_ext
            if float(np.linalg.norm(R2[free])) <= nrm * 1.001 or alpha < 1e-4:
                u = u_trial
                break
            alpha *= 0.5
        else:
            u[free] = u[free] + alpha * du_f

    return u


def compute_reaction_force_y_linear(u_full, K, n):
    f_int = K.dot(u_full)
    bottom_y_dofs = np.array([2 * idx(0, j, n) + 1 for j in range(n)], dtype=int)
    return float((-f_int[bottom_y_dofs]).sum())


def compute_reaction_force_y_nonlinear(u_full, ax, ay, bx, by, lx, ly, k1, k3, n):
    F_int, _ = assemble_force_tangent(
        u_full, ax, ay, bx, by, lx, ly, k1, k3, 2 * n * n,
    )
    bottom_y_dofs = np.array([2 * idx(0, j, n) + 1 for j in range(n)], dtype=int)
    return float((-F_int[bottom_y_dofs]).sum())


def save_displacement_plot(ux, uy, force_mag, fname, spring_model):
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

    fig.suptitle(
        f'Static spring lattice ({spring_model}), point force = {force_mag:.3g}',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Static spring-grid equilibrium (linear or Duffing cubic springs)',
        epilog=(
            'Default output folder: data/static/{n}_{force}N/ or ..._Duffing/ . '
            'Duffing demo: --spring-model duffing --grid-size 28 --force 4 '
            '--duffing-gamma 25  (use --tag to override folder name).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--grid-size', type=int, default=None,
                        help='Grid size n for an n x n lattice (prompted if omitted)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for spring stiffnesses')
    parser.add_argument('--force', type=float, default=1.0,
                        help='Point load magnitude at top-center (+uy)')
    parser.add_argument('--dx', type=float, default=1.0, help='Grid spacing')
    parser.add_argument('--spring-model', choices=('linear', 'duffing'), default='linear',
                        help='linear: one sparse solve; duffing: T=k1*dl+k3*dl^3, Newton–Raphson')
    parser.add_argument('--duffing-gamma', type=float, default=15.0,
                        help='k3 = gamma*k1/L0^2 for each spring (larger -> stronger cubic)')
    parser.add_argument('--newton-max-iter', type=int, default=80)
    parser.add_argument('--newton-tol', type=float, default=1e-8)
    parser.add_argument('--tag', type=str, default=None,
                        help='Subfolder under data/static/ (default: {n}_{force}N or ..._Duffing)')
    parser.add_argument('--newton-verbose', action='store_true',
                        help='Print Newton residual norms')
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
    print(f'Spring model: {args.spring_model}')

    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    ax, ay, bx, by, lx, ly, k1, L0, k3 = build_springs(
        n, args.dx, rng,
        spring_model=args.spring_model,
        duffing_gamma=float(args.duffing_gamma),
    )
    pinned, free = get_free_dofs(n, pinned_rows=(0,))

    f = np.zeros(n_dof, dtype=float)
    load_node = idx(n - 1, (n - 1) // 2, n)
    f[2 * load_node + 1] = float(args.force)

    if args.spring_model == 'linear':
        K = assemble_stiffness(ax, ay, bx, by, lx, ly, k1, n_dof)
        u_full = solve_static(K, free, f)
        F_reaction_y = compute_reaction_force_y_linear(u_full, K, n)
        label = 'linear static equilibrium'
    else:
        u_full = solve_static_nonlinear(
            ax, ay, bx, by, lx, ly, k1, k3, f, pinned, free, n_dof,
            max_iter=args.newton_max_iter,
            tol=args.newton_tol,
            verbose=args.newton_verbose,
        )
        F_reaction_y = compute_reaction_force_y_nonlinear(
            u_full, ax, ay, bx, by, lx, ly, k1, k3, n,
        )
        label = 'nonlinear (Duffing) static equilibrium'

    elapsed = time.perf_counter() - t0

    ux = u_full[0::2].reshape(n, n)
    uy = u_full[1::2].reshape(n, n)
    u = np.stack([ux, uy], axis=-1)[:, :, np.newaxis, :]

    jj, ii = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    node_pos = np.stack([jj * args.dx, ii * args.dx], axis=-1)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub = args.tag if args.tag else default_output_subdir(
        n, float(args.force), args.spring_model,
    )
    out_dir = os.path.join(project_root, 'data', 'static', sub)
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
        meta_spring_model=args.spring_model,
        meta_duffing_gamma=float(args.duffing_gamma),
    )

    print(f'Solved {label} in {elapsed:.2f}s')
    print(f'Springs assembled: {len(k1)}')
    print(f'max |u_y| = {np.max(np.abs(uy)):.6e}')
    print(f'Saved {out_path}')
    print(f'  node_pos: {node_pos.shape}')
    print(f'  u       : {u.shape}')
    print(f'  F_reaction_y (bottom total): {F_reaction_y:.6e}')

    save_displacement_plot(ux, uy, float(args.force), out_plot, args.spring_model)
    print(f'Saved {out_plot}')


if __name__ == '__main__':
    main()
