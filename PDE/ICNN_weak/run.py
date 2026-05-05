"""
run.py  (ICNN_weak)
-------------------
Weak-form pipeline: Adam warm-start (500 ep) -> L-BFGS (2000 ep).
Known issues: rxn_factor scale conflict, see README.md.
See ../ICNN/run.py for the nn-EUCLID nodal-force version.
"""

import argparse, os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from smoothing import GaussianSmoother, make_eval_grid
from train import train as train_model
from nn_model import ICNN


def load_data(path):
    d = np.load(path)
    F_reaction_y = float(d['F_reaction_y']) if 'F_reaction_y' in d else None
    return d['node_pos'], d['u'], float(d['DX']), F_reaction_y


def smooth(node_pos, u, DX, margin=3):
    smoother  = GaussianSmoother(node_pos, u, h=2.0*DX)
    eval_grid = make_eval_grid(node_pos, downsample=1, margin=margin)
    u_hat, eps = smoother.compute_vectorised(eval_grid)
    return eval_grid, u_hat, eps


def resolve_data_path(data_arg, condition, grid_size):
    if data_arg: return os.path.abspath(data_arg)
    return os.path.abspath(os.path.join(
        SCRIPT_DIR, '..', '..', 'data', condition, str(grid_size),
        f'lattice_{condition}.npz'))


def infer_tags(data_path):
    parts = os.path.normpath(data_path).split(os.sep)
    if len(parts) >= 3 and parts[-3] == 'static':
        return 'lattice_static', parts[-2]
    return os.path.splitext(os.path.basename(data_path))[0], parts[-2] if len(parts) >= 2 else 'unknown'


def prompt(msg, default):
    v = input(f"{msg} [{default}]: ").strip()
    return v if v else str(default)


def _save_final(model, hidden, layers, act_scale, dropout, losses, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        torch.save({'state_dict': model.state_dict(), 'hidden': hidden,
                    'layers': layers, 'act_scale': act_scale,
                    'dropout': dropout, 'losses': losses}, path)
        print(f'Saved model -> {path}')
    except Exception as e:
        print(f'ERROR: save FAILED: {e}'); raise


def plot_loss(losses, K, out):
    lk = np.asarray(losses) / max(K, 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(lk, lw=1.2)
    tail = max(len(lk)//10, 1)
    fm = lk[-tail:].mean()
    em = lk[-2*tail:-tail].mean() if len(lk) >= 2*tail else lk[0]
    status = 'still dropping' if fm < 0.9*em else 'converged/plateau'
    ax.set_xlabel('Epoch'); ax.set_ylabel('loss/K'); ax.set_title('Weak-form training loss/K')
    ax.annotate(f'final={fm:.2e}\n{status}', xy=(0.98,0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=8, bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
    ax.grid(True, which='both', alpha=0.4); fig.tight_layout(); fig.savefig(out, dpi=150)
    plt.close(fig); print(f'Saved  {out}')


def plot_stress_strain(model, eps_static, out):
    model.eval()
    e_max = float(np.abs(eps_static).max())
    e_range = np.linspace(-1.5*e_max, 1.5*e_max, 400)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    comp_labels = [r'$\varepsilon_{11}$', r'$\varepsilon_{22}$', r'$\varepsilon_{12}$']
    sig_labels  = [r'$\sigma_{11}$', r'$\sigma_{22}$', r'$\sigma_{12}$']
    for comp in range(3):
        ax = axes[comp]; e = np.zeros((len(e_range), 3)); e[:, comp] = e_range
        with torch.no_grad():
            sigma = model(torch.tensor(e, dtype=torch.float32)).numpy()
        for s in range(3):
            ax.plot(e_range, sigma[:,s], label=sig_labels[s], lw=1.8)
        mask = np.abs(e_range) <= 0.1*e_max
        if mask.sum() > 3:
            slope = np.polyfit(e_range[mask], sigma[mask,0], 1)[0]
            ax.plot(e_range, slope*e_range, 'k--', lw=1.2, label=f'tangent≈{slope:.3f}')
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax.set_xlabel(comp_labels[comp]); ax.set_ylabel('stress')
        ax.set_title(f'Sweep {comp_labels[comp]}'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Constitutive law: ICNN_weak  sigma-eps', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'Saved  {out}')


def plot_pde_residual(model, eps_static, eval_grid, out):
    M = eval_grid.shape[0]
    dx = float(eval_grid[0,1,0]-eval_grid[0,0,0]); dy = float(eval_grid[1,0,1]-eval_grid[0,0,1])
    model.eval()
    with torch.no_grad():
        sigma_f = model(torch.tensor(eps_static.reshape(-1,3), dtype=torch.float32)).numpy()
    sigma = sigma_f.reshape(M, M, 3)
    s11, s22, s12 = sigma[:,:,0], sigma[:,:,1], sigma[:,:,2]
    div_x = np.gradient(s11, dx, axis=1) + np.gradient(s12, dy, axis=0)
    div_y = np.gradient(s12, dx, axis=1) + np.gradient(s22, dy, axis=0)
    div_mag = np.sqrt(div_x**2 + div_y**2)
    vmax_mag = np.percentile(div_mag[2:-2,2:-2], 99)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field, title, cmap in zip(axes, [div_x, div_y, div_mag],
        [r'$(\nabla\cdot\sigma)_x$', r'$(\nabla\cdot\sigma)_y$', r'$|\nabla\cdot\sigma|$'],
        ['RdBu_r','RdBu_r','hot_r']):
        if cmap == 'RdBu_r':
            vmax = np.percentile(np.abs(field[2:-2,2:-2]), 99)
            im = ax.imshow(field, origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(field, origin='lower', cmap=cmap, vmin=0, vmax=vmax_mag)
        ax.set_title(title, fontsize=9); ax.set_xlabel('j'); ax.set_ylabel('i')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(r'PDE residual $\nabla\cdot\sigma$ (weak form)', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'Saved  {out}')


def plot_vertical_resultant(model, eps_static, eval_grid, F_reaction_y, out):
    M = eval_grid.shape[0]; dx = float(eval_grid[0,1,0]-eval_grid[0,0,0])
    model.eval()
    with torch.no_grad():
        sigma_f = model(torch.tensor(eps_static.reshape(-1,3), dtype=torch.float32)).numpy()
    sigma = sigma_f.reshape(M, M, 3)
    Ry = sigma[:,:,1].sum(axis=1) * dx; y_row = eval_grid[:,0,1]
    cv = float(np.std(Ry[1:-1]) / max(abs(np.mean(Ry[1:-1])), 1e-12))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(Ry, y_row, 'b.-', lw=1.5, markersize=4, label=r'$\int\sigma_{yy}dx$ (ICNN_weak)')
    if F_reaction_y is not None:
        ax.axvline(F_reaction_y, color='r', ls='--', lw=2, label=f'target={F_reaction_y:.4g}')
    cv_str = f'CV={cv:.3f} ({"good" if cv<0.05 else "not flat"})'
    ax.annotate(cv_str, xy=(0.98,0.05), xycoords='axes fraction', ha='right', va='bottom',
                fontsize=8, bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))
    ax.set_xlabel(r'$\int\sigma_{yy}dx$'); ax.set_ylabel('y')
    ax.set_title('Vertical resultant vs height'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'Saved  {out}  ({cv_str})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=['static','dynamic'], default=None)
    parser.add_argument('--grid-size', type=int, default=None)
    parser.add_argument('--data', default=None)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--force-train', action='store_true')
    args = parser.parse_args()

    if torch.cuda.is_available(): print(f'CUDA: yes | GPU: {torch.cuda.get_device_name(0)}')
    else: print('CUDA: no | CPU')

    condition = args.condition; grid_size = args.grid_size
    if args.data is None:
        if condition is None: condition = prompt('Condition', 'static').lower()
        if grid_size is None: grid_size = int(prompt('Grid size', 100))
    data_path = resolve_data_path(args.data, condition, grid_size)
    cond_tag, grid_tag = infer_tags(data_path)

    results_dir = os.path.join(SCRIPT_DIR, 'results', cond_tag, str(grid_tag))
    os.makedirs(results_dir, exist_ok=True)
    print(f'Results dir: {results_dir}')

    model_path = os.path.abspath(args.model_path) if args.model_path else os.path.join(results_dir, 'icnn_model.pt')

    print(f'Loading {data_path} ...')
    node_pos, u, DX, F_reaction_y = load_data(data_path)
    print(f'  F_reaction_y = {F_reaction_y}')

    print('Smoothing ...')
    eval_grid, u_hat, eps = smooth(node_pos, u, DX, margin=3)
    eps_static = eps[:,:,0,:]
    np.savez(os.path.join(results_dir, 'smoothed_data.npz'), eval_grid=eval_grid, u_hat=u_hat, eps=eps)

    bdry_eps = eps_static[-1,:,:] if F_reaction_y is not None else None
    hidden, layers, act_scale, dropout = 32, 3, 1/12, 0.0
    K = 50

    auto_load = os.path.exists(model_path) and not args.force_train
    if args.load_model or auto_load:
        ckpt = torch.load(model_path, map_location='cpu')
        model = ICNN(hidden=ckpt['hidden'], layers=ckpt['layers'],
                     act_scale=ckpt.get('act_scale',1/12), dropout=ckpt.get('dropout',0.0))
        model.load_state_dict(ckpt['state_dict']); losses = ckpt.get('losses',[])
        print('Loaded model (skipping training)')
    else:
        print('--- Phase 1: Adam warm-start (500 epochs, PDE loss only) ---')
        model, losses_p1 = train_model(
            eps_static, eval_grid, DX=DX, K=K, epochs=500,
            hidden=hidden, layers=layers, act_scale=act_scale, dropout=dropout,
            optimizer='adam', base_lr=1e-3, max_lr=5e-3,
            lr_schedule='cyclic', cycle_steps=100,
            reaction_force=None, bdry_eps=None, checkpoint_path=None, seed=0)

        print('--- Phase 2: L-BFGS refinement (2000 epochs, + reaction loss) ---')
        model, losses_p2 = train_model(
            eps_static, eval_grid, DX=DX, K=K, epochs=2000,
            hidden=hidden, layers=layers, act_scale=act_scale, dropout=dropout,
            optimizer='lbfgs', base_lr=1.0, lr_schedule='constant',
            reaction_force=F_reaction_y, bdry_eps=bdry_eps, rxn_factor=1e2,
            checkpoint_path=model_path, checkpoint_every=200,
            init_state_dict=model.state_dict(), seed=0)
        losses = losses_p1 + losses_p2
        _save_final(model, hidden, layers, act_scale, dropout, losses, model_path)

    plot_loss(losses, K, os.path.join(results_dir, 'loss_curve.png'))
    plot_stress_strain(model, eps_static, os.path.join(results_dir, 'stress_strain.png'))
    plot_pde_residual(model, eps_static, eval_grid, os.path.join(results_dir, 'pde_residual.png'))
    plot_vertical_resultant(model, eps_static, eval_grid, F_reaction_y,
                            os.path.join(results_dir, 'vertical_resultant.png'))
    print(f'\nAll done.  Results in {results_dir}')


if __name__ == '__main__':
    main()
