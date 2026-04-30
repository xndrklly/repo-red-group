"""
Spring-grid K regression report for the physical forced case using a single
SINDy Adam dense model with explicit L1 regularization.

Setup:
  * 10 x 10 grid (100 DOFs).
  * Top row clamped (pinned). Bottom row carries the applied force.
  * Mass M = I, damping C = alpha M + beta K (Rayleigh, alpha=0.05, beta=0.02).
  * Force is a sinusoidal point load with a Gaussian envelope localized at
    the centre of the bottom row.
  * Initial conditions are identically zero (no initial displacement / velocity).
  * Damping and forcing are treated as KNOWN: the regression target is
        target = a + C v - f
    so that target = -K u and only K is estimated.

Method (single model):
  * SINDy Adam dense (L1=1e-3)

The report keeps the same diagnostics as the physical forced two-model report:
loss curves, K-error curve, recovered K matrix, eigenvalue trajectories,
rollout snapshots/animations, node traces, parameter history, and modal
comparisons.

CPU only.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import generate_spring_grid_physical_forced_report as forced
import spring_grid_K_regression as base


MODEL_NAME = "SINDy Adam dense (L1=1e-3)"
DEFAULT_OUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "figures"
    / "spring_grid_10x10_physical_forced_adam_dense_l1_1e_3_report.html"
)


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>{report_title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1180px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
h3 {{ margin-top: 1.6em; }}
h4 {{ margin-bottom: 0.4em; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.4em 0.8em; text-align: right; }}
th {{ background: #f3f3f3; }}
img {{ max-width: 100%; height: auto; border: 1px solid #e5e7eb; }}
video {{ width: 100%; height: auto; border: 1px solid #e5e7eb; background: #000; }}
h5 {{ margin: 0 0 0.4em 0; font-size: 0.95rem; }}
.method-block {{ margin-bottom: 2.5em; }}
code {{ background: #f5f5f5; padding: 0.1em 0.3em; border-radius: 3px; }}
.media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
               gap: 1rem; align-items: start; }}
.video-card {{ display: block; }}
.note-box {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.9em 1em; border-radius: 6px; }}
</style>
</head>
<body>
<h1>{report_title}</h1>

<h2>Setup</h2>
<ul>
<li>Grid: <code>{n}x{n}</code> ({n_dofs} free DOFs after pinning the top row)</li>
<li>Springs: {spring_description}</li>
<li>Mass M = I (unit point masses).</li>
<li>Damping C = <code>alpha * M + beta * K</code> with alpha = {alpha:.3f}, beta = {beta:.3f}
    (Rayleigh damping, treated as KNOWN in the regression).</li>
<li>Forcing: {force_description} (treated as KNOWN in the regression).</li>
<li>Initial conditions: identically zero (no initial displacement, no initial
    velocity). The response is purely driven by the localized force.</li>
<li>Trajectories: {n_traj}, dt = {dt}, t_end = {t_end}, total samples = {n_total}.</li>
<li>Train/test split: {n_train} / {n_test} samples (80/20 over pooled samples).</li>
<li>Regression target: <code>target = a + C v - f</code>, so that <code>target = -K u</code>.
    Only K is estimated; C and f are constants of the regression problem.</li>
<li>Recovered K = <code>-Xi^T</code>, symmetrised for physical diagnostics.</li>
<li>Dense optimization: no locality mask. All {n_active} / {n_full} ({pct_active:.1f}%)
    Xi entries are trainable.</li>
<li>Optimizer: Adam derivative matching with explicit L1 penalty
    <code>lambda = {adam_l1_lambda:.0e}</code> and no proximal step.</li>
</ul>

<h2>Summary table</h2>
{summary_table}

<h2>Train / test loss curves</h2>
<img src="data:image/png;base64,{loss_curve_b64}" />

<h2>Relative K error vs epoch</h2>
<img src="data:image/png;base64,{k_err_b64}" />

<h2>Recovered K matrices (final)</h2>
<img src="data:image/png;base64,{k_matrix_b64}" />

<h2>Eigenvalue trajectories</h2>
{eigenvalue_blocks}

<h2>Per-method physical diagnostics</h2>
<div class="note-box">
<p>The predicted K is rolled out under the same known M / C and the forcing
applied to the first trajectory. The "True / data" panels are the forward
simulation that produced the regression dataset; the "Regressed / model"
panels are the forward simulation of the regressed K under the identical
inputs, so any visible discrepancy is attributable to errors in K only.</p>
</div>

{presentation_blocks}

<h2>Output locations</h2>
{summary_notes}

<h2>Notes</h2>
<p>The regression treats damping and forcing as KNOWN inputs. With M = I and
the relation <code>u_tt + C u_t = -K u + f</code>, moving the known terms to
the LHS yields <code>(u_tt + C u_t - f) = -K u</code>. We pass
<code>target = a + C v - f</code> as the regression target so that the
recovered Xi satisfies <code>K = -Xi^T</code> directly. This report uses a
single dense SINDy Adam fit: no entries are masked out, and the explicit
<code>L1</code> penalty acts on the full Xi matrix.</p>
</body>
</html>
"""


def main(
    *,
    n: int = 10,
    n_traj: int = 4,
    t_end: float = 20.0,
    dt: float = 0.01,
    residual_epochs: int = 1500,
    alpha: float = 0.05,
    beta: float = 0.02,
    adam_l1_lambda: float = 1e-3,
    force_amplitude: float = 1.0,
    force_omega: float = 2.0 * np.pi,
    force_sigma_cols: float = 1.5,
    out_path: Path = DEFAULT_OUT_PATH,
) -> Path:
    dtype = torch.float64
    device = torch.device("cpu")

    K_phys = forced.build_physical_K_clamped_top(n)
    C_phys = forced.build_rayleigh_damping(K_phys, alpha=alpha, beta=beta)
    force_fns = forced.build_per_trajectory_force_fns(
        n,
        n_traj=n_traj,
        amplitude=force_amplitude,
        base_omega=force_omega,
        sigma_cols=force_sigma_cols,
    )

    print("\n=== Spring-Grid Physical Forced Dense Adam Report (CPU) ===")
    print(f"  n = {n}, n_traj = {n_traj}, t_end = {t_end}, dt = {dt}")
    print(f"  Rayleigh damping: alpha = {alpha}, beta = {beta}")
    print(f"  Adam dense L1 lambda = {adam_l1_lambda:.3e}")
    print(
        f"  Forcing: amplitude = {force_amplitude}, omega = {force_omega:.4f}, "
        f"sigma_cols = {force_sigma_cols}, applied at row 0 (bottom)"
    )

    data = base.generate_dataset(
        n=n,
        n_traj=n_traj,
        t_end=t_end,
        dt=dt,
        seed=0,
        device=device,
        simulation_backend="scipy",
        dtype=dtype,
        stiffness_matrix=K_phys,
        damping_matrix=C_phys,
        force_fn=force_fns,
        pinned_rows=(n - 1,),
        initial_condition_mode="zero",
    )
    print(f"  free DOFs = {data['free'].size}, total samples = {data['U'].shape[0]}")
    print(
        f"  max |u| = {np.abs(data['U']).max():.3e}, "
        f"max |a| = {np.abs(data['A']).max():.3e}, "
        f"max |a + C v - f| = {np.abs(data['A_target']).max():.3e}"
    )

    grad_problem = base.build_regression_problem(data, device=device, dtype=dtype)
    selected_nodes = base.select_representative_nodes(
        data["n"], data["free"], data["trajectories"][0]["U_full"]
    )
    incident_specs = base.build_incident_specs(selected_nodes, data["n"])

    dense_result = base.train_gradient_method(
        grad_problem["theta_train"],
        grad_problem["target_train"],
        grad_problem["theta_test"],
        grad_problem["target_test"],
        grad_problem["K_true"],
        n_features=grad_problem["n_features"],
        n_states=grad_problem["n_states"],
        n_epochs=residual_epochs,
        lr=5e-2,
        l1_lambda=adam_l1_lambda,
        proximal=False,
        snapshot_every=max(1, residual_epochs // 60),
        n_eig=5,
        seed=0,
        xi_mask=None,
        history_specs=incident_specs,
        history_every=1,
        free=data["free"],
        n_total_dofs=data["N"],
    )
    grad_results = OrderedDict([(MODEL_NAME, dense_result)])
    stls_results: OrderedDict[str, dict] = OrderedDict()
    print(
        f"  {MODEL_NAME}: train={dense_result['train_losses'][-1]:.3e} "
        f"test={dense_result['test_losses'][-1]:.3e} "
        f"K_err={dense_result['snap_K_err'][-1]:.3e}"
    )

    output_dirs = base.build_output_dirs(out_path)
    loss_curve_b64 = base.plot_loss_curves(
        grad_results,
        output_stem=output_dirs["figure_dir"] / "current_loss_curves",
    )
    k_err_b64 = base.plot_K_error(
        grad_results,
        output_stem=output_dirs["figure_dir"] / "current_relative_k_error",
    )
    k_matrix_b64 = base.plot_K_matrices(
        grad_problem["K_true"],
        OrderedDict([(MODEL_NAME, base.xi_to_K(dense_result["xi"]))]),
        output_stem=output_dirs["figure_dir"] / "current_recovered_k_matrices",
    )

    eig_blocks = []
    for name, res in grad_results.items():
        b64 = base.plot_eigenvalues(
            name,
            res,
            output_stem=output_dirs["figure_dir"] / f"current_eigenvalues_{base.slugify(name)}",
        )
        eig_blocks.append(
            f'<div class="method-block"><h3>{name}</h3>'
            f'<img src="data:image/png;base64,{b64}" /></div>'
        )

    summary_table = forced.build_summary_table(stls_results, grad_results)
    presentation_blocks, summary_notes = forced.build_diagnostic_blocks(
        data=data,
        stls_results=stls_results,
        grad_results=grad_results,
        out_path=out_path,
        rollout_force_fn=force_fns[0],
    )

    n_full = grad_problem["n_features"] * grad_problem["n_states"]
    n_active = n_full
    spring_description = (
        "deterministic support-weighted lattice with the clamped support row "
        "at <code>i = n-1</code> (top): vertical springs are slightly stiffer "
        "than horizontal ones, diagonals are softer braces, all springs stiffen "
        "near the clamped top row, and the centre of the panel has mild "
        "reinforcement"
    )
    force_description = (
        f"sinusoidal Gaussian point load applied along row 0 (bottom of plate). "
        f"Each of the {n_traj} trajectories uses the same amplitude "
        f"({force_amplitude}) and Gaussian width (sigma = {force_sigma_cols} columns) "
        f"but a different angular frequency (multiples of base omega = {force_omega:.4f}) "
        f"and a slightly shifted column centre, so the data spans enough of the state "
        f"space for K to be identifiable from zero initial conditions"
    )

    html = HTML_TEMPLATE.format(
        report_title=(
            "Spring-Grid K Regression Report "
            "(10x10 Physical, Damping + Bottom Forcing, SINDy Adam Dense L1=1e-3)"
        ),
        n=data["n"],
        n_dofs=data["free"].size,
        n_traj=data["n_traj"],
        dt=data["dt"],
        t_end=data["t_end"],
        spring_description=spring_description,
        alpha=alpha,
        beta=beta,
        adam_l1_lambda=adam_l1_lambda,
        force_description=force_description,
        n_total=grad_problem["n_total"],
        n_train=grad_problem["n_train"],
        n_test=grad_problem["n_total"] - grad_problem["n_train"],
        n_active=n_active,
        n_full=n_full,
        pct_active=100.0,
        summary_table=summary_table,
        loss_curve_b64=loss_curve_b64,
        k_err_b64=k_err_b64,
        k_matrix_b64=k_matrix_b64,
        eigenvalue_blocks="\n".join(eig_blocks),
        presentation_blocks=presentation_blocks,
        summary_notes=summary_notes,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  Wrote report to: {out_path.resolve()}")
    print(f"  Assets under: {output_dirs['figure_dir']}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--n-traj", type=int, default=4)
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--residual-epochs", type=int, default=1500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--adam-l1-lambda", type=float, default=1e-3)
    parser.add_argument("--force-amplitude", type=float, default=1.0)
    parser.add_argument("--force-omega", type=float, default=2.0 * np.pi)
    parser.add_argument("--force-sigma-cols", type=float, default=1.5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    main(
        n=args.n,
        n_traj=args.n_traj,
        t_end=args.t_end,
        dt=args.dt,
        residual_epochs=args.residual_epochs,
        alpha=args.alpha,
        beta=args.beta,
        adam_l1_lambda=args.adam_l1_lambda,
        force_amplitude=args.force_amplitude,
        force_omega=args.force_omega,
        force_sigma_cols=args.force_sigma_cols,
        out_path=args.out,
    )
