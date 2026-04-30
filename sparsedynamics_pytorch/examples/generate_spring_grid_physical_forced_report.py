"""
Spring-grid K regression report with physically motivated K, known Rayleigh
damping, and a known localized forcing on the bottom of the plate.

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

Methods (only two, by request):
  * STLS dense (lam=0.05)
  * SINDy Adam local (no L1)

The report skips trajectory-based regression methods and trajectory rollout
diagnostics (figures 2A/2B/2C/2D and time traces); it keeps the lattice
schematic, training loss / K-error curves, recovered K matrices, eigenvalue
trajectories, per-node parameter history, and modal comparison figures.

CPU only.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import spring_grid_K_regression as base
from sindy_torch.systems.spring_grid import node_index


STLS_NAME = "STLS dense (lam=0.05)"
ADAM_RESIDUAL_NAME = "SINDy Adam local (no L1)"


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
<li>Locality: 8-connected neighbours (H + V + both diagonals) + self;
    {n_active} / {n_full} ({pct_active:.1f}%) of Xi entries are free
    parameters under the local mask.</li>
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
<p>Each method's predicted K is rolled out under the same known M / C and the
forcing applied to the first trajectory. The "True / data" panels are the
forward simulation that produced the regression dataset; the
"Regressed / model" panels are the forward simulation of the regressed K
under the identical inputs, so any visible discrepancy is attributable to
errors in K only.</p>
</div>

{presentation_blocks}

<h2>Output locations</h2>
{summary_notes}

<h2>Notes</h2>
<p>The regression treats damping and forcing as KNOWN inputs. With M = I and
the relation <code>u_tt + C u_t = -K u + f</code>, moving the known terms to
the LHS yields <code>(u_tt + C u_t - f) = -K u</code>. We pass
<code>target = a + C v - f</code> as the regression target so that the
recovered Xi satisfies <code>K = -Xi^T</code> directly. STLS dense uses
sequential thresholded least squares with no locality mask. SINDy Adam local
zeros gradients on entries outside the 8-connected neighbour stencil.</p>
</body>
</html>
"""


def build_physical_K_clamped_top(n: int) -> sp.csr_matrix:
    """Physical K with stiffening near the clamped TOP row (i = n-1).

    Mirror of ``base.build_physically_motivated_stiffness_matrix`` but with the
    support row at i = n-1 instead of i = 0. Preserves the rest of the
    physically motivated pattern (vertical > horizontal stiffness, softer
    diagonal braces, mild centre reinforcement).
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    center = (n - 1) / 2.0
    sigma = max(1.0, 0.22 * n)

    def add_pair(a: int, b: int, k: float) -> None:
        rows.extend([a, b, a, b])
        cols.extend([a, b, b, a])
        vals.extend([k, k, -k, -k])

    def support_factor(row_mid: float) -> float:
        # Distance from clamped top row n-1.
        d = (n - 1) - row_mid
        return 1.0 + 0.65 * np.exp(-d / max(1.0, 0.28 * n))

    def center_factor(col_mid: float) -> float:
        return 1.0 + 0.18 * np.exp(-((col_mid - center) ** 2) / (2.0 * sigma * sigma))

    for i in range(n):
        for j in range(n - 1):
            a = node_index(i, j, n)
            b = node_index(i, j + 1, n)
            k = 0.95 * support_factor(float(i)) * center_factor(j + 0.5)
            add_pair(a, b, float(k))

    for i in range(n - 1):
        for j in range(n):
            a = node_index(i, j, n)
            b = node_index(i + 1, j, n)
            k = 1.20 * support_factor(i + 0.5) * center_factor(float(j))
            add_pair(a, b, float(k))

    for i in range(n - 1):
        for j in range(n - 1):
            a = node_index(i, j, n)
            b = node_index(i + 1, j + 1, n)
            k = 0.42 * support_factor(i + 0.5) * (0.9 + 0.1 * center_factor(j + 0.5))
            add_pair(a, b, float(k))

    N = n * n
    K = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    K.sum_duplicates()
    return K


def build_bottom_localized_force_fn(
    n: int,
    *,
    amplitude: float = 1.0,
    omega: float = 2.0 * np.pi,
    sigma_cols: float = 1.5,
    center_col: float | None = None,
    phase: float = 0.0,
):
    """Sinusoidal force, Gaussian-windowed across the bottom row (i = 0)."""
    N = n * n
    cols = np.arange(n, dtype=float)
    if center_col is None:
        center_col = (n - 1) / 2.0
    spatial = np.exp(-((cols - center_col) ** 2) / (2.0 * sigma_cols ** 2))
    spatial /= spatial.max()
    bottom_nodes = np.array([node_index(0, j, n) for j in range(n)], dtype=int)
    spatial_full = np.zeros(N, dtype=float)
    spatial_full[bottom_nodes] = amplitude * spatial

    def f(t: float) -> np.ndarray:
        return spatial_full * np.sin(omega * float(t) + phase)

    return f


def build_per_trajectory_force_fns(
    n: int,
    *,
    n_traj: int,
    amplitude: float,
    base_omega: float,
    sigma_cols: float,
) -> list:
    """Per-trajectory bottom-row Gaussian forces with varied frequencies and centres.

    Each trajectory is still a localized sinusoidal point load on the bottom of
    the plate, but the centre column and angular frequency are jittered so the
    response spans more of the state space (necessary for K identifiability
    with zero initial conditions and a single forcing pattern would otherwise
    produce a low-rank dataset).
    """
    omega_factors = [0.7, 1.0, 1.3, 1.7, 2.1, 2.7, 3.3, 4.1]
    center_col_default = (n - 1) / 2.0
    # Symmetric column offsets that stay inside the bottom row.
    column_offsets = [0.0, -1.5, 1.5, -2.5, 2.5, -1.0, 1.0, 0.5]
    fns = []
    for traj in range(n_traj):
        omega_t = base_omega * omega_factors[traj % len(omega_factors)]
        center_t = center_col_default + column_offsets[traj % len(column_offsets)]
        center_t = float(np.clip(center_t, 0.0, n - 1))
        phase_t = (traj * np.pi) / max(1, n_traj)
        fns.append(
            build_bottom_localized_force_fn(
                n,
                amplitude=amplitude,
                omega=omega_t,
                sigma_cols=sigma_cols,
                center_col=center_t,
                phase=phase_t,
            )
        )
    return fns


def build_rayleigh_damping(
    K: sp.csr_matrix,
    *,
    alpha: float = 0.05,
    beta: float = 0.02,
) -> sp.csr_matrix:
    """C = alpha * M + beta * K with M = I."""
    N = K.shape[0]
    return (alpha * sp.eye(N, format="csr") + beta * K).tocsr()


def build_summary_table(stls_results: dict, grad_results: dict) -> str:
    rows = [
        "<tr><th>Method</th><th>Train MSE</th><th>Test MSE</th>"
        "<th>Rel. K error</th><th>Final sparsity (%)</th></tr>"
    ]
    for name, res in stls_results.items():
        sparsity = (res["xi"].abs() < 1e-10).float().mean().item() * 100
        rows.append(
            f"<tr><td>{name}</td><td>{res['train_mse']:.3e}</td>"
            f"<td>{res['test_mse']:.3e}</td>"
            f"<td>{res['K_err']:.3e}</td>"
            f"<td>{sparsity:.1f}</td></tr>"
        )
    for name, res in grad_results.items():
        sparsity = (res["xi"].abs() < 1e-10).float().mean().item() * 100
        rows.append(
            f"<tr><td>{name}</td><td>{res['train_losses'][-1]:.3e}</td>"
            f"<td>{res['test_losses'][-1]:.3e}</td>"
            f"<td>{res['snap_K_err'][-1]:.3e}</td>"
            f"<td>{sparsity:.1f}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def build_diagnostic_blocks(
    *,
    data: dict,
    stls_results: dict,
    grad_results: dict,
    out_path: Path,
    rollout_force_fn,
) -> tuple[str, str]:
    """Lattice, rollout (quiver / heatmap / traces), parameter history, modal.

    Each method's predicted K is rolled out under the same known M / C /
    force_fn used by the data so the comparison is apples-to-apples.
    """
    output_dirs = base.build_output_dirs(out_path)
    figure_dir = output_dirs["figure_dir"]
    animation_dir = output_dirs["animation_dir"]
    n = data["n"]
    free = data["free"]
    n_total = data["N"]
    K_true_full = data["K_full"]
    K_true_free = data["K_free"]
    C_full = data["C_full"]
    base_traj = data["trajectories"][0]

    selected_nodes = base.select_representative_nodes(n, free, base_traj["U_full"])
    incident_specs = base.build_incident_specs(selected_nodes, n)

    figure1_assets = base.plot_lattice_stiffness(
        K_true_full,
        n,
        figure_dir / "figure_1_true_stiffness_lattice",
    )
    blocks = [
        '<div class="method-block">',
        "<h3>Figure 1: lattice schematic</h3>",
        base.render_image_tag(out_path.parent, figure1_assets["png"], "True stiffness lattice"),
        "</div>",
    ]

    method_names = [STLS_NAME, ADAM_RESIDUAL_NAME]
    for model_name in method_names:
        if model_name in stls_results:
            K_pred_free = stls_results[model_name]["K_pred"].detach().cpu().numpy()
            K_pred_full = base.build_full_K_from_free(K_pred_free, free, n_total)
            pseudo_epochs = np.array([0.0, 1.0])
            pseudo_history = {}
            for spec in incident_specs:
                val = float(-K_pred_full[spec["node"], spec["neighbor"]])
                pseudo_history[spec["series_key"]] = np.array([val, val])
            history_epochs = pseudo_epochs
            history_values = pseudo_history
            param_note = (
                "STLS is a direct sparse solve, so this panel shows the final "
                "estimate as a constant pseudo-history."
            )
        else:
            res = grad_results[model_name]
            K_pred_free = base.xi_to_K(res["xi"]).detach().cpu().numpy()
            history_epochs = res["parameter_history_epochs"]
            history_values = res["parameter_history"]
            param_note = None

        slug = base.slugify(model_name)

        pred_rollout = base.rollout_linear_model(
            K_pred_free,
            free,
            n,
            dt=data["dt"],
            t_end=data["t_end"],
            u0_full=base_traj["u0_full"],
            v0_full=base_traj["v0_full"],
            device=torch.device("cpu"),
            simulation_backend="scipy",
            dtype=torch.float64,
            damping_matrix=C_full,
            force_fn=rollout_force_fn,
        )

        quiver_assets = base.make_quiver_snapshot_figure(
            base_traj,
            pred_rollout,
            n,
            model_name,
            figure_dir / f"figure_2A_quiver_snapshots_true_vs_regressed_{slug}",
        )
        quiver_vmax = base.compute_visual_max(base_traj["U_full"], pred_rollout["U_full"])
        quiver_vis_scale = 0.75 / quiver_vmax
        anim_true_quiver = base.make_quiver_animation(
            base_traj,
            n,
            f"True scalar-deflection quiver, {model_name}",
            animation_dir / f"animation_2B_true_quiver_{slug}",
            vis_scale=quiver_vis_scale,
        )
        anim_pred_quiver = base.make_quiver_animation(
            pred_rollout,
            n,
            f"Regressed scalar-deflection quiver, {model_name}",
            animation_dir / f"animation_2B_regressed_quiver_{slug}",
            vis_scale=quiver_vis_scale,
        )

        heatmap_assets = base.make_heatmap_snapshot_figure(
            base_traj,
            pred_rollout,
            n,
            model_name,
            figure_dir / f"figure_2C_heatmap_snapshots_true_vs_regressed_{slug}",
        )
        heatmap_vmax = base.compute_visual_max(
            np.abs(base_traj["U_full"]), np.abs(pred_rollout["U_full"])
        )
        anim_true_heat = base.make_heatmap_animation(
            base_traj,
            n,
            f"True displacement heatmap, {model_name}",
            animation_dir / f"animation_2D_true_heatmap_{slug}",
            vmax=heatmap_vmax,
        )
        anim_pred_heat = base.make_heatmap_animation(
            pred_rollout,
            n,
            f"Regressed displacement heatmap, {model_name}",
            animation_dir / f"animation_2D_regressed_heatmap_{slug}",
            vmax=heatmap_vmax,
        )

        traces_assets = base.make_node_trace_plots(
            base_traj,
            pred_rollout,
            selected_nodes,
            n,
            model_name,
            figure_dir / f"figure_3_node_traces_{slug}",
        )

        param_assets = base.make_parameter_history_plots(
            model_name,
            selected_nodes,
            incident_specs,
            K_true_full,
            history_values,
            history_epochs,
            n,
            figure_dir / f"figure_5_parameter_history_{slug}",
            note=param_note,
        )
        modal_assets = base.make_modal_comparison(
            model_name,
            K_true_free,
            K_pred_free,
            figure_dir / f"figure_9_modes_true_vs_regressed_{slug}",
        )
        mode_shape_assets = base.make_mode_shape_examples(
            model_name,
            K_true_free,
            K_pred_free,
            n,
            free,
            figure_dir / f"figure_9_mode_shapes_example_{slug}",
        )
        blocks.extend([
            '<div class="method-block">',
            f"<h3>{model_name}</h3>",
            "<h4>Figure 2A: static quiver snapshots</h4>",
            base.render_image_tag(out_path.parent, quiver_assets["png"], f"Quiver snapshots for {model_name}"),
            "<h4>Figure 2B: quiver animations</h4>",
            '<div class="media-grid">',
            base.render_video_tag(out_path.parent, anim_true_quiver, "True / data quiver"),
            base.render_video_tag(out_path.parent, anim_pred_quiver, "Regressed / model quiver"),
            "</div>",
            "<h4>Figure 2C: static heatmap snapshots</h4>",
            base.render_image_tag(out_path.parent, heatmap_assets["png"], f"Heatmap snapshots for {model_name}"),
            "<h4>Figure 2D: heatmap animations</h4>",
            '<div class="media-grid">',
            base.render_video_tag(out_path.parent, anim_true_heat, "True / data heatmap"),
            base.render_video_tag(out_path.parent, anim_pred_heat, "Regressed / model heatmap"),
            "</div>",
            "<h4>Figure 3: selected node traces</h4>",
            base.render_image_tag(out_path.parent, traces_assets["png"], f"Node traces for {model_name}"),
            "<h4>Figure 5: local stiffness evolution</h4>",
            base.render_image_tag(out_path.parent, param_assets["png"], f"Parameter history for {model_name}"),
            "<h4>Figure 9: modal comparison</h4>",
            base.render_image_tag(out_path.parent, modal_assets["png"], f"Modal comparison for {model_name}"),
            "<h5>Representative mode shapes</h5>",
            base.render_image_tag(out_path.parent, mode_shape_assets["png"], f"Mode shapes for {model_name}"),
            "</div>",
        ])

    notes = [
        "<ul>",
        f"<li>Static figure assets saved under <code>{figure_dir}</code>.</li>",
        f"<li>Animation assets saved under <code>{animation_dir}</code>.</li>",
        "<li>Node selection used for Figure 3 / Figure 5: "
        + base.build_node_summary_list(selected_nodes, n)
        + "</li>",
        "<li>Each predicted rollout uses the dataset's known damping C and the "
        "first trajectory's known force_fn so the comparison isolates errors "
        "in K alone.</li>",
        "</ul>",
    ]
    return "\n".join(blocks), "\n".join(notes)


def main(
    *,
    n: int = 10,
    n_traj: int = 4,
    t_end: float = 20.0,
    dt: float = 0.01,
    residual_epochs: int = 1500,
    alpha: float = 0.05,
    beta: float = 0.02,
    force_amplitude: float = 1.0,
    force_omega: float = 2.0 * np.pi,
    force_sigma_cols: float = 1.5,
    out_path: Path = Path("figures/spring_grid_10x10_physical_forced_two_model_report.html"),
) -> Path:
    dtype = torch.float64
    device = torch.device("cpu")

    K_phys = build_physical_K_clamped_top(n)
    C_phys = build_rayleigh_damping(K_phys, alpha=alpha, beta=beta)
    force_fns = build_per_trajectory_force_fns(
        n,
        n_traj=n_traj,
        amplitude=force_amplitude,
        base_omega=force_omega,
        sigma_cols=force_sigma_cols,
    )

    print("\n=== Spring-Grid Physical Forced Report (CPU) ===")
    print(f"  n = {n}, n_traj = {n_traj}, t_end = {t_end}, dt = {dt}")
    print(f"  Rayleigh damping: alpha = {alpha}, beta = {beta}")
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
        pinned_rows=(n - 1,),  # clamp top row
        initial_condition_mode="zero",
    )
    print(f"  free DOFs = {data['free'].size}, total samples = {data['U'].shape[0]}")
    print(
        f"  max |u| = {np.abs(data['U']).max():.3e}, "
        f"max |a| = {np.abs(data['A']).max():.3e}, "
        f"max |a + C v - f| = {np.abs(data['A_target']).max():.3e}"
    )

    stls_problem = base.build_regression_problem(data, device=device, dtype=dtype)
    grad_problem = base.build_regression_problem(data, device=device, dtype=dtype)

    selected_nodes = base.select_representative_nodes(
        data["n"], data["free"], data["trajectories"][0]["U_full"]
    )
    incident_specs = base.build_incident_specs(selected_nodes, data["n"])

    stls_results = OrderedDict()
    stls_results[STLS_NAME] = base.run_stls(
        stls_problem["theta_train"],
        stls_problem["target_train"],
        stls_problem["theta_test"],
        stls_problem["target_test"],
        stls_problem["K_true"],
        lam=0.05,
        mask=None,
    )
    print(
        f"  {STLS_NAME}: train={stls_results[STLS_NAME]['train_mse']:.3e} "
        f"test={stls_results[STLS_NAME]['test_mse']:.3e} "
        f"K_err={stls_results[STLS_NAME]['K_err']:.3e}"
    )

    residual_result = base.train_gradient_method(
        grad_problem["theta_train"],
        grad_problem["target_train"],
        grad_problem["theta_test"],
        grad_problem["target_test"],
        grad_problem["K_true"],
        n_features=grad_problem["n_features"],
        n_states=grad_problem["n_states"],
        n_epochs=residual_epochs,
        lr=5e-2,
        l1_lambda=0.0,
        proximal=False,
        snapshot_every=max(1, residual_epochs // 60),
        n_eig=5,
        seed=0,
        xi_mask=grad_problem["xi_mask"],
        history_specs=incident_specs,
        history_every=1,
        free=data["free"],
        n_total_dofs=data["N"],
    )
    grad_results = OrderedDict()
    grad_results[ADAM_RESIDUAL_NAME] = residual_result
    print(
        f"  {ADAM_RESIDUAL_NAME}: train={residual_result['train_losses'][-1]:.3e} "
        f"test={residual_result['test_losses'][-1]:.3e} "
        f"K_err={residual_result['snap_K_err'][-1]:.3e}"
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

    K_preds = OrderedDict()
    K_preds[STLS_NAME] = base.xi_to_K(stls_results[STLS_NAME]["xi"])
    for name, res in grad_results.items():
        K_preds[name] = base.xi_to_K(res["xi"])
    k_matrix_b64 = base.plot_K_matrices(
        stls_problem["K_true"],
        K_preds,
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

    summary_table = build_summary_table(stls_results, grad_results)
    presentation_blocks, summary_notes = build_diagnostic_blocks(
        data=data,
        stls_results=stls_results,
        grad_results=grad_results,
        out_path=out_path,
        rollout_force_fn=force_fns[0],
    )

    n_active = int(data["locality_free"].sum())
    n_full = grad_problem["n_features"] * grad_problem["n_states"]

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
        report_title="Spring-Grid K Regression Report (10x10 Physical, Damping + Bottom Forcing)",
        n=data["n"],
        n_dofs=data["free"].size,
        n_traj=data["n_traj"],
        dt=data["dt"],
        t_end=data["t_end"],
        spring_description=spring_description,
        alpha=alpha,
        beta=beta,
        force_description=force_description,
        n_total=grad_problem["n_total"],
        n_train=grad_problem["n_train"],
        n_test=grad_problem["n_total"] - grad_problem["n_train"],
        n_active=n_active,
        n_full=n_full,
        pct_active=100 * n_active / n_full,
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
    parser.add_argument("--force-amplitude", type=float, default=1.0)
    parser.add_argument("--force-omega", type=float, default=2.0 * np.pi)
    parser.add_argument("--force-sigma-cols", type=float, default=1.5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/spring_grid_10x10_physical_forced_two_model_report.html"),
    )
    args = parser.parse_args()

    main(
        n=args.n,
        n_traj=args.n_traj,
        t_end=args.t_end,
        dt=args.dt,
        residual_epochs=args.residual_epochs,
        alpha=args.alpha,
        beta=args.beta,
        force_amplitude=args.force_amplitude,
        force_omega=args.force_omega,
        force_sigma_cols=args.force_sigma_cols,
        out_path=args.out,
    )
