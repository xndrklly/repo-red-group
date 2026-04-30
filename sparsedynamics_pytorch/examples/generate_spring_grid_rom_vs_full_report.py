"""
Compare a full spring-grid regression model against a block-averaged ROM.

The report mirrors the physical forced two-model report structure. Both the
fine-grid model and the ROM use the same estimator family, selectable as:

  * ``adam``: locality-masked Adam on the linear regression problem.
  * ``stls``: sequential thresholded least squares on the same regression
    problem, with diagnostics recorded at every STLS iteration, including the
    initial least-squares solve at iteration 0.

The report keeps the established rollout / trace / modal diagnostics per
model, and adds cross-scale figures that compare both models on the same
coarse observables.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import spring_grid_K_regression as base
import generate_spring_grid_physical_forced_report as forced
from sindy_torch.systems import BlockAverageReduction, apply_block_average_reduction, build_block_average_reduction


FULL_MODEL_NAME_TEMPLATES = {
    "adam": "Full SINDy Adam local (fine grid)",
    "stls": "Full STLS local (fine grid)",
}
ROM_MODEL_NAME_TEMPLATES = {
    "adam": "ROM SINDy Adam local (block size = {block_size})",
    "stls": "ROM STLS local (block size = {block_size})",
}

DEFAULT_OUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "figures"
    / "spring_grid_10x10_physical_forced_rom_vs_full_report.html"
)


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>{report_title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1220px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
h3 {{ margin-top: 1.6em; }}
h4 {{ margin-bottom: 0.4em; }}
h5 {{ margin: 0 0 0.4em 0; font-size: 0.95rem; }}
table {{ border-collapse: collapse; margin: 1em 0; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.45em 0.65em; text-align: right; }}
th {{ background: #f3f3f3; }}
td.label, th.label {{ text-align: left; }}
img {{ max-width: 100%; height: auto; border: 1px solid #e5e7eb; }}
video {{ width: 100%; height: auto; border: 1px solid #e5e7eb; background: #000; }}
code {{ background: #f5f5f5; padding: 0.1em 0.3em; border-radius: 3px; }}
.media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
               gap: 1rem; align-items: start; }}
.video-card {{ display: block; }}
.note-box {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.9em 1em; border-radius: 6px; }}
.method-block {{ margin-bottom: 2.5em; }}
</style>
</head>
<body>
<h1>{report_title}</h1>

<h2>Setup</h2>
<ul>
<li>Physical dataset: <code>{n}x{n}</code> fine grid with <code>{n_free}</code> free DOFs, clamped top row, known Rayleigh damping, known bottom-row localized forcing, and zero initial conditions.</li>
<li>ROM construction: square block averaging with <code>block_size = {block_size}</code>, giving a <code>{n_rom}x{n_rom}</code> coarse grid with <code>{n_rom_dofs}</code> states.</li>
<li>General constraint enforced by the code: the fine grid side length must be an integer multiple of the block size.</li>
<li>Regression target on the fine grid: <code>a + C v - f = -K u</code>.</li>
<li>Regression target on the ROM: block-average the same fine-grid target and state snapshots, then fit the coarse operator in the same linear form.</li>
<li>Both learned models use the same estimator family: {solver_description}.</li>
<li>Cross-model fairness: the direct apples-to-apples comparison is on coarse observables, where the full model rollout is projected into the ROM space and compared against the ROM rollout.</li>
</ul>

<h2>Model summary</h2>
{summary_table}

<h2>Native training losses</h2>
<div class="note-box">
<p>The native train/test MSE values below live in different state spaces: the fine model is trained on the full free-DOF state, while the ROM is trained on the block-averaged state. They are useful as optimization diagnostics, but the cross-scale comparison figures below are the fairest way to compare the two models directly.</p>
{native_diagnostic_note}
</div>
<img src="data:image/png;base64,{loss_curve_b64}" />

<h2>Relative K Error vs {iteration_axis_heading}</h2>
<div class="note-box">
<p>The fine-model curve uses the true fine-grid <code>K_free</code>. The ROM curve uses the symmetrized projected coarse operator derived from the same physical fine model.</p>
</div>
<img src="data:image/png;base64,{k_err_b64}" />

<h2>Operator comparison</h2>
<img src="data:image/png;base64,{operator_matrix_b64}" />

<h2>Eigenvalue trajectories</h2>
{eigenvalue_blocks}

<h2>Cross-scale comparison</h2>
<div class="note-box">
<p>The figures in this section put both models in the ROM observation space. The "full model" curve is the fine-grid rollout projected through the same block-average operator used to define the ROM data. The "ROM" curve is the native ROM rollout. This isolates reduction error from pure visualization differences.</p>
</div>
<h3>Coarse observable traces</h3>
<img src="data:image/png;base64,{coarse_trace_b64}" />
<h3>Coarse RMS error maps</h3>
<img src="data:image/png;base64,{coarse_error_b64}" />
<h3>Fine-grid snapshots with lifted ROM</h3>
<img src="data:image/png;base64,{lifted_snapshot_b64}" />

<h2>Full-model diagnostics</h2>
{full_model_block}

<h2>ROM diagnostics</h2>
{rom_model_block}

<h2>Output locations and notes</h2>
{summary_notes}
</body>
</html>
"""


def make_reduced_force_fn(
    fine_force_fn,
    reduction: BlockAverageReduction,
) -> callable:
    """Reduce a fine-grid full force vector into the ROM space."""
    fine_free = reduction.free_dofs

    def f(t: float) -> np.ndarray:
        fine_full = np.asarray(fine_force_fn(float(t)), dtype=float).reshape(
            reduction.fine_grid_size ** 2
        )
        fine_free_values = fine_full[fine_free]
        return reduction.restriction @ fine_free_values

    return f


def select_diagnostic_nodes_flexible(
    n: int,
    free: np.ndarray,
    trajectory_full: np.ndarray,
) -> OrderedDict[str, int]:
    """Use the standard node picker when possible, with a small-grid fallback."""
    try:
        return base.select_representative_nodes(n, free, trajectory_full)
    except RuntimeError:
        amplitude = np.std(trajectory_full, axis=0)
        ranked = sorted((int(idx) for idx in free), key=lambda idx: amplitude[idx], reverse=True)
        labels = ["probe1", "probe2", "probe3", "probe4"]
        return OrderedDict(
            (labels[i], ranked[i]) for i in range(min(len(labels), len(ranked)))
        )

def fit_regression_model(
    *,
    solver: str,
    problem: dict,
    selected_nodes: OrderedDict[str, int],
    n: int,
    free: np.ndarray,
    n_total_dofs: int,
    residual_epochs: int,
    l1_lambda: float = 0.0,
    n_eig: int = 5,
) -> tuple[dict, list[dict]]:
    """Fit either Adam or STLS and return a report-ready result bundle."""
    incident_specs = base.build_incident_specs(selected_nodes, n)
    if solver == "adam":
        result = base.train_gradient_method(
            problem["theta_train"],
            problem["target_train"],
            problem["theta_test"],
            problem["target_test"],
            problem["K_true"],
            n_features=problem["n_features"],
            n_states=problem["n_states"],
            n_epochs=residual_epochs,
            lr=5e-2,
            l1_lambda=l1_lambda,
            proximal=False,
            snapshot_every=max(1, residual_epochs // 60),
            n_eig=n_eig,
            seed=0,
            xi_mask=problem["xi_mask"],
            history_specs=incident_specs,
            history_every=1,
            free=free,
            n_total_dofs=n_total_dofs,
        )
        result["solver_kind"] = "adam"
        return result, incident_specs

    if solver == "stls":
        result = base.run_stls(
            problem["theta_train"],
            problem["target_train"],
            problem["theta_test"],
            problem["target_test"],
            problem["K_true"],
            lam=0.05,
            mask=problem["xi_mask"],
            n_iter=10,
            history_specs=incident_specs,
            free=free,
            n_total_dofs=n_total_dofs,
            n_eig=n_eig,
            return_history=True,
        )
        result["solver_kind"] = "stls"
        return result, incident_specs

    raise ValueError(f"Unknown solver: {solver!r}")


def build_rom_diagnostic_view(rom_data: dict) -> dict:
    """Create a dataset-like view so existing plotting helpers can be reused."""
    n_rom = int(rom_data["n_rom"])
    N_rom = int(rom_data["N_rom"])
    free_rom = np.arange(N_rom, dtype=int)
    trajectories = []
    for traj in rom_data["trajectories"]:
        trajectories.append({
            "times": traj["times"].copy(),
            "U_full": traj["U_rom"].copy(),
            "V_full": traj["V_rom"].copy(),
            "A_full": traj["A_rom"].copy(),
            "u0_full": traj["u0_rom"].copy(),
            "v0_full": traj["v0_rom"].copy(),
            "traj_index": traj["traj_index"],
        })
    return {
        "n": n_rom,
        "N": N_rom,
        "free": free_rom,
        "U": rom_data["U_rom"],
        "V": rom_data["V_rom"],
        "A": rom_data["A_rom"],
        "F": rom_data["F_rom"],
        "A_target": rom_data["A_target_rom"],
        "K_full": rom_data["K_rom"],
        "K_free": rom_data["K_rom"],
        "C_full": rom_data["C_rom"],
        "C_free": rom_data["C_rom"],
        "locality_free": rom_data["locality_rom"],
        "n_traj": rom_data["n_traj"],
        "dt": rom_data["dt"],
        "t_end": rom_data["t_end"],
        "trajectories": trajectories,
    }


def project_fine_rollout_to_rom(
    fine_rollout: dict,
    reduction: BlockAverageReduction,
    fine_free: np.ndarray,
    n_rom: int,
) -> dict:
    """Project a fine-grid rollout into the ROM observation space."""
    U = apply_block_average_reduction(fine_rollout["U_full"][:, fine_free], reduction)
    V = apply_block_average_reduction(fine_rollout["V_full"][:, fine_free], reduction)
    A = apply_block_average_reduction(fine_rollout["A_full"][:, fine_free], reduction)
    return {
        "times": fine_rollout["times"].copy(),
        "U_full": U,
        "V_full": V,
        "A_full": A,
        "u0_full": U[0].copy(),
        "v0_full": V[0].copy(),
        "n": int(n_rom),
    }


def lift_rom_rollout_to_fine(
    rom_rollout: dict,
    prolongation: np.ndarray,
    fine_free: np.ndarray,
    fine_n_total: int,
) -> dict:
    """Lift a ROM rollout back to a block-constant fine-grid field."""
    U_free = rom_rollout["U_full"] @ prolongation.T
    V_free = rom_rollout["V_full"] @ prolongation.T
    A_free = rom_rollout["A_full"] @ prolongation.T
    return {
        "times": rom_rollout["times"].copy(),
        "U_full": base.full_series_from_free(U_free, fine_free, fine_n_total),
        "V_full": base.full_series_from_free(V_free, fine_free, fine_n_total),
        "A_full": base.full_series_from_free(A_free, fine_free, fine_n_total),
    }


def plot_operator_matrix_overview(
    *,
    K_full_true: np.ndarray,
    K_full_pred: np.ndarray,
    K_rom_true: np.ndarray,
    K_rom_pred: np.ndarray,
    output_stem: Path | None = None,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    full_vmax = float(np.max(np.abs(K_full_true)))
    rom_vmax = float(np.max(np.abs(K_rom_true)))
    entries = [
        (axes[0, 0], K_full_true, full_vmax, "True fine-grid K"),
        (axes[0, 1], K_full_pred, full_vmax, "Learned fine-grid K"),
        (axes[1, 0], K_rom_true, rom_vmax, "Projected coarse K (sym.)"),
        (axes[1, 1], K_rom_pred, rom_vmax, "Learned ROM K"),
    ]
    for ax, matrix, vmax, title in entries:
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, shrink=0.78)
    fig.suptitle("Fine vs ROM operator matrices", fontsize=13)
    return base.maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


def plot_coarse_trace_comparison(
    *,
    true_rollout: dict,
    full_projected_rollout: dict,
    rom_rollout: dict,
    selected_nodes: OrderedDict[str, int],
    n_rom: int,
    output_stem: Path | None = None,
) -> str:
    times = true_rollout["times"]
    quantities = [
        ("Displacement u", "U_full"),
        ("Velocity v", "V_full"),
        ("Acceleration a", "A_full"),
    ]
    fig, axes = plt.subplots(
        len(selected_nodes),
        len(quantities),
        figsize=(15.8, 2.7 * len(selected_nodes)),
        sharex=True,
        constrained_layout=True,
    )
    if len(selected_nodes) == 1:
        axes = np.asarray([axes])

    for row_idx, (node_label, node) in enumerate(selected_nodes.items()):
        node_desc = base.describe_node(node, n_rom)
        for col_idx, (quantity_label, key) in enumerate(quantities):
            ax = axes[row_idx, col_idx]
            ax.plot(times, true_rollout[key][:, node], color="#2563eb", lw=1.9, label="True coarse data")
            ax.plot(times, full_projected_rollout[key][:, node], color="#d97706", lw=1.4, linestyle="--", label="Full model projected")
            ax.plot(times, rom_rollout[key][:, node], color="#059669", lw=1.4, linestyle=":", label="ROM")
            if row_idx == 0:
                ax.set_title(quantity_label)
            if col_idx == 0:
                ax.set_ylabel(f"{node_label}\n{node_desc}")
            ax.grid(alpha=0.25)

    for ax in axes[-1]:
        ax.set_xlabel("Time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Coarse observable traces: true vs projected full model vs ROM", fontsize=13)
    return base.maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


def plot_coarse_error_maps(
    *,
    true_rollout: dict,
    full_projected_rollout: dict,
    rom_rollout: dict,
    n_rom: int,
    output_stem: Path | None = None,
) -> str:
    err_full_u = np.sqrt(np.mean((full_projected_rollout["U_full"] - true_rollout["U_full"]) ** 2, axis=0))
    err_rom_u = np.sqrt(np.mean((rom_rollout["U_full"] - true_rollout["U_full"]) ** 2, axis=0))
    err_full_v = np.sqrt(np.mean((full_projected_rollout["V_full"] - true_rollout["V_full"]) ** 2, axis=0))
    err_rom_v = np.sqrt(np.mean((rom_rollout["V_full"] - true_rollout["V_full"]) ** 2, axis=0))

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0), constrained_layout=True)
    u_vmax = max(float(err_full_u.max()), float(err_rom_u.max()), 1e-10)
    v_vmax = max(float(err_full_v.max()), float(err_rom_v.max()), 1e-10)
    panels = [
        (axes[0, 0], err_full_u, u_vmax, "Projected full-model displacement RMSE"),
        (axes[0, 1], err_rom_u, u_vmax, "ROM displacement RMSE"),
        (axes[1, 0], err_full_v, v_vmax, "Projected full-model velocity RMSE"),
        (axes[1, 1], err_rom_v, v_vmax, "ROM velocity RMSE"),
    ]
    for ax, values, vmax, title in panels:
        image = ax.imshow(base.reshape_node_values(values, n_rom), origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.colorbar(image, ax=ax, shrink=0.82)
    fig.suptitle("Coarse-space RMS rollout errors", fontsize=13)
    return base.maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


def plot_lifted_snapshot_comparison(
    *,
    true_rollout: dict,
    full_rollout: dict,
    rom_lifted_rollout: dict,
    n_fine: int,
    output_stem: Path | None = None,
) -> str:
    times = true_rollout["times"]
    idxs = base.pick_time_indices(times, [1.0 / 8.0, 3.0 / 8.0, 5.0 / 8.0, 7.0 / 8.0])
    arrays = [
        ("True / data", np.abs(true_rollout["U_full"])),
        ("Full model", np.abs(full_rollout["U_full"])),
        ("Lifted ROM", np.abs(rom_lifted_rollout["U_full"])),
    ]
    vmax = max(float(np.max(arr)) for _, arr in arrays)
    fig, axes = plt.subplots(3, 4, figsize=(15.5, 10.4), constrained_layout=True)
    image = None
    for row_idx, (row_name, series) in enumerate(arrays):
        for col_idx, frame_idx in enumerate(idxs):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                base.reshape_node_values(series[frame_idx], n_fine),
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_title(f"{row_name}, t = {times[frame_idx]:.2f}")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.94)
        cbar.set_label("Displacement magnitude")
    fig.suptitle(
        "Fine-grid displacement snapshots: true data, learned fine model, and lifted ROM",
        fontsize=13,
    )
    return base.maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


def build_summary_table(
    *,
    full_name: str,
    rom_name: str,
    full_result: dict,
    rom_result: dict,
    full_problem: dict,
    rom_problem: dict,
    full_rollout: dict,
    fine_true_rollout: dict,
    full_projected_rollout: dict,
    rom_rollout: dict,
    rom_true_rollout: dict,
    rom_lifted_rollout: dict,
) -> str:
    def active_params(problem: dict) -> int:
        return int(problem["xi_mask"].sum().item())

    def full_u_mse(pred: dict, true: dict) -> float:
        return float(np.mean((pred["U_full"] - true["U_full"]) ** 2))

    def full_v_mse(pred: dict, true: dict) -> float:
        return float(np.mean((pred["V_full"] - true["V_full"]) ** 2))

    rows = [
        "<tr>"
        "<th class=\"label\">Model</th>"
        "<th>State DOFs</th>"
        "<th>Active Xi entries</th>"
        "<th>Native train MSE</th>"
        "<th>Native test MSE</th>"
        "<th>Native rel. K error</th>"
        "<th>Coarse rollout u MSE</th>"
        "<th>Coarse rollout v MSE</th>"
        "<th>Fine rollout u MSE</th>"
        "</tr>"
    ]
    rows.append(
        f"<tr><td class=\"label\">{full_name}</td>"
        f"<td>{full_problem['n_states']}</td>"
        f"<td>{active_params(full_problem)}</td>"
        f"<td>{full_result['train_losses'][-1]:.3e}</td>"
        f"<td>{full_result['test_losses'][-1]:.3e}</td>"
        f"<td>{full_result['snap_K_err'][-1]:.3e}</td>"
        f"<td>{full_u_mse(full_projected_rollout, rom_true_rollout):.3e}</td>"
        f"<td>{full_v_mse(full_projected_rollout, rom_true_rollout):.3e}</td>"
        f"<td>{full_u_mse(full_rollout, fine_true_rollout):.3e}</td></tr>"
    )
    rows.append(
        f"<tr><td class=\"label\">{rom_name}</td>"
        f"<td>{rom_problem['n_states']}</td>"
        f"<td>{active_params(rom_problem)}</td>"
        f"<td>{rom_result['train_losses'][-1]:.3e}</td>"
        f"<td>{rom_result['test_losses'][-1]:.3e}</td>"
        f"<td>{rom_result['snap_K_err'][-1]:.3e}</td>"
        f"<td>{full_u_mse(rom_rollout, rom_true_rollout):.3e}</td>"
        f"<td>{full_v_mse(rom_rollout, rom_true_rollout):.3e}</td>"
        f"<td>{full_u_mse(rom_lifted_rollout, fine_true_rollout):.3e}</td></tr>"
    )
    return "<table>" + "".join(rows) + "</table>"


def build_model_diagnostic_block(
    *,
    dataset_view: dict,
    result: dict,
    model_name: str,
    out_path: Path,
    rollout_force_fn,
    section_slug: str,
    selected_nodes: OrderedDict[str, int],
    incident_specs: list[dict],
) -> tuple[str, str]:
    output_dirs = base.build_output_dirs(out_path)
    figure_dir = output_dirs["figure_dir"]
    animation_dir = output_dirs["animation_dir"]
    n = dataset_view["n"]
    free = dataset_view["free"]
    n_total = dataset_view["N"]
    K_true_full = dataset_view["K_full"]
    K_true_free = dataset_view["K_free"]
    C_full = dataset_view["C_full"]
    base_traj = dataset_view["trajectories"][0]

    K_pred_free = base.xi_to_K(result["xi"]).detach().cpu().numpy()
    pred_rollout = base.rollout_linear_model(
        K_pred_free,
        free,
        n,
        dt=dataset_view["dt"],
        t_end=dataset_view["t_end"],
        u0_full=base_traj["u0_full"],
        v0_full=base_traj["v0_full"],
        device=torch.device("cpu"),
        simulation_backend="scipy",
        dtype=torch.float64,
        damping_matrix=C_full,
        force_fn=rollout_force_fn,
    )

    lattice_assets = base.plot_lattice_stiffness(
        K_true_full,
        n,
        figure_dir / f"{section_slug}_true_stiffness_lattice",
    )
    quiver_assets = base.make_quiver_snapshot_figure(
        base_traj,
        pred_rollout,
        n,
        model_name,
        figure_dir / f"{section_slug}_quiver_snapshots",
    )
    quiver_vmax = base.compute_visual_max(base_traj["U_full"], pred_rollout["U_full"])
    quiver_vis_scale = 0.75 / quiver_vmax
    anim_true_quiver = base.make_quiver_animation(
        base_traj,
        n,
        f"True scalar-deflection quiver, {model_name}",
        animation_dir / f"{section_slug}_true_quiver",
        vis_scale=quiver_vis_scale,
    )
    anim_pred_quiver = base.make_quiver_animation(
        pred_rollout,
        n,
        f"Model quiver, {model_name}",
        animation_dir / f"{section_slug}_pred_quiver",
        vis_scale=quiver_vis_scale,
    )

    heatmap_assets = base.make_heatmap_snapshot_figure(
        base_traj,
        pred_rollout,
        n,
        model_name,
        figure_dir / f"{section_slug}_heatmap_snapshots",
    )
    heatmap_vmax = base.compute_visual_max(
        np.abs(base_traj["U_full"]),
        np.abs(pred_rollout["U_full"]),
    )
    anim_true_heat = base.make_heatmap_animation(
        base_traj,
        n,
        f"True displacement heatmap, {model_name}",
        animation_dir / f"{section_slug}_true_heatmap",
        vmax=heatmap_vmax,
    )
    anim_pred_heat = base.make_heatmap_animation(
        pred_rollout,
        n,
        f"Model displacement heatmap, {model_name}",
        animation_dir / f"{section_slug}_pred_heatmap",
        vmax=heatmap_vmax,
    )

    traces_assets = base.make_node_trace_plots(
        base_traj,
        pred_rollout,
        selected_nodes,
        n,
        model_name,
        figure_dir / f"{section_slug}_node_traces",
    )
    param_note = None
    if result.get("solver_kind") == "stls":
        param_note = (
            "Iteration 0 is the initial masked least-squares solve; later "
            "iterations are threshold/refit STLS iterates."
        )
    param_assets = base.make_parameter_history_plots(
        model_name,
        selected_nodes,
        incident_specs,
        K_true_full,
        result["parameter_history"],
        result["parameter_history_epochs"],
        n,
        figure_dir / f"{section_slug}_parameter_history",
        note=param_note,
    )
    modal_assets = base.make_modal_comparison(
        model_name,
        K_true_free,
        K_pred_free,
        figure_dir / f"{section_slug}_modal_comparison",
    )
    mode_shape_assets = base.make_mode_shape_examples(
        model_name,
        K_true_free,
        K_pred_free,
        n,
        free,
        figure_dir / f"{section_slug}_mode_shapes",
    )

    block = "\n".join([
        "<div class=\"method-block\">",
        f"<h3>{model_name}</h3>",
        "<h4>Lattice schematic</h4>",
        base.render_image_tag(out_path.parent, lattice_assets["png"], f"True lattice for {model_name}"),
        "<h4>Static quiver snapshots</h4>",
        base.render_image_tag(out_path.parent, quiver_assets["png"], f"Quiver snapshots for {model_name}"),
        "<h4>Quiver animations</h4>",
        "<div class=\"media-grid\">",
        base.render_video_tag(out_path.parent, anim_true_quiver, "True / data quiver"),
        base.render_video_tag(out_path.parent, anim_pred_quiver, "Model quiver"),
        "</div>",
        "<h4>Static heatmap snapshots</h4>",
        base.render_image_tag(out_path.parent, heatmap_assets["png"], f"Heatmap snapshots for {model_name}"),
        "<h4>Heatmap animations</h4>",
        "<div class=\"media-grid\">",
        base.render_video_tag(out_path.parent, anim_true_heat, "True / data heatmap"),
        base.render_video_tag(out_path.parent, anim_pred_heat, "Model heatmap"),
        "</div>",
        "<h4>Selected traces</h4>",
        base.render_image_tag(out_path.parent, traces_assets["png"], f"Node traces for {model_name}"),
        "<h4>Local stiffness evolution</h4>",
        base.render_image_tag(out_path.parent, param_assets["png"], f"Parameter history for {model_name}"),
        "<h4>Modal comparison</h4>",
        base.render_image_tag(out_path.parent, modal_assets["png"], f"Modal comparison for {model_name}"),
        "<h5>Representative mode shapes</h5>",
        base.render_image_tag(out_path.parent, mode_shape_assets["png"], f"Mode shapes for {model_name}"),
        "</div>",
    ])
    note = (
        f"<li><code>{model_name}</code> diagnostic nodes/cells: "
        f"{base.build_node_summary_list(selected_nodes, n)}</li>"
    )
    return block, note


def main(
    *,
    n: int = 10,
    block_size: int = 2,
    n_traj: int = 4,
    t_end: float = 20.0,
    dt: float = 0.01,
    residual_epochs: int = 600,
    solver: str = "adam",
    l1_lambda: float = 0.0,
    alpha: float = 0.05,
    beta: float = 0.02,
    force_amplitude: float = 1.0,
    force_omega: float = 2.0 * np.pi,
    force_sigma_cols: float = 1.5,
    out_path: Path = DEFAULT_OUT_PATH,
) -> Path:
    dtype = torch.float64
    device = torch.device("cpu")

    if solver not in {"adam", "stls"}:
        raise ValueError("solver must be one of: 'adam', 'stls'")
    if l1_lambda < 0.0:
        raise ValueError("l1_lambda must be nonnegative")

    full_name = FULL_MODEL_NAME_TEMPLATES[solver]
    rom_name = ROM_MODEL_NAME_TEMPLATES[solver].format(block_size=block_size)
    if solver == "adam" and l1_lambda > 0.0:
        l1_label = f"L1 = {l1_lambda:.1e}"
        full_name = f"{full_name}, {l1_label}"
        rom_name = f"{rom_name}, {l1_label}"
    K_phys = forced.build_physical_K_clamped_top(n)
    C_phys = forced.build_rayleigh_damping(K_phys, alpha=alpha, beta=beta)
    force_fns = forced.build_per_trajectory_force_fns(
        n,
        n_traj=n_traj,
        amplitude=force_amplitude,
        base_omega=force_omega,
        sigma_cols=force_sigma_cols,
    )

    print("\n=== Spring-Grid ROM vs Full Report (CPU) ===")
    print(
        f"  fine grid = {n}x{n}, block_size = {block_size}, "
        f"n_traj = {n_traj}, t_end = {t_end}, dt = {dt}"
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
    rom_data = base.build_block_average_rom_dataset(data, block_size=block_size)
    rom_view = build_rom_diagnostic_view(rom_data)
    reduction = build_block_average_reduction(n, block_size, free_dofs=data["free"])
    rom_force_fn = make_reduced_force_fn(force_fns[0], reduction)

    full_problem = base.build_regression_problem(data, device=device, dtype=dtype)
    rom_problem = base.build_regression_problem(
        rom_data,
        device=device,
        dtype=dtype,
        state_key="U_rom",
        target_key="A_target_rom",
        stiffness_key="K_rom",
        mask_key="locality_rom",
    )

    full_selected_nodes = base.select_representative_nodes(
        data["n"],
        data["free"],
        data["trajectories"][0]["U_full"],
    )
    rom_selected_nodes = select_diagnostic_nodes_flexible(
        rom_view["n"],
        rom_view["free"],
        rom_view["trajectories"][0]["U_full"],
    )
    full_result, full_incident_specs = fit_regression_model(
        solver=solver,
        problem=full_problem,
        selected_nodes=full_selected_nodes,
        n=data["n"],
        free=data["free"],
        n_total_dofs=data["N"],
        residual_epochs=residual_epochs,
        l1_lambda=l1_lambda,
        n_eig=5,
    )
    rom_result, rom_incident_specs = fit_regression_model(
        solver=solver,
        problem=rom_problem,
        selected_nodes=rom_selected_nodes,
        n=rom_view["n"],
        free=rom_view["free"],
        n_total_dofs=rom_view["N"],
        residual_epochs=residual_epochs,
        l1_lambda=l1_lambda,
        n_eig=5,
    )

    full_K_pred = base.xi_to_K(full_result["xi"]).detach().cpu().numpy()
    rom_K_pred = base.xi_to_K(rom_result["xi"]).detach().cpu().numpy()
    fine_true_rollout = data["trajectories"][0]
    rom_true_rollout = rom_view["trajectories"][0]
    full_rollout = base.rollout_linear_model(
        full_K_pred,
        data["free"],
        data["n"],
        dt=data["dt"],
        t_end=data["t_end"],
        u0_full=fine_true_rollout["u0_full"],
        v0_full=fine_true_rollout["v0_full"],
        device=device,
        simulation_backend="scipy",
        dtype=dtype,
        damping_matrix=data["C_full"],
        force_fn=force_fns[0],
    )
    rom_rollout = base.rollout_linear_model(
        rom_K_pred,
        rom_view["free"],
        rom_view["n"],
        dt=rom_view["dt"],
        t_end=rom_view["t_end"],
        u0_full=rom_true_rollout["u0_full"],
        v0_full=rom_true_rollout["v0_full"],
        device=device,
        simulation_backend="scipy",
        dtype=dtype,
        damping_matrix=rom_view["C_full"],
        force_fn=rom_force_fn,
    )
    full_projected_rollout = project_fine_rollout_to_rom(
        full_rollout,
        reduction,
        data["free"],
        rom_view["n"],
    )
    rom_lifted_rollout = lift_rom_rollout_to_fine(
        rom_rollout,
        rom_data["P_block"],
        data["free"],
        data["N"],
    )

    output_dirs = base.build_output_dirs(out_path)
    x_label = "Epoch"
    title_x_name = "epoch"
    iteration_axis_heading = "Epoch"
    if solver == "stls":
        x_label = "STLS iteration"
        title_x_name = "STLS iteration"
        iteration_axis_heading = "STLS Iteration"

    loss_curve_b64 = base.plot_loss_curves(
        OrderedDict([
            (full_name, full_result),
            (rom_name, rom_result),
        ]),
        output_stem=output_dirs["figure_dir"] / "native_loss_curves",
        x_label=x_label,
    )
    k_err_b64 = base.plot_K_error(
        OrderedDict([
            (full_name, full_result),
            (rom_name, rom_result),
        ]),
        output_stem=output_dirs["figure_dir"] / "relative_k_error",
        x_label=x_label,
        title_x_name=title_x_name,
    )
    operator_matrix_b64 = plot_operator_matrix_overview(
        K_full_true=data["K_free"],
        K_full_pred=full_K_pred,
        K_rom_true=rom_data["K_rom"],
        K_rom_pred=rom_K_pred,
        output_stem=output_dirs["figure_dir"] / "operator_matrix_overview",
    )
    coarse_trace_b64 = plot_coarse_trace_comparison(
        true_rollout=rom_true_rollout,
        full_projected_rollout=full_projected_rollout,
        rom_rollout=rom_rollout,
        selected_nodes=rom_selected_nodes,
        n_rom=rom_view["n"],
        output_stem=output_dirs["figure_dir"] / "coarse_trace_comparison",
    )
    coarse_error_b64 = plot_coarse_error_maps(
        true_rollout=rom_true_rollout,
        full_projected_rollout=full_projected_rollout,
        rom_rollout=rom_rollout,
        n_rom=rom_view["n"],
        output_stem=output_dirs["figure_dir"] / "coarse_error_maps",
    )
    lifted_snapshot_b64 = plot_lifted_snapshot_comparison(
        true_rollout=fine_true_rollout,
        full_rollout=full_rollout,
        rom_lifted_rollout=rom_lifted_rollout,
        n_fine=data["n"],
        output_stem=output_dirs["figure_dir"] / "lifted_snapshot_comparison",
    )

    eigen_blocks = []
    for name, result in ((full_name, full_result), (rom_name, rom_result)):
        b64 = base.plot_eigenvalues(
            name,
            result,
            output_stem=output_dirs["figure_dir"] / f"eigenvalues_{base.slugify(name)}",
            x_label=x_label,
            title_x_name=title_x_name,
        )
        eigen_blocks.append(
            f'<div class="method-block"><h3>{name}</h3><img src="data:image/png;base64,{b64}" /></div>'
        )

    full_block, full_note = build_model_diagnostic_block(
        dataset_view=data,
        result=full_result,
        model_name=full_name,
        out_path=out_path,
        rollout_force_fn=force_fns[0],
        section_slug="full_model",
        selected_nodes=full_selected_nodes,
        incident_specs=full_incident_specs,
    )
    rom_block, rom_note = build_model_diagnostic_block(
        dataset_view=rom_view,
        result=rom_result,
        model_name=rom_name,
        out_path=out_path,
        rollout_force_fn=rom_force_fn,
        section_slug="rom_model",
        selected_nodes=rom_selected_nodes,
        incident_specs=rom_incident_specs,
    )

    summary_table = build_summary_table(
        full_name=full_name,
        rom_name=rom_name,
        full_result=full_result,
        rom_result=rom_result,
        full_problem=full_problem,
        rom_problem=rom_problem,
        full_rollout=full_rollout,
        fine_true_rollout=fine_true_rollout,
        full_projected_rollout=full_projected_rollout,
        rom_rollout=rom_rollout,
        rom_true_rollout=rom_true_rollout,
        rom_lifted_rollout=rom_lifted_rollout,
    )

    compression_ratio = full_problem["n_states"] / rom_problem["n_states"]
    param_ratio = int(full_problem["xi_mask"].sum().item()) / int(rom_problem["xi_mask"].sum().item())
    summary_notes = "\n".join([
        "<ul>",
        f"<li>Report written to <code>{out_path}</code>.</li>",
        f"<li>Static figures saved under <code>{output_dirs['figure_dir']}</code>.</li>",
        f"<li>Animations saved under <code>{output_dirs['animation_dir']}</code>.</li>",
        f"<li>State compression ratio (fine / ROM): <code>{compression_ratio:.2f}x</code>.</li>",
        f"<li>Active Xi-entry compression ratio under locality masks: <code>{param_ratio:.2f}x</code>.</li>",
        f"<li>The ROM uses active-node-normalized block averages, so coarse blocks touching the clamped edge average only the active fine DOFs in those blocks.</li>",
        f"<li>The projected coarse operator stored in the dataset is symmetrized for physical diagnostics; the raw projected matrices are available as <code>K_rom_raw</code> and <code>C_rom_raw</code>.</li>",
        f"{full_note}",
        f"{rom_note}",
        "</ul>",
    ])
    if solver == "adam":
        if l1_lambda > 0.0:
            solver_description = (
                f"locality-masked Adam on a linear library with L1 penalty "
                f"<code>lambda = {l1_lambda:.1e}</code>"
            )
            report_title = (
                "Spring-Grid ROM vs Full Report "
                "(10x10 Physical, Damping + Bottom Forcing, Adam + L1)"
            )
        else:
            solver_description = "locality-masked Adam on a linear library"
            report_title = "Spring-Grid ROM vs Full Report (10x10 Physical, Damping + Bottom Forcing)"
        native_diagnostic_note = ""
    else:
        solver_description = "locality-masked STLS on the linear regression problem"
        native_diagnostic_note = (
            "<p>For STLS, iteration 0 is the initial masked least-squares solve. "
            "Iterations 1..10 are the sequential threshold/refit steps. The loss, "
            "relative-K-error, eigenvalue, and local-stiffness panels are plotted "
            "against those actual STLS iterations.</p>"
        )
        report_title = "Spring-Grid ROM vs Full STLS Report (10x10 Physical, Damping + Bottom Forcing)"

    html = HTML_TEMPLATE.format(
        report_title=report_title,
        n=data["n"],
        n_free=data["free"].size,
        block_size=block_size,
        n_rom=rom_view["n"],
        n_rom_dofs=rom_view["N"],
        solver_description=solver_description,
        summary_table=summary_table,
        native_diagnostic_note=native_diagnostic_note,
        iteration_axis_heading=iteration_axis_heading,
        loss_curve_b64=loss_curve_b64,
        k_err_b64=k_err_b64,
        operator_matrix_b64=operator_matrix_b64,
        eigenvalue_blocks="\n".join(eigen_blocks),
        coarse_trace_b64=coarse_trace_b64,
        coarse_error_b64=coarse_error_b64,
        lifted_snapshot_b64=lifted_snapshot_b64,
        full_model_block=full_block,
        rom_model_block=rom_block,
        summary_notes=summary_notes,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  Wrote report to: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--n-traj", type=int, default=4)
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--residual-epochs", type=int, default=600)
    parser.add_argument("--solver", choices=("adam", "stls"), default="adam")
    parser.add_argument("--l1-lambda", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--force-amplitude", type=float, default=1.0)
    parser.add_argument("--force-omega", type=float, default=2.0 * np.pi)
    parser.add_argument("--force-sigma-cols", type=float, default=1.5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    main(
        n=args.n,
        block_size=args.block_size,
        n_traj=args.n_traj,
        t_end=args.t_end,
        dt=args.dt,
        residual_epochs=args.residual_epochs,
        solver=args.solver,
        l1_lambda=args.l1_lambda,
        alpha=args.alpha,
        beta=args.beta,
        force_amplitude=args.force_amplitude,
        force_omega=args.force_omega,
        force_sigma_cols=args.force_sigma_cols,
        out_path=args.out,
    )
