"""Frequency response prediction for the spring-grid Full and ROM regressions.

Trains the same Full + ROM STLS models that the rom-vs-full report uses,
then sweeps a single forcing frequency `omega` applied uniformly across the
bottom row of the lattice and predicts the steady-state amplitude for:

  * the true physical operator,
  * the regressed Full operator (fine grid),
  * the regressed ROM operator (block-averaged, lifted back to fine).

The response is computed analytically in the frequency domain by solving
`(K - omega**2 M + i omega C) U = F0` for each omega. Output is a self-
contained HTML report with response curves, per-DOF amplitude/phase maps at
the dominant resonance, error curves, and quiver/heatmap animations of the
steady-state oscillation at three representative frequencies (small,
medium, and big response on the true model).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import spring_grid_K_regression as base
import generate_spring_grid_physical_forced_report as forced
import generate_spring_grid_rom_vs_full_report as rom_vs_full
from sindy_torch.systems import build_block_average_reduction
from sindy_torch.analysis import (
    amplitude_metrics,
    bottom_row_uniform_force,
    compute_frequency_response,
    pick_animation_frequencies,
    steady_state_rollout,
)


DEFAULT_OUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "figures"
    / "spring_grid_10x10_frequency_response_report.html"
)


def add_measurement_noise(
    data: dict,
    *,
    noise_std: float,
    seed: int = 0,
) -> dict:
    """Add per-signal relative Gaussian noise to U/V/A and recompute A_target.

    The noise standard deviation per signal is ``noise_std * rms(signal)``,
    matching the usual relative-noise convention in SINDy-style studies. The
    regression target ``A_target = A + V @ C_free.T - F`` is recomputed from
    the noisy state so the noise propagates correctly into the regression.

    A ``noise_std <= 0`` is a no-op and the original ``data`` is returned.
    """
    if noise_std is None or noise_std <= 0.0:
        return data
    rng = np.random.default_rng(seed)
    C_free = np.asarray(data["C_free"])
    free = np.asarray(data["free"], dtype=int)

    U_list, V_list, A_list, Atgt_list = [], [], [], []
    new_trajectories = []
    for traj in data["trajectories"]:
        U = np.asarray(traj["U_free"])
        V = np.asarray(traj["V_free"])
        A = np.asarray(traj["A_free"])
        F = np.asarray(traj["F_free"])
        u_rms = float(np.sqrt(np.mean(U ** 2))) + 1e-30
        v_rms = float(np.sqrt(np.mean(V ** 2))) + 1e-30
        a_rms = float(np.sqrt(np.mean(A ** 2))) + 1e-30
        U_n = U + noise_std * u_rms * rng.standard_normal(U.shape)
        V_n = V + noise_std * v_rms * rng.standard_normal(V.shape)
        A_n = A + noise_std * a_rms * rng.standard_normal(A.shape)
        A_target_n = A_n + V_n @ C_free.T - F

        U_full_n = traj["U_full"].copy()
        V_full_n = traj["V_full"].copy()
        A_full_n = traj["A_full"].copy()
        U_full_n[:, free] = U_n
        V_full_n[:, free] = V_n
        A_full_n[:, free] = A_n

        new_traj = dict(traj)
        new_traj.update({
            "U_free": U_n, "V_free": V_n, "A_free": A_n, "A_target": A_target_n,
            "U_full": U_full_n, "V_full": V_full_n, "A_full": A_full_n,
        })
        new_trajectories.append(new_traj)
        U_list.append(U_n); V_list.append(V_n); A_list.append(A_n); Atgt_list.append(A_target_n)

    out = dict(data)
    out["U"] = np.concatenate(U_list, axis=0)
    out["V"] = np.concatenate(V_list, axis=0)
    out["A"] = np.concatenate(A_list, axis=0)
    out["A_target"] = np.concatenate(Atgt_list, axis=0)
    out["trajectories"] = new_trajectories
    out["measurement_noise_std"] = float(noise_std)
    return out


MODEL_COLORS = {
    "true": "#000000",
    "full": "#1d4ed8",
    "rom": "#dc2626",
}
MODEL_LABELS = {
    "true": "True physical model",
    "full": "Regressed Full (STLS)",
    "rom": "Regressed ROM (STLS, lifted)",
}
# Plot styles: True is wide and underneath; Full is dashed and overlays;
# ROM is thin solid. With this ordering a near-perfect Full regression
# shows up as a dashed blue line tracking the wide black True curve.
MODEL_LINE_STYLES = {
    "true": {"lw": 3.6, "ls": "-",  "alpha": 0.9, "zorder": 2},
    "full": {"lw": 1.6, "ls": "--", "alpha": 1.0, "zorder": 4},
    "rom":  {"lw": 1.8, "ls": "-",  "alpha": 1.0, "zorder": 3},
}
PLOT_ORDER = ("true", "rom", "full")  # True drawn first, Full drawn last (on top)


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
.video-card h5 {{ margin: 0 0 0.4em 0; font-size: 0.95rem; }}
.note-box {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.9em 1em; border-radius: 6px; }}
.freq-block {{ margin-bottom: 2.2em; }}
</style>
</head>
<body>
<h1>{report_title}</h1>

<h2>Setup</h2>
<ul>
<li>Fine grid: <code>{n}x{n}</code> with <code>{n_free}</code> free DOFs (top row clamped).</li>
<li>ROM: block size <code>{block_size}</code>, coarse grid <code>{n_rom}x{n_rom}</code>, <code>{n_rom_dofs}</code> DOFs.</li>
<li>Both regressed operators are fit by locality-masked STLS on the linear regression problem <code>a + C v - f = -K u</code>, using the same multi-frequency bottom-row training forcing as the rom-vs-full report.</li>
<li>Frequency response forcing is held fixed at unit amplitude on every bottom-row free DOF (in phase). The ROM forcing is the block-average restriction of this fine-grid forcing.</li>
<li>Mass: <code>M = I</code>. Damping: Rayleigh, <code>C = alpha M + beta K_phys</code> with <code>alpha = {alpha}</code>, <code>beta = {beta}</code>. The same physical damping is paired with each predicted K when computing its response (damping is treated as known).</li>
<li>Measurement noise on training data: {noise_description}</li>
<li>Steady-state response is computed analytically by solving <code>(K - omega^2 M + i omega C) U = F0</code> at each omega.</li>
</ul>

<h2>Response curves</h2>
<div class="note-box">
<p>Three amplitude metrics over the same omega sweep. <code>L2</code> is the
Euclidean norm of <code>|U(omega)|</code> over all 90 fine free DOFs (ROM is
lifted to fine via block-constant prolongation before measurement);
<code>max</code> is the peak DOF amplitude; <code>top-near</code> is the mean
amplitude on the row just below the clamped top. Vertical guides mark the
true natural frequencies <code>sqrt(eig(K_true))</code>.</p>
<p><b>Reading the curves:</b> the True model is plotted as a wide solid black
line. The Regressed Full overlays it as a dashed blue line, and the Regressed
ROM is a solid red line. When the Full regression converges to the true
operator, the dashed blue line lies directly on top of the black True line.
Quantitative deviation is shown in the dedicated error figure below.</p>
</div>
<img src="data:image/png;base64,{response_curves_b64}" />

<h2>Headline: L2 response with eigenfrequency overlay</h2>
<img src="data:image/png;base64,{headline_curve_b64}" />

<h2>Response error vs omega</h2>
<div class="note-box">
<p>For each amplitude metric, <code>|metric_pred(omega) - metric_true(omega)|</code> for the Full and ROM regressed models.</p>
</div>
<img src="data:image/png;base64,{error_curves_b64}" />

<h2>Per-DOF amplitude at the resonance peak</h2>
<div class="note-box">
<p>Steady-state displacement magnitude <code>|U_j(omega_peak)|</code> on the
fine grid for each of the three models. <code>omega_peak = {omega_peak:.3f}</code>
(argmax of true L2 response).</p>
</div>
<img src="data:image/png;base64,{amplitude_map_b64}" />

<h2>Per-DOF phase at the resonance peak</h2>
<img src="data:image/png;base64,{phase_map_b64}" />

<h2>Selected animation frequencies</h2>
<table>
<tr><th class="label">Class</th><th>omega</th><th>True L2 amplitude</th></tr>
{animation_table_rows}
</table>

<h2>Steady-state animations</h2>
<div class="note-box">
<p>For each of the three selected frequencies, the steady-state response
<code>u(t) = Re(U(omega) e^(i omega t))</code> is rendered for two periods of
oscillation. Quiver arrows show the scalar deflection field as a vertical
out-of-plane component on the lattice; heatmaps show <code>|u(x, t)|</code>.
The "lifted ROM" frames are block-constant on each <code>{block_size}x{block_size}</code> block
by construction of the prolongation operator.</p>
</div>
{animation_blocks}

<h2>Notes</h2>
<ul>
<li>HTML report: <code>{out_path}</code></li>
<li>Static figures: <code>{figure_dir}</code></li>
<li>Animations: <code>{animation_dir}</code></li>
<li>Sweep: <code>{n_omega}</code> log-spaced points from <code>omega = {omega_min:.3e}</code> to <code>omega = {omega_max:.3e}</code>.</li>
<li>Static-limit sanity check (omega -&gt; 0): residual <code>{static_residual:.2e}</code> against direct solve <code>K^-1 F0</code> on the true model.</li>
</ul>
</body>
</html>
"""


def assemble_response_curve_figure(
    *,
    omegas: np.ndarray,
    metrics_by_model: dict[str, dict[str, np.ndarray]],
    output_stem: Path | None,
) -> str:
    metric_keys = ["l2", "max", "top_row_mean"]
    metric_titles = {"l2": "L2 norm of |U|", "max": "Max |U_j|", "top_row_mean": "Top-near row mean |U|"}
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(16.5, 4.4), constrained_layout=True)
    for ax, key in zip(axes, metric_keys):
        for model_id in PLOT_ORDER:
            curve = metrics_by_model[model_id].get(key)
            if curve is None:
                continue
            style = MODEL_LINE_STYLES[model_id]
            ax.plot(omegas, curve, color=MODEL_COLORS[model_id],
                    label=MODEL_LABELS[model_id], **style)
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel("Amplitude")
        ax.set_title(metric_titles[key])
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Steady-state amplitude vs forcing frequency", fontsize=13)
    return base.maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


def assemble_headline_curve(
    *,
    omegas: np.ndarray,
    metrics_by_model: dict[str, dict[str, np.ndarray]],
    natural_frequencies: np.ndarray,
    output_stem: Path | None,
) -> str:
    fig, ax = plt.subplots(figsize=(11.0, 5.0), constrained_layout=True)
    for w_n in natural_frequencies:
        ax.axvline(w_n, color="#94a3b8", lw=0.6, alpha=0.55, zorder=1)
    for model_id in PLOT_ORDER:
        style = MODEL_LINE_STYLES[model_id]
        ax.plot(
            omegas,
            metrics_by_model[model_id]["l2"],
            color=MODEL_COLORS[model_id],
            label=MODEL_LABELS[model_id],
            **style,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Forcing frequency $\omega$")
    ax.set_ylabel(r"$\| U(\omega) \|_2$")
    ax.set_title("Frequency response: L2 amplitude with true natural frequencies")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    return base.maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


def assemble_error_curves(
    *,
    omegas: np.ndarray,
    metrics_by_model: dict[str, dict[str, np.ndarray]],
    output_stem: Path | None,
) -> str:
    metric_keys = ["l2", "max", "top_row_mean"]
    metric_titles = {"l2": "L2 norm error", "max": "Max-DOF error", "top_row_mean": "Top-near row error"}
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(16.5, 4.4), constrained_layout=True)
    true_metrics = metrics_by_model["true"]
    for ax, key in zip(axes, metric_keys):
        for model_id in ("full", "rom"):
            pred = metrics_by_model[model_id].get(key)
            tru = true_metrics.get(key)
            if pred is None or tru is None:
                continue
            ax.plot(omegas, np.abs(pred - tru), color=MODEL_COLORS[model_id], lw=1.5,
                    label=MODEL_LABELS[model_id])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel("|metric_pred - metric_true|")
        ax.set_title(metric_titles[key])
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Frequency response error vs true model", fontsize=13)
    return base.maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


def assemble_per_dof_map(
    *,
    fields_by_model: dict[str, np.ndarray],
    n: int,
    title: str,
    cmap: str,
    output_stem: Path | None,
) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), constrained_layout=True)
    vmax = max(float(np.max(field)) for field in fields_by_model.values())
    vmin = min(float(np.min(field)) for field in fields_by_model.values())
    if cmap == "magma":
        vmin = 0.0
    image = None
    for ax, model_id in zip(axes, ("true", "full", "rom")):
        field = fields_by_model[model_id]
        image = ax.imshow(
            base.reshape_node_values(field, n),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(MODEL_LABELS[model_id])
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.9)
    fig.suptitle(title, fontsize=13)
    return base.maybe_save_for_html(fig, output_stem=output_stem, return_base64=True)


def make_side_by_side_quiver_animation(
    rollouts_by_model: dict[str, dict],
    n: int,
    title: str,
    output_stem: Path,
    vis_scale: float,
) -> dict[str, Path]:
    """3-panel quiver animation showing all three models in lockstep."""
    positions = base.grid_positions(n)
    edge_specs = base.build_physical_edge_specs(n)

    model_ids = ("true", "full", "rom")
    times = rollouts_by_model["true"]["times"]
    n_frames = len(times)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), constrained_layout=True)
    quivs = {}
    title_texts = {}
    for ax, model_id in zip(axes, model_ids):
        base.draw_lattice_background(ax, edge_specs, positions)
        ax.scatter(positions[:, 0], positions[:, 1], s=10, c="#334155", zorder=2)
        vals0 = rollouts_by_model[model_id]["U_full"][0]
        quivs[model_id] = ax.quiver(
            positions[:, 0],
            positions[:, 1],
            np.zeros(positions.shape[0]),
            vals0 * vis_scale,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.005,
            headwidth=3.8,
            headlength=4.8,
            headaxislength=4.5,
            color=MODEL_COLORS[model_id],
            zorder=3,
        )
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.grid(alpha=0.12)
        title_texts[model_id] = ax.set_title(f"{MODEL_LABELS[model_id]}\n t = {times[0]:.2f}")
    fig.suptitle(title, fontsize=13)

    def update(frame_id):
        for model_id in model_ids:
            vals = rollouts_by_model[model_id]["U_full"][frame_id]
            quivs[model_id].set_UVC(np.zeros_like(vals), vals * vis_scale)
            title_texts[model_id].set_text(f"{MODEL_LABELS[model_id]}\n t = {times[frame_id]:.2f}")
        return list(quivs.values()) + list(title_texts.values())

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=45, blit=False)
    return base.save_animation(anim, output_stem, fps=20)


def make_side_by_side_heatmap_animation(
    rollouts_by_model: dict[str, dict],
    n: int,
    title: str,
    output_stem: Path,
    vmax: float,
) -> dict[str, Path]:
    model_ids = ("true", "full", "rom")
    times = rollouts_by_model["true"]["times"]
    n_frames = len(times)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), constrained_layout=True)
    images = {}
    title_texts = {}
    for ax, model_id in zip(axes, model_ids):
        vals0 = np.abs(rollouts_by_model[model_id]["U_full"][0])
        images[model_id] = ax.imshow(
            base.reshape_node_values(vals0, n),
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        title_texts[model_id] = ax.set_title(f"{MODEL_LABELS[model_id]}\n t = {times[0]:.2f}")
    fig.colorbar(images["true"], ax=axes, shrink=0.9, label="|u|")
    fig.suptitle(title, fontsize=13)

    def update(frame_id):
        for model_id in model_ids:
            vals = np.abs(rollouts_by_model[model_id]["U_full"][frame_id])
            images[model_id].set_data(base.reshape_node_values(vals, n))
            title_texts[model_id].set_text(f"{MODEL_LABELS[model_id]}\n t = {times[frame_id]:.2f}")
        return list(images.values()) + list(title_texts.values())

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=45, blit=False)
    return base.save_animation(anim, output_stem, fps=20)


def lift_rom_complex_to_fine(
    U_rom: np.ndarray,
    P_block: np.ndarray,
    free: np.ndarray,
    n_total: int,
) -> np.ndarray:
    """Lift a complex (n_omega, n_rom) ROM response to (n_omega, n_total) fine."""
    U_free = U_rom @ P_block.T
    out = np.zeros((U_rom.shape[0], int(n_total)), dtype=complex)
    out[:, np.asarray(free, dtype=int)] = U_free
    return out


def main(
    *,
    n: int = 10,
    block_size: int = 2,
    n_traj: int = 4,
    t_end: float = 20.0,
    dt: float = 0.01,
    alpha: float = 0.05,
    beta: float = 0.02,
    force_amplitude: float = 1.0,
    force_omega: float = 2.0 * np.pi,
    force_sigma_cols: float = 1.5,
    n_omega: int = 400,
    omega_pad: float = 1.05,
    omega_floor_frac: float = 0.05,
    noise_std: float = 0.0,
    noise_seed: int = 0,
    out_path: Path = DEFAULT_OUT_PATH,
) -> Path:
    dtype = torch.float64
    device = torch.device("cpu")

    print("\n=== Spring-Grid Frequency Response Report (CPU) ===")
    print(f"  fine grid = {n}x{n}, block_size = {block_size}")

    K_phys = forced.build_physical_K_clamped_top(n)
    C_phys = forced.build_rayleigh_damping(K_phys, alpha=alpha, beta=beta)
    train_force_fns = forced.build_per_trajectory_force_fns(
        n,
        n_traj=n_traj,
        amplitude=force_amplitude,
        base_omega=force_omega,
        sigma_cols=force_sigma_cols,
    )

    print("  Generating training data...")
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
        force_fn=train_force_fns,
        pinned_rows=(n - 1,),
        initial_condition_mode="zero",
    )
    if noise_std > 0.0:
        print(f"  Injecting Gaussian measurement noise (relative std = {noise_std})...")
        data = add_measurement_noise(data, noise_std=noise_std, seed=noise_seed)
    rom_data = base.build_block_average_rom_dataset(data, block_size=block_size)
    rom_view = rom_vs_full.build_rom_diagnostic_view(rom_data)
    reduction = build_block_average_reduction(n, block_size, free_dofs=data["free"])

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
        data["n"], data["free"], data["trajectories"][0]["U_full"],
    )
    rom_selected_nodes = rom_vs_full.select_diagnostic_nodes_flexible(
        rom_view["n"], rom_view["free"], rom_view["trajectories"][0]["U_full"],
    )

    print("  Fitting STLS on the full grid...")
    full_result, _ = rom_vs_full.fit_regression_model(
        solver="stls", problem=full_problem,
        selected_nodes=full_selected_nodes,
        n=data["n"], free=data["free"], n_total_dofs=data["N"],
        residual_epochs=600, l1_lambda=0.0, n_eig=5,
    )
    print("  Fitting STLS on the ROM...")
    rom_result, _ = rom_vs_full.fit_regression_model(
        solver="stls", problem=rom_problem,
        selected_nodes=rom_selected_nodes,
        n=rom_view["n"], free=rom_view["free"], n_total_dofs=rom_view["N"],
        residual_epochs=600, l1_lambda=0.0, n_eig=5,
    )

    K_full_pred = base.xi_to_K(full_result["xi"]).detach().cpu().numpy()
    K_rom_pred = base.xi_to_K(rom_result["xi"]).detach().cpu().numpy()

    free = np.asarray(data["free"], dtype=int)
    n_free = free.size
    K_full_true = np.asarray(data["K_free"])
    C_full_true = np.asarray(data["C_free"])
    M_full = np.eye(n_free)

    rom_free = np.asarray(rom_view["free"], dtype=int)
    n_rom_dofs = rom_free.size
    K_rom_true_proj = np.asarray(rom_data["K_rom"])
    C_rom_proj = np.asarray(rom_data["C_rom"])
    M_rom = np.eye(n_rom_dofs)

    F0_full = bottom_row_uniform_force(n, free, amplitude=1.0)
    R_avg = np.asarray(reduction.restriction)
    P_block = np.asarray(reduction.prolongation)
    F0_rom = R_avg @ F0_full

    eig_K_true = np.linalg.eigvalsh(0.5 * (K_full_true + K_full_true.T))
    eig_K_true = np.clip(eig_K_true, 1e-12, None)
    natural_frequencies = np.sqrt(eig_K_true)
    omega_min_phys = float(natural_frequencies.min())
    omega_max_phys = float(natural_frequencies.max())
    omega_lo = max(omega_floor_frac * omega_min_phys, 1e-3)
    omega_hi = omega_pad * omega_max_phys
    omegas = np.logspace(np.log10(omega_lo), np.log10(omega_hi), n_omega)
    print(f"  Sweep: {n_omega} log-spaced omegas in [{omega_lo:.3e}, {omega_hi:.3e}]")

    print("  Computing frequency response: true...")
    U_true = compute_frequency_response(K_full_true, M_full, C_full_true, F0_full, omegas)
    print("  Computing frequency response: full regressed...")
    U_full = compute_frequency_response(K_full_pred, M_full, C_full_true, F0_full, omegas)
    print("  Computing frequency response: ROM regressed...")
    U_rom_native = compute_frequency_response(K_rom_pred, M_rom, C_rom_proj, F0_rom, omegas)
    U_rom_lifted = lift_rom_complex_to_fine(U_rom_native, P_block, free, data["N"])
    U_rom_lifted_free = U_rom_lifted[:, free]

    near_top_row = n - 2
    top_near_full_idx = np.array([near_top_row * n + j for j in range(n)], dtype=int)
    top_near_free_mask = np.isin(top_near_full_idx, free)
    top_near_free_global = top_near_full_idx[top_near_free_mask]
    top_near_free_local = np.array(
        [int(np.where(free == g)[0][0]) for g in top_near_free_global], dtype=int
    )

    metrics_by_model = {
        "true": amplitude_metrics(U_true, top_row_indices=top_near_free_local),
        "full": amplitude_metrics(U_full, top_row_indices=top_near_free_local),
        "rom": amplitude_metrics(U_rom_lifted_free, top_row_indices=top_near_free_local),
    }

    static_solve = np.linalg.solve(K_full_true, F0_full)
    static_residual = float(np.linalg.norm(np.real(U_true[0]) - static_solve) / max(np.linalg.norm(static_solve), 1e-30))
    print(f"  Static-limit residual (true): {static_residual:.3e}")

    output_dirs = base.build_output_dirs(out_path)
    figure_dir = output_dirs["figure_dir"]
    animation_dir = output_dirs["animation_dir"]

    print("  Building response curve figures...")
    response_curves_b64 = assemble_response_curve_figure(
        omegas=omegas, metrics_by_model=metrics_by_model,
        output_stem=figure_dir / "response_curves",
    )
    headline_curve_b64 = assemble_headline_curve(
        omegas=omegas, metrics_by_model=metrics_by_model,
        natural_frequencies=natural_frequencies,
        output_stem=figure_dir / "headline_l2_curve",
    )
    error_curves_b64 = assemble_error_curves(
        omegas=omegas, metrics_by_model=metrics_by_model,
        output_stem=figure_dir / "error_curves",
    )

    peak_idx = int(np.argmax(metrics_by_model["true"]["l2"]))
    omega_peak = float(omegas[peak_idx])
    print(f"  Resonance peak (true L2): omega = {omega_peak:.4f}")

    def expand_free_to_full(values_free: np.ndarray) -> np.ndarray:
        full = np.zeros(int(data["N"]), dtype=values_free.dtype)
        full[free] = values_free
        return full

    amp_fields = {
        "true": expand_free_to_full(np.abs(U_true[peak_idx])),
        "full": expand_free_to_full(np.abs(U_full[peak_idx])),
        "rom": np.abs(U_rom_lifted[peak_idx]),
    }
    phase_fields = {
        "true": expand_free_to_full(np.angle(U_true[peak_idx])),
        "full": expand_free_to_full(np.angle(U_full[peak_idx])),
        "rom": np.angle(U_rom_lifted[peak_idx]),
    }
    amplitude_map_b64 = assemble_per_dof_map(
        fields_by_model=amp_fields, n=n,
        title=f"|U_j(omega_peak)|, omega_peak = {omega_peak:.3f}",
        cmap="magma",
        output_stem=figure_dir / "amplitude_map_at_peak",
    )
    phase_map_b64 = assemble_per_dof_map(
        fields_by_model=phase_fields, n=n,
        title=f"arg U_j(omega_peak), omega_peak = {omega_peak:.3f}",
        cmap="twilight",
        output_stem=figure_dir / "phase_map_at_peak",
    )

    picks = pick_animation_frequencies(omegas, metrics_by_model["true"]["l2"])
    print(f"  Animation frequencies: small={picks['small']['omega']:.3f}, "
          f"medium={picks['medium']['omega']:.3f}, big={picks['big']['omega']:.3f}")

    animation_blocks = []
    table_rows = []
    for category in ("small", "medium", "big"):
        info = picks[category]
        idx = info["index"]
        w = info["omega"]
        slug = f"omega_{category}"
        table_rows.append(
            f"<tr><td class=\"label\">{category}</td>"
            f"<td>{w:.4f}</td><td>{info['amp']:.4e}</td></tr>"
        )

        rollouts = {
            "true": steady_state_rollout(
                U_true[idx], w, n_periods=2.0, n_frames=120,
                free=free, n_total=int(data["N"]),
            ),
            "full": steady_state_rollout(
                U_full[idx], w, n_periods=2.0, n_frames=120,
                free=free, n_total=int(data["N"]),
            ),
            "rom": {
                "times": np.linspace(0.0, 2.0 * (2.0 * np.pi / max(w, 1e-12)), 120),
                "U_full": np.real(np.outer(
                    np.exp(1j * w * np.linspace(0.0, 2.0 * (2.0 * np.pi / max(w, 1e-12)), 120)),
                    U_rom_lifted[idx],
                )),
            },
        }
        rollouts["rom"]["V_full"] = np.zeros_like(rollouts["rom"]["U_full"])
        rollouts["rom"]["A_full"] = np.zeros_like(rollouts["rom"]["U_full"])
        rollouts["rom"]["u0_full"] = rollouts["rom"]["U_full"][0].copy()
        rollouts["rom"]["v0_full"] = np.zeros(rollouts["rom"]["U_full"].shape[1])

        amp_max = max(
            float(np.max(np.abs(r["U_full"]))) for r in rollouts.values()
        )
        amp_max = max(amp_max, 1e-12)
        vis_scale = 0.75 / amp_max

        side_quiver = make_side_by_side_quiver_animation(
            rollouts, n,
            title=f"Steady-state quiver, {category} response (omega = {w:.3f})",
            output_stem=animation_dir / f"{slug}_side_by_side_quiver",
            vis_scale=vis_scale,
        )
        side_heatmap = make_side_by_side_heatmap_animation(
            rollouts, n,
            title=f"Steady-state |u| heatmap, {category} response (omega = {w:.3f})",
            output_stem=animation_dir / f"{slug}_side_by_side_heatmap",
            vmax=amp_max,
        )

        single_quivers = {}
        single_heatmaps = {}
        for model_id in ("true", "full", "rom"):
            single_quivers[model_id] = base.make_quiver_animation(
                rollouts[model_id], n,
                title=f"{MODEL_LABELS[model_id]} ({category}, omega={w:.3f})",
                output_stem=animation_dir / f"{slug}_{model_id}_quiver",
                vis_scale=vis_scale,
            )
            single_heatmaps[model_id] = base.make_heatmap_animation(
                rollouts[model_id], n,
                title=f"{MODEL_LABELS[model_id]} ({category}, omega={w:.3f})",
                output_stem=animation_dir / f"{slug}_{model_id}_heatmap",
                vmax=amp_max,
            )

        block_html = "\n".join([
            "<div class=\"freq-block\">",
            f"<h3>{category.capitalize()} response &mdash; omega = {w:.4f}</h3>",
            "<h4>Side-by-side quiver</h4>",
            base.render_video_tag(out_path.parent, side_quiver, "All three models, quiver"),
            "<h4>Side-by-side heatmap</h4>",
            base.render_video_tag(out_path.parent, side_heatmap, "All three models, heatmap"),
            "<h4>Per-model quiver</h4>",
            "<div class=\"media-grid\">",
            base.render_video_tag(out_path.parent, single_quivers["true"], MODEL_LABELS["true"]),
            base.render_video_tag(out_path.parent, single_quivers["full"], MODEL_LABELS["full"]),
            base.render_video_tag(out_path.parent, single_quivers["rom"], MODEL_LABELS["rom"]),
            "</div>",
            "<h4>Per-model heatmap</h4>",
            "<div class=\"media-grid\">",
            base.render_video_tag(out_path.parent, single_heatmaps["true"], MODEL_LABELS["true"]),
            base.render_video_tag(out_path.parent, single_heatmaps["full"], MODEL_LABELS["full"]),
            base.render_video_tag(out_path.parent, single_heatmaps["rom"], MODEL_LABELS["rom"]),
            "</div>",
            "</div>",
        ])
        animation_blocks.append(block_html)

    if noise_std > 0.0:
        noise_description = (
            f"relative Gaussian noise with std <code>{noise_std:.3g}</code> "
            f"of each signal's RMS, applied independently to <code>U</code>, "
            f"<code>V</code>, and <code>A</code>; <code>A_target</code> recomputed "
            f"from the noisy state (seed <code>{noise_seed}</code>)."
        )
        title_suffix = f", noise={noise_std:.3g}"
    else:
        noise_description = "none (clean data)."
        title_suffix = ""
    html = HTML_TEMPLATE.format(
        report_title=f"Spring-Grid Frequency Response Report ({n}x{n}, STLS Full + ROM{title_suffix})",
        n=data["n"],
        n_free=n_free,
        block_size=block_size,
        n_rom=rom_view["n"],
        n_rom_dofs=n_rom_dofs,
        alpha=alpha,
        beta=beta,
        noise_description=noise_description,
        response_curves_b64=response_curves_b64,
        headline_curve_b64=headline_curve_b64,
        error_curves_b64=error_curves_b64,
        amplitude_map_b64=amplitude_map_b64,
        phase_map_b64=phase_map_b64,
        omega_peak=omega_peak,
        animation_table_rows="\n".join(table_rows),
        animation_blocks="\n".join(animation_blocks),
        out_path=out_path,
        figure_dir=figure_dir,
        animation_dir=animation_dir,
        n_omega=n_omega,
        omega_min=omega_lo,
        omega_max=omega_hi,
        static_residual=static_residual,
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
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--force-amplitude", type=float, default=1.0)
    parser.add_argument("--force-omega", type=float, default=2.0 * np.pi)
    parser.add_argument("--force-sigma-cols", type=float, default=1.5)
    parser.add_argument("--n-omega", type=int, default=400)
    parser.add_argument(
        "--noise-std", type=float, default=0.0,
        help="Relative Gaussian measurement noise on U/V/A (default 0 = off; e.g. 0.02 for 2%% RMS).",
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.out is None:
        if args.noise_std and args.noise_std > 0.0:
            slug = f"noise_{args.noise_std:.3g}".replace(".", "p").replace("-", "m")
            out_path = DEFAULT_OUT_PATH.with_name(
                DEFAULT_OUT_PATH.stem + f"_{slug}" + DEFAULT_OUT_PATH.suffix
            )
        else:
            out_path = DEFAULT_OUT_PATH
    else:
        out_path = args.out

    main(
        n=args.n,
        block_size=args.block_size,
        n_traj=args.n_traj,
        t_end=args.t_end,
        dt=args.dt,
        alpha=args.alpha,
        beta=args.beta,
        force_amplitude=args.force_amplitude,
        force_omega=args.force_omega,
        force_sigma_cols=args.force_sigma_cols,
        n_omega=args.n_omega,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
        out_path=out_path,
    )
