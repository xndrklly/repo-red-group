"""
Long parallel Lorenz benchmark for SINDy and Neural ODE training methods.

The script trains each method in a separate CPU worker process, then writes
early-horizon trajectory comparisons, loss curves, and long-horizon 3D
butterfly plots.

Outputs:
    figures/lorenz_method_summary.csv
    figures/lorenz_sindy_method_comparison.png
    figures/lorenz_neural_ode_method_comparison.png
    figures/lorenz_sindy_loss_epoch.png
    figures/lorenz_neural_ode_loss_epoch.png
    figures/lorenz_sindy_butterfly_3d_grid.png
    figures/lorenz_neural_ode_butterfly_3d_grid.png
    figures/lorenz_*_butterfly_3d.png
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchdiffeq import odeint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sindy_torch
from example_plotting import figures_dir, save_loss_plot


SEED = 11
DTYPE = torch.float64
MAX_WORKERS = 4
WORKER_TORCH_THREADS = 1

EARLY_T_END = 1.0
EARLY_N_TIMES = 201
LONG_T_END = 30.0
LONG_N_TIMES = 6001

SINDY_DERIVATIVE_EPOCHS = 3000
SINDY_TRAJECTORY_EPOCHS = 500
NEURAL_DERIVATIVE_EPOCHS = 8000
NEURAL_TRAJECTORY_EPOCHS = 600

LORENZ_SIGMA = 10.0
LORENZ_BETA = 8.0 / 3.0
LORENZ_RHO = 28.0
LORENZ_X0 = [-8.0, 8.0, 27.0]

METHOD_COLORS = {
    "True": "black",
    "STLS": "tab:blue",
    "Adam derivative": "tab:green",
    "Adam trajectory": "tab:orange",
    "Sensitivity trajectory": "tab:red",
    "Adjoint trajectory": "tab:purple",
}

_THREADS_CONFIGURED = False


@dataclass(frozen=True)
class MethodSpec:
    family: str
    method: str
    training_mode: str
    gradient_method: str | None
    epochs: int
    lr: float
    l1_lambda: float = 0.0


METHOD_SPECS = [
    MethodSpec("SINDy", "STLS", "stls", None, 0, 0.0),
    MethodSpec("SINDy", "Adam derivative", "derivative", None, SINDY_DERIVATIVE_EPOCHS, 5e-3, 1e-4),
    MethodSpec("SINDy", "Adam trajectory", "trajectory", "autograd", SINDY_TRAJECTORY_EPOCHS, 1e-3, 1e-4),
    MethodSpec("SINDy", "Sensitivity trajectory", "trajectory", "sensitivity", SINDY_TRAJECTORY_EPOCHS, 1e-3, 1e-4),
    MethodSpec("SINDy", "Adjoint trajectory", "trajectory", "adjoint", SINDY_TRAJECTORY_EPOCHS, 1e-3, 1e-4),
    MethodSpec("Neural ODE", "Adam derivative", "derivative", None, NEURAL_DERIVATIVE_EPOCHS, 5e-3),
    MethodSpec("Neural ODE", "Adam trajectory", "trajectory", "autograd", NEURAL_TRAJECTORY_EPOCHS, 1e-3),
    MethodSpec("Neural ODE", "Sensitivity trajectory", "trajectory", "sensitivity", NEURAL_TRAJECTORY_EPOCHS, 1e-3),
    MethodSpec("Neural ODE", "Adjoint trajectory", "trajectory", "adjoint", NEURAL_TRAJECTORY_EPOCHS, 1e-3),
]


def configure_worker_threads() -> None:
    """Limit PyTorch thread usage once per worker process."""
    global _THREADS_CONFIGURED
    if _THREADS_CONFIGURED:
        return
    torch.set_num_threads(WORKER_TORCH_THREADS)
    try:
        torch.set_num_interop_threads(WORKER_TORCH_THREADS)
    except RuntimeError:
        # set_num_interop_threads can only be called before parallel work starts.
        pass
    _THREADS_CONFIGURED = True


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def relative_error(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
    return (torch.norm(x_true - x_pred) / torch.norm(x_true)).item()


def as_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def lorenz_rhs(t_i: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return sindy_torch.lorenz(t_i, y, LORENZ_SIGMA, LORENZ_BETA, LORENZ_RHO)


def make_lorenz_grid(t_end: float, n_times: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cpu")
    x0 = torch.tensor(LORENZ_X0, dtype=DTYPE, device=device)
    t = torch.linspace(0.0, t_end, n_times, dtype=DTYPE, device=device)
    return x0, t


def make_lorenz_data(
    t_end: float,
    n_times: int,
    *,
    with_derivatives: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x0, t = make_lorenz_grid(t_end, n_times)
    with torch.no_grad():
        x_true = odeint(lorenz_rhs, x0, t, rtol=1e-10, atol=1e-10)
    dx_true = None
    if with_derivatives:
        dx_true = torch.stack([lorenz_rhs(t[i], x_true[i]) for i in range(len(t))])
    return x0, t, x_true, dx_true


def build_sindy_inputs(
    x_true: torch.Tensor,
    dx_true: torch.Tensor,
) -> tuple[sindy_torch.PolynomialLibrary, torch.Tensor, torch.Tensor, torch.Tensor]:
    library = sindy_torch.PolynomialLibrary(n_vars=3, poly_order=3)
    theta = library(x_true)
    xi_stls = sindy_torch.stls(theta, dx_true, lam=0.025)
    gen = torch.Generator(device=x_true.device)
    gen.manual_seed(101)
    active_mask = (xi_stls.abs() > 1e-8).to(xi_stls.dtype)
    xi_init = xi_stls + 0.2 * active_mask * torch.randn(
        xi_stls.shape,
        dtype=xi_stls.dtype,
        device=xi_stls.device,
        generator=gen,
    )
    return library, theta, xi_stls, xi_init


def build_sindy_model(
    library: sindy_torch.PolynomialLibrary,
    xi: torch.Tensor,
) -> sindy_torch.SINDyModule:
    model = sindy_torch.SINDyModule(library, library.n_features, n_states=3)
    model.set_xi(xi)
    return model


def build_neural_model() -> sindy_torch.NeuralODEModule:
    # Same seed for every Neural ODE method, so each method starts identically.
    torch.manual_seed(202)
    return sindy_torch.NeuralODEModule(
        n_states=3,
        hidden_width=64,
        hidden_depth=2,
        dtype=DTYPE,
        device=torch.device("cpu"),
    )


def evaluate_model(
    model: torch.nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    x_true: torch.Tensor,
    dx_true: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    with torch.no_grad():
        x_pred = sindy_torch.ODEModel(model, rtol=1e-6, atol=1e-8)(x0, t)
        dx_pred = model(torch.tensor(0.0, dtype=DTYPE), x_true)
    return x_pred, relative_error(x_true, x_pred), F.mse_loss(dx_pred, dx_true).item()


def evaluate_long_butterfly(
    model: torch.nn.Module,
    x0: torch.Tensor,
    t_long: torch.Tensor,
) -> tuple[torch.Tensor | None, str]:
    try:
        with torch.no_grad():
            x_long = sindy_torch.ODEModel(model, rtol=1e-6, atol=1e-8)(x0, t_long)
        if torch.isfinite(x_long).all():
            return x_long, "ok"
        return None, "nonfinite"
    except Exception as exc:
        return None, f"failed: {type(exc).__name__}"


def train_sindy(spec: MethodSpec) -> tuple[torch.nn.Module, list[float], float]:
    x0, t, x_true, dx_true = make_lorenz_data(
        EARLY_T_END,
        EARLY_N_TIMES,
        with_derivatives=True,
    )
    assert dx_true is not None
    library, theta, xi_stls, xi_init = build_sindy_inputs(x_true, dx_true)

    if spec.training_mode == "stls":
        return build_sindy_model(library, xi_stls), [], 0.0

    model = build_sindy_model(library, xi_init)
    losses: list[float] = []
    train_start = time.perf_counter()
    if spec.training_mode == "derivative":
        optimizer = sindy_torch.SparseOptimizer(
            model.xi,
            l1_lambda=spec.l1_lambda,
            optimizer_kwargs={"lr": spec.lr},
        )
        for _ in range(spec.epochs):
            loss_dict = optimizer.step_derivative_matching(theta, dx_true)
            losses.append(loss_dict["mse"])
    elif spec.training_mode == "trajectory":
        ode_model = sindy_torch.ODEModel(model, rtol=1e-4, atol=1e-6)
        optimizer = sindy_torch.SparseOptimizer(
            model.xi,
            l1_lambda=spec.l1_lambda,
            optimizer_kwargs={"lr": spec.lr},
        )
        for epoch in range(spec.epochs):
            loss_dict = optimizer.step_trajectory_matching(
                ode_model,
                x0,
                t,
                x_true,
                gradient_method=spec.gradient_method or "autograd",
            )
            losses.append(loss_dict["mse"])
            if (epoch + 1) % 100 == 0:
                optimizer.threshold(0.05)
        optimizer.threshold(0.05)
    else:
        raise ValueError(f"Unknown SINDy training mode {spec.training_mode!r}")
    return model, losses, time.perf_counter() - train_start


def train_neural(spec: MethodSpec) -> tuple[torch.nn.Module, list[float], float]:
    x0, t, x_true, dx_true = make_lorenz_data(
        EARLY_T_END,
        EARLY_N_TIMES,
        with_derivatives=True,
    )
    assert dx_true is not None
    model = build_neural_model()
    losses: list[float] = []
    train_start = time.perf_counter()

    if spec.training_mode == "derivative":
        optimizer = sindy_torch.GradientOptimizer(
            model,
            optimizer_kwargs={"lr": spec.lr},
        )
        for _ in range(spec.epochs):
            loss_dict = optimizer.step_derivative_matching(model, x_true, dx_true)
            losses.append(loss_dict["mse"])
    elif spec.training_mode == "trajectory":
        ode_model = sindy_torch.ODEModel(model, rtol=1e-4, atol=1e-6)
        optimizer = sindy_torch.GradientOptimizer(
            model,
            optimizer_kwargs={"lr": spec.lr},
        )
        for _ in range(spec.epochs):
            loss_dict = optimizer.step_trajectory_matching(
                ode_model,
                x0,
                t,
                x_true,
                gradient_method=spec.gradient_method or "autograd",
            )
            losses.append(loss_dict["mse"])
    else:
        raise ValueError(f"Unknown Neural ODE training mode {spec.training_mode!r}")
    return model, losses, time.perf_counter() - train_start


def run_method_worker(spec: MethodSpec) -> dict[str, Any]:
    configure_worker_threads()
    torch.manual_seed(SEED)
    start = time.perf_counter()
    row: dict[str, Any] = {
        "model_family": spec.family,
        "method": spec.method,
        "status": "ok",
        "runtime_seconds": math.nan,
        "training_seconds": math.nan,
        "final_training_loss": math.nan,
        "early_trajectory_relative_error": math.nan,
        "derivative_mse": math.nan,
        "long_butterfly_status": "not_run",
        "butterfly_plot_path": "",
    }
    result: dict[str, Any] = {
        "family": spec.family,
        "method": spec.method,
        "row": row,
        "loss_history": [],
        "early_trajectory": None,
        "long_trajectory": None,
        "traceback": "",
    }

    try:
        x0, t, x_true, dx_true = make_lorenz_data(
            EARLY_T_END,
            EARLY_N_TIMES,
            with_derivatives=True,
        )
        assert dx_true is not None
        x0_long, t_long = make_lorenz_grid(
            LONG_T_END,
            LONG_N_TIMES,
        )

        if spec.family == "SINDy":
            model, losses, training_seconds = train_sindy(spec)
        elif spec.family == "Neural ODE":
            model, losses, training_seconds = train_neural(spec)
        else:
            raise ValueError(f"Unknown model family {spec.family!r}")

        x_pred, traj_err, deriv_mse = evaluate_model(model, x0, t, x_true, dx_true)
        x_long, long_status = evaluate_long_butterfly(model, x0_long, t_long)

        row.update(
            {
                "training_seconds": training_seconds,
                "final_training_loss": losses[-1] if losses else deriv_mse,
                "early_trajectory_relative_error": traj_err,
                "derivative_mse": deriv_mse,
                "long_butterfly_status": long_status,
            }
        )
        result["loss_history"] = losses
        result["early_trajectory"] = as_numpy(x_pred)
        if x_long is not None:
            result["long_trajectory"] = as_numpy(x_long)
    except Exception:
        row["status"] = "failed"
        row["long_butterfly_status"] = "not_run"
        result["traceback"] = traceback.format_exc()
    finally:
        row["runtime_seconds"] = time.perf_counter() - start

    return result


def truth_trajectories() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, t_early, x_early, _ = make_lorenz_data(
        EARLY_T_END,
        EARLY_N_TIMES,
        with_derivatives=False,
    )
    _, t_long, x_long, _ = make_lorenz_data(
        LONG_T_END,
        LONG_N_TIMES,
        with_derivatives=False,
    )
    return t_early, x_early, t_long, x_long


def plot_family_comparison(
    family_name: str,
    t: torch.Tensor,
    trajectories: dict[str, Any],
    output_path: Path,
) -> Path:
    t_np = as_numpy(t)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    for name, x in trajectories.items():
        if x is None:
            continue
        x_np = as_numpy(x) if isinstance(x, torch.Tensor) else x
        color = METHOD_COLORS.get(name, None)
        linewidth = 2.4 if name == "True" else 1.8
        axes[0].plot(x_np[:, 0], x_np[:, 1], label=name, color=color, linewidth=linewidth)
        axes[1].plot(t_np, x_np[:, 0], label=name, color=color, linewidth=linewidth)
        axes[2].plot(t_np, x_np[:, 1], label=name, color=color, linewidth=linewidth)
        axes[3].plot(t_np, x_np[:, 2], label=name, color=color, linewidth=linewidth)

    axes[0].set_title(f"{family_name}: early x-y phase portrait")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[1].set_title("Early x(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    axes[2].set_title("Early y(t)")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("y")
    axes[3].set_title("Early z(t)")
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("z")
    axes[0].legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_butterfly_3d(
    family: str,
    method: str,
    true_long: torch.Tensor,
    method_long: Any,
    output_path: Path,
) -> Path:
    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    true_np = as_numpy(true_long)
    ax.plot(
        true_np[:, 0],
        true_np[:, 1],
        true_np[:, 2],
        color="0.72",
        linewidth=0.6,
        alpha=0.75,
        label="True Lorenz",
    )
    if method_long is not None:
        method_np = as_numpy(method_long) if isinstance(method_long, torch.Tensor) else method_long
        ax.plot(
            method_np[:, 0],
            method_np[:, 1],
            method_np[:, 2],
            color=METHOD_COLORS.get(method, "tab:blue"),
            linewidth=0.7,
            label=method,
        )
    ax.set_title(f"{family}: {method}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-55)
    ax.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_butterfly_grid(
    family: str,
    true_long: torch.Tensor,
    results: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    family_results = [result for result in results if result["family"] == family]
    n_cols = 3
    n_rows = math.ceil(len(family_results) / n_cols)
    fig = plt.figure(figsize=(5.2 * n_cols, 4.8 * n_rows), constrained_layout=True)
    true_np = as_numpy(true_long)

    for idx, result in enumerate(family_results, start=1):
        ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
        ax.plot(
            true_np[:, 0],
            true_np[:, 1],
            true_np[:, 2],
            color="0.75",
            linewidth=0.45,
            alpha=0.7,
        )
        method_long = result.get("long_trajectory")
        method = result["method"]
        if method_long is not None:
            ax.plot(
                method_long[:, 0],
                method_long[:, 1],
                method_long[:, 2],
                color=METHOD_COLORS.get(method, "tab:blue"),
                linewidth=0.55,
            )
        else:
            ax.text2D(0.08, 0.88, "No finite trajectory", transform=ax.transAxes)
        ax.set_title(method)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=24, azim=-55)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_family",
        "method",
        "status",
        "runtime_seconds",
        "training_seconds",
        "final_training_loss",
        "early_trajectory_relative_error",
        "derivative_mse",
        "long_butterfly_status",
        "butterfly_plot_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nSummary metrics:")
    print(
        f"{'Family':18s} {'Method':24s} {'Status':8s} {'Train loss':>12s} "
        f"{'Early err':>12s} {'dX MSE':>12s} {'Runtime':>10s}  Butterfly"
    )
    print("-" * 126)
    for row in rows:
        print(
            f"{row['model_family']:18s} {row['method']:24s} {row['status']:8s} "
            f"{row['final_training_loss']:12.4e} "
            f"{row['early_trajectory_relative_error']:12.4e} "
            f"{row['derivative_mse']:12.4e} "
            f"{row['runtime_seconds']:10.2f}  {row['long_butterfly_status']}"
        )


def sort_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {(spec.family, spec.method): idx for idx, spec in enumerate(METHOD_SPECS)}
    return sorted(results, key=lambda result: order[(result["family"], result["method"])])


def run_parallel_sweep() -> list[dict[str, Any]]:
    print(f"Running {len(METHOD_SPECS)} methods with max_workers={MAX_WORKERS}")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_spec = {
            executor.submit(run_method_worker, spec): spec
            for spec in METHOD_SPECS
        }
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                result = future.result()
            except Exception:
                result = {
                    "family": spec.family,
                    "method": spec.method,
                    "row": {
                        "model_family": spec.family,
                        "method": spec.method,
                        "status": "failed",
                        "runtime_seconds": math.nan,
                        "training_seconds": math.nan,
                        "final_training_loss": math.nan,
                        "early_trajectory_relative_error": math.nan,
                        "derivative_mse": math.nan,
                        "long_butterfly_status": "not_run",
                        "butterfly_plot_path": "",
                    },
                    "loss_history": [],
                    "early_trajectory": None,
                    "long_trajectory": None,
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            row = result["row"]
            print(
                f"Finished {spec.family} / {spec.method}: "
                f"{row['status']} in {row['runtime_seconds']:.1f}s"
            )
            if result.get("traceback"):
                print(result["traceback"])
    return sort_results(results)


def main():
    torch.manual_seed(SEED)
    out_dir = figures_dir()
    out_dir.mkdir(exist_ok=True)
    print("Device: cpu")
    print(
        "Long Lorenz benchmark: "
        f"early t=[0,{EARLY_T_END}], long t=[0,{LONG_T_END}], "
        f"{MAX_WORKERS} workers"
    )

    t_early, x_true_early, _, x_true_long = truth_trajectories()
    results = run_parallel_sweep()

    rows = []
    for result in results:
        family_slug = slugify(result["family"])
        method_slug = slugify(result["method"])
        butterfly_path = out_dir / f"lorenz_{family_slug}_{method_slug}_butterfly_3d.png"
        plot_butterfly_3d(
            result["family"],
            result["method"],
            x_true_long,
            result.get("long_trajectory"),
            butterfly_path,
        )
        result["row"]["butterfly_plot_path"] = str(butterfly_path)
        rows.append(result["row"])

    trajectories_by_family: dict[str, dict[str, Any]] = {
        "SINDy": {"True": x_true_early},
        "Neural ODE": {"True": x_true_early},
    }
    losses_by_family: dict[str, dict[str, list[float]]] = {
        "SINDy": {},
        "Neural ODE": {},
    }
    for result in results:
        if result.get("early_trajectory") is not None:
            trajectories_by_family[result["family"]][result["method"]] = result["early_trajectory"]
        if result.get("loss_history"):
            losses_by_family[result["family"]][result["method"]] = result["loss_history"]

    sindy_fig = plot_family_comparison(
        "SINDy",
        t_early,
        trajectories_by_family["SINDy"],
        out_dir / "lorenz_sindy_method_comparison.png",
    )
    neural_fig = plot_family_comparison(
        "Neural ODE",
        t_early,
        trajectories_by_family["Neural ODE"],
        out_dir / "lorenz_neural_ode_method_comparison.png",
    )
    sindy_loss_fig = save_loss_plot(
        losses_by_family["SINDy"],
        out_dir / "lorenz_sindy_loss_epoch.png",
        "Lorenz SINDy training loss",
        ylabel="Method objective MSE",
    )
    neural_loss_fig = save_loss_plot(
        losses_by_family["Neural ODE"],
        out_dir / "lorenz_neural_ode_loss_epoch.png",
        "Lorenz Neural ODE training loss",
        ylabel="Method objective MSE",
    )
    sindy_grid = plot_butterfly_grid(
        "SINDy",
        x_true_long,
        results,
        out_dir / "lorenz_sindy_butterfly_3d_grid.png",
    )
    neural_grid = plot_butterfly_grid(
        "Neural ODE",
        x_true_long,
        results,
        out_dir / "lorenz_neural_ode_butterfly_3d_grid.png",
    )
    summary_csv = write_summary_csv(rows, out_dir / "lorenz_method_summary.csv")

    print_summary(rows)
    print("\nSaved outputs:")
    print(f"  {summary_csv}")
    print(f"  {sindy_fig}")
    print(f"  {neural_fig}")
    print(f"  {sindy_loss_fig}")
    print(f"  {neural_loss_fig}")
    print(f"  {sindy_grid}")
    print(f"  {neural_grid}")
    print("  Individual butterfly plots: figures/lorenz_*_butterfly_3d.png")

    return {
        "rows": rows,
        "summary_csv": summary_csv,
        "sindy_figure": sindy_fig,
        "neural_figure": neural_fig,
        "sindy_loss_figure": sindy_loss_fig,
        "neural_loss_figure": neural_loss_fig,
        "sindy_butterfly_grid": sindy_grid,
        "neural_butterfly_grid": neural_grid,
    }


if __name__ == "__main__":
    main()
