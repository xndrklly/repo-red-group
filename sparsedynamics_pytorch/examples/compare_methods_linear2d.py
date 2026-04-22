"""
Compare SINDy and Neural ODE methods on a small Linear 2D system.

Outputs:
    figures/linear2d_method_comparison.png
    figures/linear2d_error_comparison.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchdiffeq import odeint

import sindy_torch
from example_plotting import figures_dir, save_loss_plot


def relative_error(x_true, x_pred):
    return (torch.norm(x_true - x_pred) / torch.norm(x_true)).item()


def main(device_arg: str = "auto"):
    torch.manual_seed(7)
    device = sindy_torch.get_device(device_arg)
    dtype = torch.float64
    out_dir = figures_dir()
    out_dir.mkdir(exist_ok=True)
    print(f"Device: {device}")

    # True system: dx/dt = A x
    A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]], dtype=dtype, device=device)
    rhs = lambda t, x: x @ A.T
    x0 = torch.tensor([2.0, 0.0], dtype=dtype, device=device)
    t = torch.linspace(0.0, 5.0, 251, dtype=dtype, device=device)

    with torch.no_grad():
        x_true = odeint(rhs, x0, t, rtol=1e-10, atol=1e-10)
    dx_true = x_true @ A.T

    # Method 1: classical SINDy via STLS.
    library = sindy_torch.PolynomialLibrary(n_vars=2, poly_order=2)
    theta = library(x_true)
    xi_stls = sindy_torch.stls(theta, dx_true, lam=0.05)
    sindy_stls = sindy_torch.SINDyModule(library, library.n_features, n_states=2).to(device)
    sindy_stls.set_xi(xi_stls)

    # Method 2: train the SINDy coefficients with Adam on derivative matching.
    sindy_adam = sindy_torch.SINDyModule(library, library.n_features, n_states=2).to(device)
    sindy_adam_optimizer = sindy_torch.SparseOptimizer(
        sindy_adam.xi,
        l1_lambda=0.0,
        optimizer_kwargs={"lr": 5e-2},
    )
    sindy_losses = []
    for _ in range(500):
        losses = sindy_adam_optimizer.step_derivative_matching(theta, dx_true)
        sindy_losses.append(losses["mse"])

    # Method 3: Neural ODE with Adam on derivative matching.
    neural_ode = sindy_torch.NeuralODEModule(
        n_states=2,
        hidden_width=16,
        hidden_depth=1,
        dtype=dtype,
        device=device,
    )
    neural_optimizer = sindy_torch.GradientOptimizer(
        neural_ode,
        optimizer_kwargs={"lr": 1e-2},
    )
    neural_losses = []
    for _ in range(500):
        losses = neural_optimizer.step_derivative_matching(neural_ode, x_true, dx_true)
        neural_losses.append(losses["mse"])

    models = {
        "SINDy STLS": sindy_stls,
        "SINDy Adam": sindy_adam,
        "Neural ODE": neural_ode,
    }

    trajectories = {"True": x_true}
    errors = {}
    derivative_mse = {}
    for name, model in models.items():
        with torch.no_grad():
            x_pred = sindy_torch.ODEModel(model, rtol=1e-6, atol=1e-8)(x0, t)
            dx_pred = model(torch.tensor(0.0, dtype=dtype, device=device), x_true)
        trajectories[name] = x_pred
        errors[name] = relative_error(x_true, x_pred)
        derivative_mse[name] = F.mse_loss(dx_pred, dx_true).item()

    t_np = sindy_torch.as_numpy(t)
    colors = {
        "True": "black",
        "SINDy STLS": "tab:blue",
        "SINDy Adam": "tab:green",
        "Neural ODE": "tab:red",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for name, x in trajectories.items():
        x_np = sindy_torch.as_numpy(x)
        axes[0].plot(x_np[:, 0], x_np[:, 1], label=name, color=colors[name], linewidth=2)
        axes[1].plot(t_np, x_np[:, 0], label=name, color=colors[name], linewidth=2)
        axes[2].plot(t_np, x_np[:, 1], label=name, color=colors[name], linewidth=2)

    axes[0].set_title("Phase portrait")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[1].set_title("x(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    axes[2].set_title("y(t)")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("y")
    axes[0].legend()

    comparison_path = out_dir / "linear2d_method_comparison.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    method_names = list(errors.keys())
    axes[0].bar(method_names, [errors[name] for name in method_names], color="tab:blue")
    axes[0].set_title("Relative trajectory error")
    axes[0].set_ylabel("||x_true - x_pred|| / ||x_true||")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(method_names, [derivative_mse[name] for name in method_names], color="tab:green")
    axes[1].set_title("Derivative MSE")
    axes[1].set_ylabel("MSE(dx/dt)")
    axes[1].tick_params(axis="x", rotation=20)

    error_path = out_dir / "linear2d_error_comparison.png"
    fig.savefig(error_path, dpi=180)
    plt.close(fig)

    loss_path = save_loss_plot(
        {
            "SINDy Adam": sindy_losses,
            "Neural ODE": neural_losses,
        },
        out_dir / "linear2d_loss_epoch.png",
        "Linear 2D derivative-matching loss",
    )

    print("Saved plots:")
    print(f"  {comparison_path}")
    print(f"  {error_path}")
    print(f"  {loss_path}")
    print("\nMetrics:")
    for name in method_names:
        print(
            f"  {name:10s}  "
            f"trajectory error={errors[name]:.4e}  "
            f"derivative mse={derivative_mse[name]:.4e}"
        )

    return comparison_path, error_path, loss_path, errors, derivative_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sindy_torch.add_device_arg(parser)
    args = parser.parse_args()
    main(args.device)
