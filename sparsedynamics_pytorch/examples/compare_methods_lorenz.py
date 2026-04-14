"""
Compare SINDy and Neural ODE methods on a short Lorenz trajectory.

Outputs:
    figures/lorenz_method_comparison.png
    figures/lorenz_error_comparison.png
    figures/lorenz_loss_epoch.png
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchdiffeq import odeint

import sindy_torch
from example_plotting import figures_dir, save_loss_plot


def relative_error(x_true, x_pred):
    return (torch.norm(x_true - x_pred) / torch.norm(x_true)).item()


def as_numpy(x):
    return x.detach().cpu().numpy()


def main():
    torch.manual_seed(11)
    device = sindy_torch.get_device()
    dtype = torch.float64
    out_dir = figures_dir()
    out_dir.mkdir(exist_ok=True)
    print(f"Device: {device}")

    # Lorenz trajectories diverge quickly, so use a short comparison horizon.
    sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0
    x0 = torch.tensor([-8.0, 8.0, 27.0], dtype=dtype, device=device)
    t = torch.linspace(0.0, 1.0, 201, dtype=dtype, device=device)
    lorenz_rhs = lambda t_i, y: sindy_torch.lorenz(t_i, y, sigma, beta, rho)

    with torch.no_grad():
        x_true = odeint(lorenz_rhs, x0, t, rtol=1e-10, atol=1e-10)
    dx_true = torch.stack([lorenz_rhs(t[i], x_true[i]) for i in range(len(t))])

    # Method 1: classical SINDy via STLS.
    library = sindy_torch.PolynomialLibrary(n_vars=3, poly_order=3)
    theta = library(x_true)
    xi_stls = sindy_torch.stls(theta, dx_true, lam=0.025)
    sindy_stls = sindy_torch.SINDyModule(library, library.n_features, n_states=3).to(device)
    sindy_stls.set_xi(xi_stls)

    # Method 2: fine-tune a slightly perturbed SINDy model with Adam.
    sindy_adam = sindy_torch.SINDyModule(library, library.n_features, n_states=3).to(device)
    active_mask = (xi_stls.abs() > 1e-8).to(xi_stls.dtype)
    xi_init = xi_stls + 0.2 * active_mask * torch.randn_like(xi_stls)
    sindy_adam.set_xi(xi_init)
    sindy_optimizer = sindy_torch.SparseOptimizer(
        sindy_adam.xi,
        l1_lambda=1e-4,
        optimizer_kwargs={"lr": 5e-3},
    )
    sindy_losses = []
    print("\n--- SINDy Adam fine-tuning (derivative matching) ---")
    for epoch in range(500):
        losses = sindy_optimizer.step_derivative_matching(theta, dx_true)
        sindy_losses.append(losses["mse"])
        if (epoch + 1) % 100 == 0:
            print(
                f"  Epoch {epoch + 1:4d}: "
                f"loss={losses['total']:.4e}, mse={losses['mse']:.4e}"
            )

    # Method 3: Neural ODE with Adam on derivative matching.
    neural_ode = sindy_torch.NeuralODEModule(
        n_states=3,
        hidden_width=64,
        hidden_depth=2,
        dtype=dtype,
        device=device,
    )
    neural_optimizer = sindy_torch.GradientOptimizer(
        neural_ode,
        optimizer_kwargs={"lr": 5e-3},
    )
    neural_losses = []
    print("\n--- Neural ODE training (derivative matching) ---")
    for epoch in range(2000):
        losses = neural_optimizer.step_derivative_matching(neural_ode, x_true, dx_true)
        neural_losses.append(losses["mse"])
        if (epoch + 1) % 500 == 0:
            print(
                f"  Epoch {epoch + 1:4d}: "
                f"loss={losses['total']:.4e}, mse={losses['mse']:.4e}"
            )

    models = {
        "SINDy STLS": sindy_stls,
        "SINDy Adam fine-tune": sindy_adam,
        "Neural ODE": neural_ode,
    }

    trajectories = {"True": x_true}
    errors = {}
    derivative_mse = {}
    t0 = torch.tensor(0.0, dtype=dtype, device=device)
    for name, model in models.items():
        with torch.no_grad():
            x_pred = sindy_torch.ODEModel(model, rtol=1e-6, atol=1e-8)(x0, t)
            dx_pred = model(t0, x_true)
        trajectories[name] = x_pred
        errors[name] = relative_error(x_true, x_pred)
        derivative_mse[name] = F.mse_loss(dx_pred, dx_true).item()

    t_np = as_numpy(t)
    colors = {
        "True": "black",
        "SINDy STLS": "tab:blue",
        "SINDy Adam fine-tune": "tab:green",
        "Neural ODE": "tab:red",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for name, x in trajectories.items():
        x_np = as_numpy(x)
        axes[0].plot(x_np[:, 0], x_np[:, 1], label=name, color=colors[name], linewidth=2)
        axes[1].plot(t_np, x_np[:, 0], label=name, color=colors[name], linewidth=2)
        axes[2].plot(t_np, x_np[:, 1], label=name, color=colors[name], linewidth=2)
        axes[3].plot(t_np, x_np[:, 2], label=name, color=colors[name], linewidth=2)

    axes[0].set_title("x-y phase portrait")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[1].set_title("x(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    axes[2].set_title("y(t)")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("y")
    axes[3].set_title("z(t)")
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("z")
    axes[0].legend()

    comparison_path = out_dir / "lorenz_method_comparison.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    method_names = list(errors.keys())
    axes[0].bar(method_names, [errors[name] for name in method_names], color="tab:blue")
    axes[0].set_title("Relative trajectory error")
    axes[0].set_ylabel("||x_true - x_pred|| / ||x_true||")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(method_names, [derivative_mse[name] for name in method_names], color="tab:green")
    axes[1].set_title("Derivative MSE")
    axes[1].set_ylabel("MSE(dx/dt)")
    axes[1].tick_params(axis="x", rotation=20)

    error_path = out_dir / "lorenz_error_comparison.png"
    fig.savefig(error_path, dpi=180)
    plt.close(fig)

    loss_path = save_loss_plot(
        {
            "SINDy Adam fine-tune": sindy_losses,
            "Neural ODE": neural_losses,
        },
        out_dir / "lorenz_loss_epoch.png",
        "Lorenz derivative-matching loss",
    )

    print("\nSaved plots:")
    print(f"  {comparison_path}")
    print(f"  {error_path}")
    print(f"  {loss_path}")
    print("\nMetrics:")
    for name in method_names:
        print(
            f"  {name:22s}  "
            f"trajectory error={errors[name]:.4e}  "
            f"derivative mse={derivative_mse[name]:.4e}"
        )

    return comparison_path, error_path, loss_path, errors, derivative_mse


if __name__ == "__main__":
    main()
