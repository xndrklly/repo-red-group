"""
EX04: Linear 2D system - Neural ODE derivative matching.

This example uses the model-agnostic GradientOptimizer with Adam to train a
NeuralODEModule on derivative data, then integrates the learned dynamics.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torchdiffeq import odeint
import sindy_torch
from example_plotting import figures_dir, save_loss_plot


def main():
    device = sindy_torch.get_device()
    dtype = torch.float64
    torch.manual_seed(7)
    print(f"Device: {device}")

    A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]], dtype=dtype, device=device)
    rhs = lambda t, x: x @ A.T

    x0 = torch.tensor([2.0, 0.0], dtype=dtype, device=device)
    t = torch.linspace(0.0, 5.0, 251, dtype=dtype, device=device)

    with torch.no_grad():
        x_true = odeint(rhs, x0, t, rtol=1e-10, atol=1e-10)
    dx_true = x_true @ A.T

    model = sindy_torch.NeuralODEModule(
        n_states=2,
        hidden_width=16,
        hidden_depth=1,
        dtype=dtype,
        device=device,
    )
    optimizer = sindy_torch.GradientOptimizer(
        model,
        optimizer_kwargs={"lr": 1e-2},
    )

    with torch.no_grad():
        initial_mse = torch.nn.functional.mse_loss(model(None, x_true), dx_true).item()

    loss_history = []
    for epoch in range(300):
        losses = optimizer.step_derivative_matching(model, x_true, dx_true)
        loss_history.append(losses["mse"])
        if (epoch + 1) % 100 == 0:
            print(
                f"  Epoch {epoch + 1:4d}: "
                f"loss={losses['total']:.4e}, mse={losses['mse']:.4e}"
            )

    with torch.no_grad():
        final_mse = torch.nn.functional.mse_loss(model(None, x_true), dx_true).item()
        ode_model = sindy_torch.ODEModel(model, rtol=1e-6, atol=1e-8)
        x_pred = ode_model(x0, t)
        rel_err = torch.norm(x_true - x_pred) / torch.norm(x_true)

    print(f"\nDerivative MSE: {initial_mse:.4e} -> {final_mse:.4e}")
    print(f"Relative trajectory error: {rel_err:.4f} ({rel_err * 100:.2f}%)")
    loss_path = save_loss_plot(
        {"Neural ODE": loss_history},
        figures_dir() / "neural_ode_linear2d_loss_epoch.png",
        "Neural ODE Linear 2D derivative-matching loss",
    )
    print(f"Saved loss plot: {loss_path}")

    return model, final_mse, rel_err, loss_path


if __name__ == "__main__":
    main()
