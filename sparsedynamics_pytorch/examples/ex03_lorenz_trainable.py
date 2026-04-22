"""
EX03: Lorenz attractor — End-to-end gradient-based trajectory matching.

Demonstrates the key advantage of the PyTorch SINDy implementation:
Xi is trained by backpropagating through torchdiffeq's ODE solver,
minimizing trajectory prediction error directly.

Pipeline:
    x0 -> odeint(SINDyModule, x0, t) -> x_pred -> MSE(x_pred, x_true) + L1(Xi)
    -> backprop -> update Xi
"""

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torchdiffeq import odeint
import sindy_torch
from example_plotting import figures_dir, save_loss_plot


def main(device_arg: str = "auto"):
    device = sindy_torch.get_device(device_arg)
    print(f"Device: {device}")

    # --- Generate true Lorenz trajectory ---
    sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0
    n = 3

    x0 = torch.tensor([-8.0, 8.0, 27.0], dtype=torch.float64, device=device)
    # Short time window for CPU-friendly trajectory matching.
    t_train = torch.linspace(0, 1.0, 51, dtype=torch.float64, device=device)

    lorenz_rhs = lambda t, y: sindy_torch.lorenz(t, y, sigma, beta, rho)
    with torch.no_grad():
        x_true = odeint(lorenz_rhs, x0, t_train, rtol=1e-8, atol=1e-10)
    print(f"Training data: {x_true.shape}")

    # --- Initialize model ---
    poly_order = 3  # smaller library for faster training
    library = sindy_torch.PolynomialLibrary(n, poly_order)
    model = sindy_torch.SINDyModule(
        library, library.n_features, n,
    ).to(device)

    # Warm-start with STLS on derivative data
    dx_true = torch.stack([lorenz_rhs(t_train[i], x_true[i]) for i in range(len(t_train))])
    Theta = library(x_true)
    xi_stls = sindy_torch.stls(Theta, dx_true, lam=0.025)
    # Perturb active STLS terms to simulate starting from an imperfect estimate.
    torch.manual_seed(0)
    active_mask = (xi_stls.abs() > 1e-8).to(xi_stls.dtype)
    xi_init = xi_stls + 0.1 * active_mask * torch.randn_like(xi_stls)
    model.set_xi(xi_init)

    print("\nInitial (perturbed) system:")
    model.print_equations(["x", "y", "z"])

    # --- Set up gradient-based training ---
    ode_model = sindy_torch.ODEModel(model, rtol=1e-4, atol=1e-6)
    optimizer = sindy_torch.SparseOptimizer(
        model.xi,
        l1_lambda=1e-4,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
    )

    # --- Training loop ---
    print("\n--- Training (trajectory matching) ---")
    n_epochs = 50
    loss_history = []
    for epoch in range(n_epochs):
        try:
            losses = optimizer.step_trajectory_matching(
                ode_model, x0, t_train, x_true
            )
        except Exception as e:
            print(f"  Epoch {epoch}: ODE solver failed ({e}), reducing lr")
            for pg in optimizer.optimizer.param_groups:
                pg["lr"] *= 0.5
            continue

        loss_history.append(losses["mse"])
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:4d}: loss={losses['total']:.4e}, "
                  f"mse={losses['mse']:.4e}, l1={losses['l1']:.4e}")

        # Periodic hard thresholding
        if (epoch + 1) % 50 == 0:
            optimizer.threshold(tol=0.1)

    # Final hard threshold
    optimizer.threshold(tol=0.05)

    # --- Results ---
    print("\nDiscovered system (after training):")
    model.print_equations(["x", "y", "z"])
    print(f"\nSparsity: {model.sparsity()*100:.0f}% zero coefficients")

    # --- Verify ---
    xi_final = sindy_torch.as_numpy(model.xi)
    print("\n--- Key coefficients ---")
    labels = library.get_labels(["x", "y", "z"])
    for i in range(xi_final.shape[0]):
        for j in range(xi_final.shape[1]):
            if abs(xi_final[i, j]) > 0.05:
                print(f"  {labels[i]:>8s} -> d{'xyz'[j]}/dt: {xi_final[i,j]:+.4f}")

    loss_path = save_loss_plot(
        {"SINDy trajectory matching": loss_history},
        figures_dir() / "lorenz_trainable_loss_epoch.png",
        "Lorenz trajectory-matching loss",
        ylabel="Trajectory MSE loss",
    )
    print(f"\nSaved loss plot: {loss_path}")

    return model, loss_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sindy_torch.add_device_arg(parser)
    args = parser.parse_args()
    main(args.device)
