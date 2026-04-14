"""
EX01a: Linear 2D system — Classical STLS pipeline using PyTorch SINDy.

True system:
    dx/dt = -0.1*x + 2*y
    dy/dt = -2*x - 0.1*y
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from torchdiffeq import odeint
import sindy_torch


def main():
    device = sindy_torch.get_device()
    print(f"Device: {device}")

    # --- True system ---
    n = 2
    A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]], dtype=torch.float64, device=device)
    rhs = lambda t, x: x @ A.T  # dx/dt = A @ x, batched

    x0 = torch.tensor([2.0, 0.0], dtype=torch.float64, device=device)
    t = torch.arange(0, 25.001, 0.01, dtype=torch.float64, device=device)

    # --- Generate data ---
    with torch.no_grad():
        x = odeint(rhs, x0, t, rtol=1e-10, atol=1e-10)  # (n_times, 2)
    print(f"Data: {x.shape[0]} time steps, {n} variables")

    # --- Compute derivatives (true + noise) ---
    torch.manual_seed(42)
    eps = 0.05
    dx = (x @ A.T) + eps * torch.randn_like(x)

    # --- Build library and identify ---
    poly_order = 5
    library = sindy_torch.PolynomialLibrary(n, poly_order)
    Theta = library(x)
    print(f"Library: {Theta.shape[1]} candidate functions")

    Xi = sindy_torch.stls(Theta, dx, lam=0.05)

    # --- Display results ---
    model = sindy_torch.SINDyModule(library, library.n_features, n)
    model.set_xi(Xi)
    print(f"\nSparsity: {model.sparsity()*100:.0f}% zero coefficients")
    print("\nDiscovered system:")
    model.print_equations(["x", "y"])

    # --- Validate via ODE integration ---
    ode_model = sindy_torch.ODEModel(model, rtol=1e-10, atol=1e-10)
    with torch.no_grad():
        x_pred = ode_model(x0, t)  # (n_times, 2)

    err = torch.norm(x - x_pred) / torch.norm(x)
    print(f"\nRelative trajectory error: {err:.6f} ({err*100:.2f}%)")
    if err < 0.05:
        print("PASS: Model correctly identified.")
    else:
        print("WARN: Error higher than expected.")

    return model, err


if __name__ == "__main__":
    main()
