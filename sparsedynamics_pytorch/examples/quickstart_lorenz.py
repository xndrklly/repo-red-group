"""
Quick start: identify the Lorenz equations with PyTorch SINDy.

Run after installing the package from sparsedynamics_pytorch:
    python -m pip install -e .
    python examples/quickstart_lorenz.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torchdiffeq import odeint

import sindy_torch


def main(device_arg: str = "auto"):
    device = sindy_torch.get_device(device_arg)
    print(f"Device: {device}")

    sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0
    n_states = 3
    poly_order = 3

    x0 = torch.tensor([-8.0, 8.0, 27.0], dtype=torch.float64, device=device)
    t = torch.linspace(0.001, 5.0, 5000, dtype=torch.float64, device=device)

    lorenz_rhs = lambda t_i, y: sindy_torch.lorenz(t_i, y, sigma, beta, rho)

    with torch.no_grad():
        x = odeint(lorenz_rhs, x0, t, rtol=1e-10, atol=1e-10)

    dx = torch.stack([lorenz_rhs(t[i], x[i]) for i in range(len(t))])

    library = sindy_torch.PolynomialLibrary(n_vars=n_states, poly_order=poly_order)
    theta = library(x)
    xi = sindy_torch.stls(theta, dx, lam=0.025)

    model = sindy_torch.SINDyModule(library, library.n_features, n_states).to(device)
    model.set_xi(xi)

    print("Discovered Lorenz system:")
    model.print_equations(["x", "y", "z"])

    checks = {
        "x -> dx/dt": (xi[1, 0], -sigma),
        "y -> dx/dt": (xi[2, 0], sigma),
        "x -> dy/dt": (xi[1, 1], rho),
        "y -> dy/dt": (xi[2, 1], -1.0),
        "xz -> dy/dt": (xi[6, 1], -1.0),
        "z -> dz/dt": (xi[3, 2], -beta),
        "xy -> dz/dt": (xi[5, 2], 1.0),
    }
    for name, (found, expected) in checks.items():
        if abs(found - expected) >= 0.01:
            raise AssertionError(
                f"{name}: found {found.item():+.6f}, expected {expected:+.6f}"
            )

    print("\nAll 7 Lorenz coefficients matched the expected values.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sindy_torch.add_device_arg(parser)
    args = parser.parse_args()
    main(args.device)
