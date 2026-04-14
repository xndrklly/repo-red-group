"""
EX02: Lorenz attractor — Classical STLS pipeline using PyTorch SINDy.

True system:
    dx/dt = sigma*(y - x)         = -10*x + 10*y
    dy/dt = x*(rho - z) - y       = 28*x - y - x*z
    dz/dt = x*y - beta*z          = x*y - (8/3)*z
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torchdiffeq import odeint
import sindy_torch


def main():
    device = sindy_torch.get_device()
    print(f"Device: {device}")

    # --- Parameters ---
    sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0
    n = 3
    poly_order = 5

    x0 = torch.tensor([-8.0, 8.0, 27.0], dtype=torch.float64, device=device)
    t = torch.arange(0.001, 50.001, 0.001, dtype=torch.float64, device=device)

    # --- Generate data ---
    lorenz_rhs = lambda t, y: sindy_torch.lorenz(t, y, sigma, beta, rho)
    with torch.no_grad():
        x = odeint(lorenz_rhs, x0, t, rtol=1e-12, atol=1e-12)
    print(f"Data: {x.shape[0]} time steps, {n} variables")

    # --- Compute noiseless derivatives ---
    dx = torch.stack([lorenz_rhs(t[i], x[i]) for i in range(len(t))])

    # --- Build library and identify ---
    library = sindy_torch.PolynomialLibrary(n, poly_order)
    Theta = library(x)
    print(f"Library: {Theta.shape[1]} candidate functions")

    Xi = sindy_torch.stls(Theta, dx, lam=0.025)

    # --- Display ---
    model = sindy_torch.SINDyModule(library, library.n_features, n)
    model.set_xi(Xi)
    print(f"\nSparsity: {model.sparsity()*100:.0f}% zero coefficients")
    print("\nDiscovered Lorenz system:")
    model.print_equations(["x", "y", "z"])

    # --- Check coefficients ---
    xi_np = Xi.numpy()
    checks = [
        ("sigma (x in dx/dt)", xi_np[1, 0], -sigma),
        ("sigma (y in dx/dt)", xi_np[2, 0], sigma),
        ("rho   (x in dy/dt)", xi_np[1, 1], rho),
        ("-1    (y in dy/dt)", xi_np[2, 1], -1.0),
        ("-1    (xz in dy/dt)", xi_np[6, 1], -1.0),
        ("beta  (z in dz/dt)", xi_np[3, 2], -beta),
        ("1     (xy in dz/dt)", xi_np[5, 2], 1.0),
    ]
    print("\n--- Coefficient verification ---")
    all_pass = True
    for name, found, expected in checks:
        err = abs(found - expected)
        ok = err < 0.01
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}: found={found:+.6f}, expected={expected:+.6f}, err={err:.2e}")

    if all_pass:
        print("\nAll Lorenz coefficients correctly identified.")
    return model


if __name__ == "__main__":
    main()
