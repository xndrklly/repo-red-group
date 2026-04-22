"""
CUDA optional smoke tests.

These checks verify CPU fallback behavior everywhere and run small CUDA-only
smoke tests when a CUDA-enabled PyTorch build is available.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from torchdiffeq import odeint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sindy_torch
from sindy_torch.solvers.trajectory_gradients import manual_trajectory_loss_and_gradients


PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name}  {detail}")


def skip(name, detail="CUDA unavailable"):
    global SKIP_COUNT
    SKIP_COUNT += 1
    print(f"  SKIP: {name}  {detail}")


def test_device_selection():
    print("\n=== Test 1: Device selection ===")
    auto_device = sindy_torch.get_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    check("auto device", auto_device.type == expected, f"got {auto_device}")
    check("cpu device", sindy_torch.get_device("cpu").type == "cpu")

    if torch.cuda.is_available():
        check("cuda device", sindy_torch.get_device("cuda").type == "cuda")
    else:
        try:
            sindy_torch.get_device("cuda")
        except RuntimeError as exc:
            check("cuda unavailable error", "CUDA was requested" in str(exc), str(exc))
        else:
            check("cuda unavailable error", False, "RuntimeError not raised")


def make_cuda_linear_data():
    device = sindy_torch.get_device("cuda")
    dtype = torch.float64
    A = torch.tensor([[-0.2, 1.0], [-1.0, -0.3]], dtype=dtype, device=device)
    rhs = lambda t, x: x @ A.T
    x0 = torch.tensor([1.0, -0.5], dtype=dtype, device=device)
    t = torch.linspace(0.0, 0.25, 9, dtype=dtype, device=device)
    with torch.no_grad():
        x_true = odeint(rhs, x0, t, rtol=1e-8, atol=1e-10)
    dx_true = x_true @ A.T
    return x0, t, x_true, dx_true


def test_cuda_sindy_solve():
    print("\n=== Test 2: CUDA SINDy ODE solve ===")
    if not torch.cuda.is_available():
        skip("CUDA SINDy ODE solve")
        return

    x0, t, x_true, dx_true = make_cuda_linear_data()
    library = sindy_torch.PolynomialLibrary(n_vars=2, poly_order=1)
    theta = library(x_true)
    xi = sindy_torch.stls(theta, dx_true, lam=0.0)
    model = sindy_torch.SINDyModule(library, library.n_features, n_states=2).to(x0.device)
    model.set_xi(xi)
    with torch.no_grad():
        x_pred = sindy_torch.ODEModel(model, rtol=1e-7, atol=1e-9)(x0, t)
    check("SINDy trajectory on CUDA", x_pred.device.type == "cuda")
    check("SINDy finite trajectory", torch.isfinite(x_pred).all().item())


def test_cuda_neural_optimizer_step():
    print("\n=== Test 3: CUDA Neural ODE optimizer step ===")
    if not torch.cuda.is_available():
        skip("CUDA Neural ODE optimizer step")
        return

    x0, t, x_true, dx_true = make_cuda_linear_data()
    model = sindy_torch.NeuralODEModule(
        n_states=2,
        hidden_width=8,
        hidden_depth=1,
        dtype=torch.float64,
        device=x0.device,
    )
    optimizer = sindy_torch.GradientOptimizer(model, optimizer_kwargs={"lr": 1e-2})
    losses = optimizer.step_derivative_matching(model, x_true, dx_true)
    check("Neural optimizer finite loss", torch.isfinite(torch.tensor(losses["mse"])).item())


def test_cuda_manual_gradients():
    print("\n=== Test 4: CUDA explicit sensitivity/adjoint gradients ===")
    if not torch.cuda.is_available():
        skip("CUDA explicit sensitivity/adjoint gradients")
        return

    x0, t, x_true, _ = make_cuda_linear_data()
    library = sindy_torch.PolynomialLibrary(n_vars=2, poly_order=1)
    model = sindy_torch.SINDyModule(library, library.n_features, n_states=2).to(x0.device)
    xi = torch.zeros(library.n_features, 2, dtype=torch.float64, device=x0.device)
    xi[1, 0] = -0.18
    xi[2, 0] = 0.88
    xi[1, 1] = -0.92
    xi[2, 1] = -0.22
    model.set_xi(xi)
    ode_model = sindy_torch.ODEModel(model, rtol=1e-7, atol=1e-9)

    for method in ("sensitivity", "adjoint"):
        loss, flat_grad = manual_trajectory_loss_and_gradients(
            ode_model,
            x0,
            t,
            x_true,
            [model.xi],
            method,
        )
        check(f"{method} loss on CUDA", loss.device.type == "cuda")
        check(f"{method} gradient on CUDA", flat_grad.device.type == "cuda")
        check(f"{method} finite gradient", torch.isfinite(flat_grad).all().item())
        check(f"{method} finite loss", torch.isfinite(loss).item())


def main():
    test_device_selection()
    test_cuda_sindy_solve()
    test_cuda_neural_optimizer_step()
    test_cuda_manual_gradients()

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed, {SKIP_COUNT} skipped")
    if FAIL_COUNT == 0:
        print("CUDA optional verifications PASSED.")
    else:
        print("Some CUDA optional verifications FAILED; see above.")
    print(f"{'=' * 50}")
    return FAIL_COUNT == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
