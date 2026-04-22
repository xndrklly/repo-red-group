"""
Verification tests for NeuralODEModule, generic ODEModel, and GradientOptimizer.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torchdiffeq import odeint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sindy_torch


PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name}  {detail}")


def test_neural_ode_shapes(device):
    print("\n=== Test 1: NeuralODEModule shapes ===")
    model = sindy_torch.NeuralODEModule(3, hidden_width=8, hidden_depth=1, device=device)
    t0 = torch.tensor(0.0, dtype=torch.float64, device=device)
    x_single = torch.randn(3, dtype=torch.float64, device=device)
    x_batch = torch.randn(5, 3, dtype=torch.float64, device=device)
    x_grid = torch.randn(2, 5, 3, dtype=torch.float64, device=device)

    check("single state shape", model(t0, x_single).shape == x_single.shape)
    check("batch state shape", model(t0, x_batch).shape == x_batch.shape)
    check("leading dims state shape", model(t0, x_grid).shape == x_grid.shape)
    check("predict_derivative shape", model.predict_derivative(x_batch).shape == x_batch.shape)


def test_ode_model_backward_compatibility(device):
    print("\n=== Test 2: ODEModel with SINDyModule compatibility ===")
    library = sindy_torch.PolynomialLibrary(n_vars=2, poly_order=1)
    xi = torch.zeros(library.n_features, 2, dtype=torch.float64, device=device)
    xi[1, 0] = -0.1
    xi[2, 0] = 2.0
    xi[1, 1] = -2.0
    xi[2, 1] = -0.1
    sindy = sindy_torch.SINDyModule(library, library.n_features, n_states=2).to(device)
    sindy.set_xi(xi)

    ode_model = sindy_torch.ODEModel(sindy_module=sindy, rtol=1e-8, atol=1e-10)
    x0 = torch.tensor([2.0, 0.0], dtype=torch.float64, device=device)
    t = torch.linspace(0.0, 0.1, 5, dtype=torch.float64, device=device)
    x_pred = ode_model(x0, t)

    check("sindy_module alias", ode_model.sindy_module is sindy)
    check("dynamics_module alias", ode_model.dynamics_module is sindy)
    check("SINDy ODEModel shape", x_pred.shape == (len(t), 2))


def test_ode_model_neural_ode(device):
    print("\n=== Test 3: ODEModel with NeuralODEModule ===")
    neural = sindy_torch.NeuralODEModule(2, hidden_width=8, hidden_depth=1, device=device)
    ode_model = sindy_torch.ODEModel(neural, rtol=1e-6, atol=1e-8)
    x0 = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
    t = torch.linspace(0.0, 0.1, 5, dtype=torch.float64, device=device)
    x_pred = ode_model(x0, t)

    check("Neural ODEModel shape", x_pred.shape == (len(t), 2))


def test_gradient_optimizer_default(device):
    print("\n=== Test 4: GradientOptimizer defaults ===")
    model = sindy_torch.NeuralODEModule(2, hidden_width=8, hidden_depth=1, device=device)
    optimizer = sindy_torch.GradientOptimizer(model)
    check("default optimizer is Adam", isinstance(optimizer.optimizer, torch.optim.Adam))


def test_neural_ode_training_decreases_loss(device):
    print("\n=== Test 5: Neural ODE derivative training ===")
    torch.manual_seed(7)
    dtype = torch.float64
    A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]], dtype=dtype, device=device)
    rhs = lambda t, x: x @ A.T
    x0 = torch.tensor([2.0, 0.0], dtype=dtype, device=device)
    t = torch.linspace(0.0, 5.0, 251, dtype=dtype, device=device)

    with torch.no_grad():
        x = odeint(rhs, x0, t, rtol=1e-10, atol=1e-10)
    dx = x @ A.T

    model = sindy_torch.NeuralODEModule(
        2,
        hidden_width=16,
        hidden_depth=1,
        dtype=dtype,
        device=device,
    )
    optimizer = sindy_torch.GradientOptimizer(model, optimizer_kwargs={"lr": 1e-2})

    with torch.no_grad():
        initial_mse = F.mse_loss(model(None, x), dx).item()

    for _ in range(50):
        optimizer.step_derivative_matching(model, x, dx)

    with torch.no_grad():
        final_mse = F.mse_loss(model(None, x), dx).item()

    check(
        "training reduces derivative MSE",
        final_mse < initial_mse,
        f"initial={initial_mse:.4e}, final={final_mse:.4e}",
    )


def main(device_arg="auto"):
    device = sindy_torch.get_device(device_arg)
    print(f"Device: {device}")
    test_neural_ode_shapes(device)
    test_ode_model_backward_compatibility(device)
    test_ode_model_neural_ode(device)
    test_gradient_optimizer_default(device)
    test_neural_ode_training_decreases_loss(device)

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    if FAIL_COUNT == 0:
        print("All Neural ODE verifications PASSED.")
    else:
        print("Some Neural ODE verifications FAILED; see above.")
    print(f"{'=' * 50}")
    return FAIL_COUNT == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sindy_torch.add_device_arg(parser)
    args = parser.parse_args()
    success = main(args.device)
    sys.exit(0 if success else 1)
