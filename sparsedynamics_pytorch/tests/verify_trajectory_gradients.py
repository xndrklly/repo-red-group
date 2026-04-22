"""
Verification tests for explicit sensitivity and adjoint trajectory gradients.
"""

from __future__ import annotations

import argparse
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


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name}  {detail}")


def flatten_param_grads(params):
    chunks = []
    for param in params:
        if param.grad is None:
            chunks.append(torch.zeros(param.numel(), dtype=param.dtype, device=param.device))
        else:
            chunks.append(param.grad.detach().reshape(-1).clone())
    return torch.cat(chunks)


def autograd_loss_and_grads(ode_model, x0, t, x_true, params):
    for param in params:
        param.grad = None
    x_pred = ode_model(x0, t)
    loss = F.mse_loss(x_pred, x_true)
    loss.backward()
    flat_grad = flatten_param_grads(params)
    for param in params:
        param.grad = None
    return loss.detach(), flat_grad


def gradient_close(reference, candidate, atol=1e-5, rtol=1e-4):
    abs_err = torch.max(torch.abs(reference - candidate)).item()
    denom = max(reference.norm().item(), 1e-12)
    rel_err = torch.norm(reference - candidate).item() / denom
    return abs_err <= atol and rel_err <= rtol, abs_err, rel_err


def make_linear_data(device, dtype=torch.float64):
    A = torch.tensor([[-0.2, 1.0], [-1.0, -0.3]], dtype=dtype, device=device)
    rhs = lambda t, x: x @ A.T
    x0 = torch.tensor([1.0, -0.5], dtype=dtype, device=device)
    t = torch.linspace(0.0, 0.6, 21, dtype=dtype, device=device)
    with torch.no_grad():
        x_true = odeint(rhs, x0, t, rtol=1e-10, atol=1e-10)
    dx_true = x_true @ A.T
    return x0, t, x_true, dx_true


def build_sindy_model(device, dtype=torch.float64):
    library = sindy_torch.PolynomialLibrary(n_vars=2, poly_order=1)
    model = sindy_torch.SINDyModule(library, library.n_features, n_states=2).to(device)
    xi = torch.zeros(library.n_features, 2, dtype=dtype, device=device)
    xi[0, 0] = 0.05
    xi[1, 0] = -0.18
    xi[2, 0] = 0.88
    xi[0, 1] = -0.04
    xi[1, 1] = -0.92
    xi[2, 1] = -0.22
    model.set_xi(xi)
    return model


def build_neural_model(device, dtype=torch.float64):
    torch.manual_seed(5)
    return sindy_torch.NeuralODEModule(
        n_states=2,
        hidden_width=4,
        hidden_depth=1,
        dtype=dtype,
        device=device,
    )


def test_sindy_gradient_agreement(device):
    print("\n=== Test 1: SINDy manual gradients match autograd ===")
    x0, t, x_true, _ = make_linear_data(device)
    model = build_sindy_model(device)
    ode_model = sindy_torch.ODEModel(model, rtol=1e-8, atol=1e-10)
    params = [model.xi]

    auto_loss, auto_grad = autograd_loss_and_grads(ode_model, x0, t, x_true, params)
    sens_loss, sens_grad = manual_trajectory_loss_and_gradients(
        ode_model, x0, t, x_true, params, "sensitivity"
    )
    adj_loss, adj_grad = manual_trajectory_loss_and_gradients(
        ode_model, x0, t, x_true, params, "adjoint"
    )

    sens_ok, sens_abs, sens_rel = gradient_close(auto_grad, sens_grad)
    adj_ok, adj_abs, adj_rel = gradient_close(auto_grad, adj_grad, atol=5e-5, rtol=5e-4)

    check("SINDy sensitivity loss", torch.allclose(auto_loss, sens_loss, atol=1e-8))
    check("SINDy adjoint loss", torch.allclose(auto_loss, adj_loss, atol=1e-8))
    check(
        "SINDy sensitivity gradient",
        sens_ok,
        f"abs_err={sens_abs:.4e}, rel_err={sens_rel:.4e}",
    )
    check(
        "SINDy adjoint gradient",
        adj_ok,
        f"abs_err={adj_abs:.4e}, rel_err={adj_rel:.4e}",
    )


def test_neural_gradient_agreement(device):
    print("\n=== Test 2: Neural ODE manual gradients match autograd ===")
    x0, t, x_true, _ = make_linear_data(device)
    model = build_neural_model(device)
    ode_model = sindy_torch.ODEModel(model, rtol=1e-8, atol=1e-10)
    params = list(model.parameters())

    auto_loss, auto_grad = autograd_loss_and_grads(ode_model, x0, t, x_true, params)
    sens_loss, sens_grad = manual_trajectory_loss_and_gradients(
        ode_model, x0, t, x_true, params, "sensitivity"
    )
    adj_loss, adj_grad = manual_trajectory_loss_and_gradients(
        ode_model, x0, t, x_true, params, "adjoint"
    )

    sens_ok, sens_abs, sens_rel = gradient_close(auto_grad, sens_grad, atol=1e-5, rtol=2e-4)
    adj_ok, adj_abs, adj_rel = gradient_close(auto_grad, adj_grad, atol=5e-5, rtol=1e-3)

    check("Neural sensitivity loss", torch.allclose(auto_loss, sens_loss, atol=1e-8))
    check("Neural adjoint loss", torch.allclose(auto_loss, adj_loss, atol=1e-8))
    check(
        "Neural sensitivity gradient",
        sens_ok,
        f"abs_err={sens_abs:.4e}, rel_err={sens_rel:.4e}",
    )
    check(
        "Neural adjoint gradient",
        adj_ok,
        f"abs_err={adj_abs:.4e}, rel_err={adj_rel:.4e}",
    )


def test_manual_training_reduces_sindy_loss(device):
    print("\n=== Test 3: SINDy manual trajectory training reduces loss ===")
    x0, t, x_true, _ = make_linear_data(device)

    for method in ("sensitivity", "adjoint"):
        model = build_sindy_model(device)
        ode_model = sindy_torch.ODEModel(model, rtol=1e-8, atol=1e-10)
        optimizer = sindy_torch.SparseOptimizer(
            model.xi,
            l1_lambda=0.0,
            optimizer_kwargs={"lr": 5e-2},
        )
        with torch.no_grad():
            initial_loss = F.mse_loss(ode_model(x0, t), x_true).item()
        for _ in range(8):
            optimizer.step_trajectory_matching(
                ode_model,
                x0,
                t,
                x_true,
                gradient_method=method,
            )
        with torch.no_grad():
            final_loss = F.mse_loss(ode_model(x0, t), x_true).item()
        check(
            f"SINDy {method} training",
            final_loss < initial_loss,
            f"initial={initial_loss:.4e}, final={final_loss:.4e}",
        )


def test_manual_training_reduces_neural_loss(device):
    print("\n=== Test 4: Neural ODE manual trajectory training reduces loss ===")
    x0, t, x_true, _ = make_linear_data(device)

    for method in ("sensitivity", "adjoint"):
        model = build_neural_model(device)
        ode_model = sindy_torch.ODEModel(model, rtol=1e-8, atol=1e-10)
        optimizer = sindy_torch.GradientOptimizer(
            model,
            optimizer_kwargs={"lr": 1e-2},
        )
        with torch.no_grad():
            initial_loss = F.mse_loss(ode_model(x0, t), x_true).item()
        for _ in range(10):
            optimizer.step_trajectory_matching(
                ode_model,
                x0,
                t,
                x_true,
                gradient_method=method,
            )
        with torch.no_grad():
            final_loss = F.mse_loss(ode_model(x0, t), x_true).item()
        check(
            f"Neural {method} training",
            final_loss < initial_loss,
            f"initial={initial_loss:.4e}, final={final_loss:.4e}",
        )


def test_batched_inputs_rejected(device):
    print("\n=== Test 5: Batched inputs rejected for manual modes ===")
    x0, t, x_true, _ = make_linear_data(device)
    model = build_neural_model(device)
    ode_model = sindy_torch.ODEModel(model, rtol=1e-8, atol=1e-10)
    optimizer = sindy_torch.GradientOptimizer(model, optimizer_kwargs={"lr": 1e-2})

    x0_batch = torch.stack([x0, x0 + 0.1], dim=0)
    x_true_batch = torch.stack([x_true, x_true], dim=1)
    for method in ("sensitivity", "adjoint"):
        try:
            optimizer.step_trajectory_matching(
                ode_model,
                x0_batch,
                t,
                x_true_batch,
                gradient_method=method,
            )
        except ValueError as exc:
            message = str(exc)
            check(
                f"Batched {method} error",
                "use gradient_method='autograd'" in message,
                message,
            )
        else:
            check(f"Batched {method} error", False, "ValueError not raised")


def main(device_arg="auto"):
    device = sindy_torch.get_device(device_arg)
    print(f"Device: {device}")
    test_sindy_gradient_agreement(device)
    test_neural_gradient_agreement(device)
    test_manual_training_reduces_sindy_loss(device)
    test_manual_training_reduces_neural_loss(device)
    test_batched_inputs_rejected(device)

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    if FAIL_COUNT == 0:
        print("All trajectory-gradient verifications PASSED.")
    else:
        print("Some trajectory-gradient verifications FAILED; see above.")
    print(f"{'=' * 50}")
    return FAIL_COUNT == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sindy_torch.add_device_arg(parser)
    args = parser.parse_args()
    success = main(args.device)
    sys.exit(0 if success else 1)
