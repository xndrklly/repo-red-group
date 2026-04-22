# SINDy-PyTorch: Sparse Identification of Nonlinear Dynamics

A PyTorch implementation of the SINDy algorithm from Brunton et al. (2016), designed for extensibility and end-to-end differentiable system identification.

This package provides two model families:

- **SINDy models** for interpretable sparse equations built from candidate libraries
- **Neural ODE models** for flexible learned dynamics of the form `dx/dt = g_theta(x)`

It also provides two solver paths:

- **Classical STLS** (Sequential Thresholded Least Squares) for fast, direct sparse regression
- **Gradient-based optimization** with Adam by default, including differentiable ODE integration via `torchdiffeq`

Based on the original MATLAB code by S. L. Brunton, J. L. Proctor, and J. N. Kutz.
Paper: *"Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems"*, PNAS 2016.

For the math behind the implementation, see [`SINDy_Mathematical_Description.md`](SINDy_Mathematical_Description.md).

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- torchdiffeq
- NumPy and SciPy
- Matplotlib for the plotting examples
- MATLAB is optional, only needed to regenerate the reference fixtures
- CUDA is optional and requires a CUDA-enabled PyTorch installation

## Installation

From the repository root:

```powershell
cd sparsedynamics_pytorch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

After installation, `import sindy_torch` works from any working directory. The editable install also keeps local code changes visible without reinstalling.

For macOS/Linux shells, the activation command is:

```bash
source .venv/bin/activate
```

## Device Selection

Runnable examples and verification scripts accept a shared device flag:

```powershell
python examples\quickstart_lorenz.py --device auto
python examples\quickstart_lorenz.py --device cpu
python examples\quickstart_lorenz.py --device cuda
```

`--device auto` is the default and uses CUDA when `torch.cuda.is_available()` is true; otherwise it falls back to CPU. Passing `--device cuda` fails early with a clear error if the local PyTorch build cannot use CUDA. The long Lorenz comparison defaults to 4 CPU workers on CPU and 1 serial worker on CUDA:

```powershell
python examples\compare_methods_lorenz.py --device cuda
python examples\compare_methods_lorenz.py --device cpu --max-workers 4
```

## Quick Start

Run the checked example after installing the package:

```powershell
python examples\quickstart_lorenz.py --device auto
```

The example identifies the Lorenz equations from simulated trajectory data and checks the 7 expected nonzero coefficients.

```python
import torch
from torchdiffeq import odeint

import sindy_torch

sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0
n_states = 3
poly_order = 3

x0 = torch.tensor([-8.0, 8.0, 27.0], dtype=torch.float64)
t = torch.linspace(0.001, 5.0, 5000, dtype=torch.float64)
lorenz_rhs = lambda t_i, y: sindy_torch.lorenz(t_i, y, sigma, beta, rho)

with torch.no_grad():
    x = odeint(lorenz_rhs, x0, t, rtol=1e-10, atol=1e-10)

dx = torch.stack([lorenz_rhs(t[i], x[i]) for i in range(len(t))])

library = sindy_torch.PolynomialLibrary(n_vars=n_states, poly_order=poly_order)
theta = library(x)
xi = sindy_torch.stls(theta, dx, lam=0.025)

model = sindy_torch.SINDyModule(library, library.n_features, n_states)
model.set_xi(xi)
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
    assert abs(found - expected) < 0.01, (name, found, expected)
```

The coefficient row indexes use the polynomial library order `[1, x, y, z, x^2, xy, xz, ...]`.

Expected equations:

```text
dx/dt = -10*x + 10*y
dy/dt = 28*x - y - xz
dz/dt = xy - 2.6667*z
```

---

## Gradient-Based Trajectory Matching

`SparseOptimizer` can also train the coefficient matrix `xi` with PyTorch gradients. It defaults to Adam. For trajectory matching, choose `gradient_method="autograd"`, `"sensitivity"`, or `"adjoint"`. Recompute `theta`, `dx`, and the STLS warm start using the same library as the trainable model:

This snippet assumes you have already generated `x_true`, `t_train`, `lorenz_rhs`, and `x0` as in `examples/ex03_lorenz_trainable.py`.

```python
poly_order = 3
library = sindy_torch.PolynomialLibrary(n_vars=3, poly_order=poly_order)
theta = library(x_true)
dx_true = torch.stack([lorenz_rhs(t_train[i], x_true[i]) for i in range(len(t_train))])

xi_init = sindy_torch.stls(theta, dx_true, lam=0.025)

model = sindy_torch.SINDyModule(library, library.n_features, n_states=3)
model.set_xi(xi_init)

ode_model = sindy_torch.ODEModel(model, rtol=1e-4, atol=1e-6)
optimizer = sindy_torch.SparseOptimizer(
    model.xi,
    l1_lambda=1e-4,
    optimizer_kwargs={"lr": 1e-3},
)

for epoch in range(50):
    losses = optimizer.step_trajectory_matching(
        ode_model,
        x0,
        t_train,
        x_true,
        gradient_method="sensitivity",
    )

optimizer.threshold(tol=0.05)
```

`gradient_method="sensitivity"` and `"adjoint"` are explicit trajectory-gradient implementations in this package. `ODEModel(use_adjoint=True)` is separate: it switches the autograd backend inside `torchdiffeq` and only affects `gradient_method="autograd"`.

---

## Neural ODE Model

Use `NeuralODEModule` when you want a flexible learned dynamics model instead of an interpretable sparse equation:

```text
dx/dt = g_theta(x)
```

`GradientOptimizer` trains any dynamics module with PyTorch gradients and defaults to Adam. The included example trains a Neural ODE on derivative data from the same Linear 2D system used in the SINDy examples:

```powershell
python examples\ex04_neural_ode_linear2d.py --device auto
```

---

## Method Comparison Plots

The plotting examples save PNG files in `figures/`. The comparison scripts run multiple methods on the same data, while the standalone gradient examples save their own loss-vs-epoch plots:

```powershell
python examples\compare_methods_linear2d.py --device auto
python examples\compare_methods_lorenz.py --device auto
python examples\ex03_lorenz_trainable.py --device auto
python examples\ex04_neural_ode_linear2d.py --device auto
```

Generated plot files:

- `figures/linear2d_method_comparison.png`
- `figures/linear2d_error_comparison.png`
- `figures/linear2d_loss_epoch.png`
- `figures/lorenz_sindy_method_comparison.png`
- `figures/lorenz_neural_ode_method_comparison.png`
- `figures/lorenz_sindy_loss_epoch.png`
- `figures/lorenz_neural_ode_loss_epoch.png`
- `figures/lorenz_sindy_butterfly_3d_grid.png`
- `figures/lorenz_neural_ode_butterfly_3d_grid.png`
- `figures/lorenz_*_butterfly_3d.png`
- `figures/lorenz_method_summary.csv`
- `figures/lorenz_trainable_loss_epoch.png`
- `figures/neural_ode_linear2d_loss_epoch.png`

---

## Package Structure

```text
sparsedynamics_pytorch/
|-- pyproject.toml
|-- sindy_torch/
|   |-- library/           # Candidate function builders (nn.Module)
|   |-- solvers/           # STLS, sparse optimization, generic gradient optimization
|   |-- models/            # SINDyModule, NeuralODEModule, ODEModel
|   |-- differentiation/   # Finite differences, ODE RHS evaluation, TVRegDiff
|   |-- systems/           # Lorenz, Hopf, Logistic test systems
|   `-- utils.py
|-- examples/
|   |-- quickstart_lorenz.py
|   |-- ex01a_linear2d.py
|   |-- ex02_lorenz.py
|   |-- ex03_lorenz_trainable.py
|   |-- ex04_neural_ode_linear2d.py
|   |-- compare_methods_linear2d.py
|   `-- compare_methods_lorenz.py
`-- tests/
    |-- verify_against_matlab.py
    |-- verify_trajectory_gradients.py
    `-- matlab_reference/   # MATLAB-generated .mat fixtures
```

---

## Library Builders

All libraries inherit from `LibraryBase(nn.Module)` and support arbitrary batch dimensions.

| Library | Description | Example (2 vars) |
|---------|-------------|------------------|
| `PolynomialLibrary(n, p)` | Monomials up to degree `p` | `[1, x, y, x^2, xy, y^2, ...]` |
| `FourierLibrary(n, k)` | sin/cos for `k` harmonics | `[sin(x), cos(x), sin(y), cos(y), ...]` |
| `CompositeLibrary([...])` | Concatenate sub-libraries | Poly + Fourier + custom |
| `CustomLibrary(fns, labels)` | User-defined callables | Any `f(x) -> scalar` |

Example:

```python
library = sindy_torch.CompositeLibrary([
    sindy_torch.PolynomialLibrary(n_vars=3, poly_order=3),
    sindy_torch.FourierLibrary(n_vars=3, n_harmonics=5),
    sindy_torch.CustomLibrary(
        functions=[lambda x: torch.exp(-x[..., 0:1] ** 2)],
        labels=["exp(-x^2)"],
    ),
])
```

---

## Solvers and Models

`stls(theta, dxdt, lam, n_iter=10)` runs Sequential Thresholded Least Squares:

1. Solve least squares for `theta @ xi ~= dxdt`.
2. Set coefficients with `abs(xi) < lam` to zero.
3. Refit each state equation on its surviving terms.
4. Repeat for `n_iter` threshold/refit passes.

`SINDyModule(library, n_features, n_states)` is the interpretable sparse model. It computes:

```text
dx/dt = library(x) @ xi
```

`NeuralODEModule(n_states, hidden_width=64, hidden_depth=2)` is the flexible neural model. It computes:

```text
dx/dt = g_theta(x)
```

`ODEModel(dynamics_module, method="dopri5", use_adjoint=False)` wraps `torchdiffeq.odeint` so any dynamics module with `forward(t, x) -> dx/dt` can be integrated as an ODE. The old `sindy_module=` keyword still works for SINDy code.

When `use_adjoint=True`, `ODEModel` switches to `torchdiffeq`'s adjoint-backprop solver for the autograd training path. This is distinct from the explicit `gradient_method="adjoint"` option on the optimizers.

`SparseOptimizer` is for sparse SINDy coefficient training and defaults to Adam. It supports:

- `step_derivative_matching(theta, dxdt)`: minimize derivative prediction error plus an L1 penalty on `xi`
- `step_trajectory_matching(ode_model, x0, t, x_true, gradient_method="autograd")`: integrate the learned ODE and minimize trajectory error plus an L1 penalty, using autograd, explicit sensitivity, or explicit adjoint gradients
- `threshold(tol)`: hard sparsification
- `proximal_step()`: soft thresholding for ISTA-style proximal gradient descent

`GradientOptimizer` is model-agnostic and also defaults to Adam. It supports derivative matching and trajectory matching for any trainable dynamics module, including `NeuralODEModule`, with the same `gradient_method` options for trajectory training.

---

## Derivative Computation

Three methods are available:

| Method | When to use | Data loss |
|--------|-------------|-----------|
| `finite_difference_4th(x, dt)` | Clean, evenly spaced data | 4 samples |
| `autograd_derivative(ode, x0, t)` | Known ODE, usually synthetic data | None |
| `tv_reg_diff(data, n_iter, alpha)` | Noisy data | 0-1 samples |

`tv_reg_diff` uses SciPy/NumPy internally, so it performs its differentiation work on CPU and returns the result on the input tensor's original device.

---

## MATLAB Verification

The normal verification command does not require MATLAB. It compares PyTorch outputs against MATLAB-generated `.mat` fixtures in `tests/matlab_reference/`.

```powershell
python tests\verify_against_matlab.py --device auto
```

Current verification result:

```text
RESULTS: 26 passed, 0 failed
All MATLAB fixture verifications PASSED.
```

The checks cover:

- Polynomial library matrices for 2-variable and 3-variable inputs, within `1e-12` absolute tolerance for the small fixtures
- STLS sparsity pattern agreement with MATLAB
- STLS coefficient values within `1e-3` for the synthetic, noisy Linear 2D, and Lorenz fixture comparisons
- Expected Linear 2D and Lorenz coefficients within the tolerances used in `tests/verify_against_matlab.py`

Trajectory-gradient verification for the explicit `sensitivity` and `adjoint` backends:

```powershell
python tests\verify_trajectory_gradients.py --device auto
python tests\verify_cuda_optional.py
```

`verify_cuda_optional.py` runs CUDA smoke checks when CUDA is available and reports skipped CUDA checks on CPU-only installations.

If MATLAB is installed and you want to regenerate the fixtures, run this from the repository root:

```powershell
matlab -batch "cd('sparsedynamics'); export_for_verification('../sparsedynamics_pytorch/tests/matlab_reference'); exit"
```

Then rerun:

```powershell
cd sparsedynamics_pytorch
python tests\verify_against_matlab.py
```

---

## Reference Systems

```python
sindy_torch.lorenz(t, y, sigma=10, beta=8/3, rho=28)
sindy_torch.hopf(t, y, mu=1, omega=1, A=1)
sindy_torch.logistic(t, y, r=3.5)
```

All follow the `torchdiffeq` convention `f(t, y) -> dy` and support batch dimensions.

---

## References

- S. L. Brunton, J. L. Proctor, J. N. Kutz, "Discovering governing equations from data by sparse identification of nonlinear dynamical systems," *PNAS*, 2016.
- R. T. Q. Chen et al., "Neural Ordinary Differential Equations," *NeurIPS*, 2018.
- R. Chartrand, "Numerical differentiation of noisy, nonsmooth data," *ISRN Applied Mathematics*, 2011.
