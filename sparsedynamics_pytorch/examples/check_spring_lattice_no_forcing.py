from __future__ import annotations

import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

from sindy_torch.systems import SpringLatticeODE, make_no_forcing


def main():
    device = "cpu"
    dtype = torch.float64

    n_rows = 3
    n_cols = 3

    model = SpringLatticeODE(
        n_rows=n_rows,
        n_cols=n_cols,
        mass_init=1.0,
        stiffness_init=20.0,
        damping_init=1.0,
        forcing_fn=make_no_forcing(),
        fix_boundary=True,
        device=device,
        dtype=dtype,
    )

    x0 = model.rest_positions.clone()
    v0 = torch.zeros_like(x0)

    center_node = (n_rows * n_cols) // 2
    x0[center_node, 1] += 0.25

    # Make sure fixed nodes start exactly at rest with zero velocity
    x0[model.fixed_nodes, :] = model.rest_positions[model.fixed_nodes, :]
    v0[model.fixed_nodes, :] = 0.0

    y0 = torch.cat([x0.reshape(-1), v0.reshape(-1)])

    t = torch.linspace(0.0, 10.0, 500, device=device, dtype=dtype)

    with torch.no_grad():
        y = odeint(model, y0, t, rtol=1e-6, atol=1e-8)

    n_state_half = model.n_nodes * model.dim
    x = y[:, :n_state_half].reshape(len(t), model.n_nodes, model.dim)

    center_y = x[:, center_node, 1].cpu()
    rest_center_y = model.rest_positions[center_node, 1].cpu()
    displacement = center_y - rest_center_y

    initial_amp = torch.abs(displacement[0])
    final_amp = torch.abs(displacement[-1])

    print(f"Fixed nodes: {model.fixed_nodes.tolist()}")
    print(f"Free center node: {center_node}")
    print(f"Initial displacement magnitude: {initial_amp:.6f}")
    print(f"Final displacement magnitude:   {final_amp:.6f}")

    if final_amp < 0.01 * initial_amp:
        print("PASS: damped, fixed-boundary system decays close to equilibrium.")
    else:
        print("WARNING: system decayed, but not close to zero by final time.")

    plt.figure(figsize=(7, 4))
    plt.plot(t.cpu(), displacement)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Center node vertical displacement")
    plt.title("No-Forcing Sanity Check with Fixed Boundary")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()