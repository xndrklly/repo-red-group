from __future__ import annotations

from pathlib import Path

import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

from sindy_torch.systems import (
    SpringLatticeODE,
    make_no_forcing,
    make_constant_node_force,
    make_sinusoidal_node_force,
)


def build_forcing(forcing_type: str, n_cols: int):
    forced_node = n_cols - 1  # top-right node for this indexing convention

    if forcing_type == "none":
        return make_no_forcing()

    if forcing_type == "constant":
        return make_constant_node_force(
            node_id=forced_node,
            force_vector=(0.0, -1.0),
        )

    if forcing_type == "sinusoidal":
        return make_sinusoidal_node_force(
            node_id=forced_node,
            force_vector=(0.0, -1.0),
            omega=1.0,
        )

    raise ValueError(
        f"Unknown forcing_type={forcing_type}. "
        "Use 'none', 'constant', or 'sinusoidal'."
    )


def compute_derivatives(model, t, y):
    """
    Compute dy/dt directly from the known forward ODE model.
    This gives clean derivative data for SINDy.
    """
    dy = torch.stack([model(t_i, y_i) for t_i, y_i in zip(t, y)])
    return dy


def generate_spring_lattice_data(
    forcing_type: str = "none",
    n_rows: int = 3,
    n_cols: int = 3,
    t_final: float = 10.0,
    n_time: int = 500,
    mass_init: float = 1.0,
    stiffness_init: float = 20.0,
    damping_init: float = 1.0,
    initial_center_displacement: float = 0.25,
    save_plot: bool = True,
):
    device = "cpu"
    dtype = torch.float64

    forcing_fn = build_forcing(forcing_type, n_cols)

    model = SpringLatticeODE(
        n_rows=n_rows,
        n_cols=n_cols,
        mass_init=mass_init,
        stiffness_init=stiffness_init,
        damping_init=damping_init,
        forcing_fn=forcing_fn,
        fix_boundary=True,
        device=device,
        dtype=dtype,
    )

    x0 = model.rest_positions.clone()
    v0 = torch.zeros_like(x0)

    center_node = (n_rows * n_cols) // 2
    x0[center_node, 1] += initial_center_displacement

    # Enforce fixed boundary nodes in the initial condition.
    x0[model.fixed_nodes, :] = model.rest_positions[model.fixed_nodes, :]
    v0[model.fixed_nodes, :] = 0.0

    y0 = torch.cat([x0.reshape(-1), v0.reshape(-1)])

    t = torch.linspace(0.0, t_final, n_time, device=device, dtype=dtype)

    with torch.no_grad():
        y = odeint(model, y0, t, rtol=1e-6, atol=1e-8)
        dy = compute_derivatives(model, t, y)

    n_state_half = model.n_nodes * model.dim

    x = y[:, :n_state_half].reshape(n_time, model.n_nodes, model.dim)
    v = y[:, n_state_half:].reshape(n_time, model.n_nodes, model.dim)

    dxdt = dy[:, :n_state_half].reshape(n_time, model.n_nodes, model.dim)
    dvdt = dy[:, n_state_half:].reshape(n_time, model.n_nodes, model.dim)

    output_dir = Path("data") / "spring_lattice"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"spring_lattice_{n_rows}x{n_cols}_{forcing_type}.pt"

    torch.save(
        {
            "description": "SINDy-ready 2D spring-lattice forward ODE simulation dataset",
            "forcing_type": forcing_type,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_nodes": model.n_nodes,
            "dim": model.dim,
            "center_node": center_node,
            "fixed_nodes": model.fixed_nodes.cpu(),
            "t": t.cpu(),
            "y": y.cpu(),
            "dy": dy.cpu(),
            "x": x.cpu(),
            "v": v.cpu(),
            "dxdt": dxdt.cpu(),
            "dvdt": dvdt.cpu(),
            "edges": model.edges.cpu(),
            "rest_positions": model.rest_positions.cpu(),
            "rest_lengths": model.rest_lengths.cpu(),
            "mass": model.mass.detach().cpu(),
            "stiffness": model.k.detach().cpu(),
            "damping": model.c.detach().cpu(),
            "notes": {
                "state_definition": "y = [x_flat, v_flat]",
                "derivative_definition": "dy = [dxdt_flat, dvdt_flat]",
                "equation": "dx/dt = v; dv/dt = M^{-1}(F_spring + F_damping + F_external)",
                "boundary_condition": "fixed boundary nodes have velocity and acceleration set to zero",
            },
        },
        output_path,
    )

    print(f"Saved SINDy-ready dataset to: {output_path}")
    print(f"y shape:     {tuple(y.shape)}")
    print(f"dy shape:    {tuple(dy.shape)}")
    print(f"x shape:     {tuple(x.shape)}")
    print(f"v shape:     {tuple(v.shape)}")
    print(f"dxdt shape:  {tuple(dxdt.shape)}")
    print(f"dvdt shape:  {tuple(dvdt.shape)}")

    if save_plot:
        fig_dir = Path("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)

        center_displacement = x[:, center_node, 1].cpu() - model.rest_positions[center_node, 1].cpu()

        plt.figure(figsize=(7, 4))
        plt.plot(t.cpu(), center_displacement)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Time")
        plt.ylabel("Center node vertical displacement")
        plt.title(f"{n_rows}x{n_cols} Spring Lattice: {forcing_type} forcing")
        plt.tight_layout()

        fig_path = fig_dir / f"spring_lattice_{n_rows}x{n_cols}_{forcing_type}.png"
        plt.savefig(fig_path, dpi=300)
        plt.show()

        print(f"Saved plot to: {fig_path}")

    return output_path


if __name__ == "__main__":
    generate_spring_lattice_data(forcing_type="none")
    generate_spring_lattice_data(forcing_type="constant")
    generate_spring_lattice_data(forcing_type="sinusoidal")