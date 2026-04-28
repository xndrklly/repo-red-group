import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

from sindy_torch.systems import SpringLatticeODE

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
        device=device,
        dtype=dtype,
    )

    x0 = model.rest_positions.clone()
    v0 = torch.zeros_like(x0)

    # Initial perturbation on center node
    center_node = 4
    x0[center_node, 1] += 0.25

    y0 = torch.cat([
        x0.reshape(-1),
        v0.reshape(-1),
    ])

    t = torch.linspace(0.0, 10.0, 500, device=device, dtype=dtype)

    with torch.no_grad():
        y = odeint(model, y0, t, rtol=1e-6, atol=1e-8)

    n = model.n_nodes * model.dim
    x_t = y[:, :n].reshape(len(t), model.n_nodes, model.dim)
    v_t = y[:, n:].reshape(len(t), model.n_nodes, model.dim)

    # Plot vertical displacement of center node
    center_y = x_t[:, center_node, 1].cpu()
    rest_center_y = model.rest_positions[center_node, 1].cpu()
    displacement = center_y - rest_center_y

    plt.figure(figsize=(7, 4))
    plt.plot(t.cpu(), displacement)
    plt.xlabel("Time")
    plt.ylabel("Center node vertical displacement")
    plt.title("3x3 Spring Lattice Forward ODE Demo")
    plt.tight_layout()
    plt.show()

    torch.save(
        {
            "t": t.cpu(),
            "x": x_t.cpu(),
            "v": v_t.cpu(),
            "mass": model.mass.detach().cpu(),
            "stiffness": model.k.detach().cpu(),
            "damping": model.c.detach().cpu(),
            "edges": model.edges.cpu(),
            "rest_positions": model.rest_positions.cpu(),
        },
        "spring_lattice_demo.pt",
    )


if __name__ == "__main__":
    main()