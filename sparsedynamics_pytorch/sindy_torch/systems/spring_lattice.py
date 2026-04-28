from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn


ForcingFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class SpringLatticeODE(nn.Module):
    """
    2D spring-lattice ODE model.

    State vector:
        y = [x_flat, v_flat]

    Governing equation:
        dx/dt = v
        dv/dt = M^{-1}(F_spring + F_damping + F_external)

    Optional fixed boundary nodes are enforced by setting:
        velocity = 0
        acceleration = 0
    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        mass_init: float = 1.0,
        stiffness_init: float = 10.0,
        damping_init: float = 0.5,
        forcing_fn: Optional[ForcingFn] = None,
        fix_boundary: bool = True,
        fixed_nodes: Optional[list[int]] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_nodes = n_rows * n_cols
        self.dim = 2
        self.device = torch.device(device)
        self.dtype = dtype
        self.forcing_fn = forcing_fn
        self.fix_boundary = fix_boundary

        edges = self._build_edges()
        rest_positions = self._build_rest_positions()
        rest_lengths = self._compute_rest_lengths(rest_positions, edges)

        self.register_buffer("edges", edges)
        self.register_buffer("rest_positions", rest_positions)
        self.register_buffer("rest_lengths", rest_lengths)

        if fixed_nodes is None and fix_boundary:
            fixed_nodes = self._get_boundary_nodes()
        elif fixed_nodes is None:
            fixed_nodes = []

        self.register_buffer(
            "fixed_nodes",
            torch.tensor(fixed_nodes, device=self.device, dtype=torch.long),
        )

        n_edges = edges.shape[0]

        self.mass = nn.Parameter(
            mass_init
            * torch.ones(self.n_nodes, self.dim, device=self.device, dtype=self.dtype)
        )

        self.k = nn.Parameter(
            stiffness_init * torch.ones(n_edges, device=self.device, dtype=self.dtype)
        )

        self.c = nn.Parameter(
            damping_init
            * torch.ones(self.n_nodes, self.dim, device=self.device, dtype=self.dtype)
        )

    def _node_id(self, i: int, j: int) -> int:
        return i * self.n_cols + j

    def _build_edges(self) -> torch.Tensor:
        edges = []

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                current = self._node_id(i, j)

                if j + 1 < self.n_cols:
                    edges.append((current, self._node_id(i, j + 1)))

                if i + 1 < self.n_rows:
                    edges.append((current, self._node_id(i + 1, j)))

        return torch.tensor(edges, device=self.device, dtype=torch.long)

    def _build_rest_positions(self) -> torch.Tensor:
        positions = []

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                positions.append([float(j), float(i)])

        return torch.tensor(positions, device=self.device, dtype=self.dtype)

    def _get_boundary_nodes(self) -> list[int]:
        boundary_nodes = []

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if (
                    i == 0
                    or i == self.n_rows - 1
                    or j == 0
                    or j == self.n_cols - 1
                ):
                    boundary_nodes.append(self._node_id(i, j))

        return boundary_nodes

    @staticmethod
    def _compute_rest_lengths(
        rest_positions: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        p_i = rest_positions[edges[:, 0]]
        p_j = rest_positions[edges[:, 1]]
        return torch.linalg.norm(p_j - p_i, dim=1)

    def spring_forces(self, x: torch.Tensor) -> torch.Tensor:
        forces = torch.zeros_like(x)

        i_nodes = self.edges[:, 0]
        j_nodes = self.edges[:, 1]

        x_i = x[i_nodes]
        x_j = x[j_nodes]

        dx = x_j - x_i
        lengths = torch.linalg.norm(dx, dim=1).clamp_min(1e-12)
        directions = dx / lengths[:, None]

        extensions = lengths - self.rest_lengths
        edge_forces = self.k[:, None] * extensions[:, None] * directions

        forces.index_add_(0, i_nodes, edge_forces)
        forces.index_add_(0, j_nodes, -edge_forces)

        return forces

    def external_force(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if self.forcing_fn is None:
            return torch.zeros_like(x)

        return self.forcing_fn(t, x, v)

    def apply_fixed_nodes(
        self,
        v: torch.Tensor,
        acceleration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fixed_nodes.numel() == 0:
            return v, acceleration

        v = v.clone()
        acceleration = acceleration.clone()

        v[self.fixed_nodes, :] = 0.0
        acceleration[self.fixed_nodes, :] = 0.0

        return v, acceleration

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = self.n_nodes * self.dim

        x = y[:n].reshape(self.n_nodes, self.dim)
        v = y[n:].reshape(self.n_nodes, self.dim)

        f_spring = self.spring_forces(x)
        f_damping = -self.c * v
        f_external = self.external_force(t, x, v)

        acceleration = (f_spring + f_damping + f_external) / self.mass

        v, acceleration = self.apply_fixed_nodes(v, acceleration)

        dydt = torch.cat(
            [
                v.reshape(-1),
                acceleration.reshape(-1),
            ]
        )

        return dydt


def make_no_forcing() -> ForcingFn:
    def forcing_fn(t: torch.Tensor, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    return forcing_fn


def make_constant_node_force(
    node_id: int,
    force_vector: tuple[float, float],
) -> ForcingFn:
    def forcing_fn(t: torch.Tensor, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        force = torch.zeros_like(x)
        force[node_id, 0] = force_vector[0]
        force[node_id, 1] = force_vector[1]
        return force

    return forcing_fn


def make_sinusoidal_node_force(
    node_id: int,
    force_vector: tuple[float, float],
    omega: float = 1.0,
) -> ForcingFn:
    def forcing_fn(t: torch.Tensor, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        force = torch.zeros_like(x)
        scale = torch.sin(omega * t)
        force[node_id, 0] = force_vector[0] * scale
        force[node_id, 1] = force_vector[1] * scale
        return force

    return forcing_fn