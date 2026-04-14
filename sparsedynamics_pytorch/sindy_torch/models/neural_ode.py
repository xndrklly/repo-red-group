"""
Neural ODE right-hand-side module.

This is a generic trainable dynamics model with the same torchdiffeq
signature as SINDyModule: forward(t, x) -> dx/dt.
"""

from typing import Type

import torch
from torch import Tensor, nn


class NeuralODEModule(nn.Module):
    """Autonomous Neural ODE dynamics: dx/dt = net(x).

    Parameters
    ----------
    n_states : int
        Number of state variables.
    hidden_width : int
        Width of each hidden layer.
    hidden_depth : int
        Number of hidden layers. If 0, use a direct linear map.
    activation_cls : type
        Activation module class used between hidden layers.
    dtype : torch.dtype
        Parameter dtype.
    device : torch.device, optional
        Parameter device.
    """

    def __init__(
        self,
        n_states: int,
        hidden_width: int = 64,
        hidden_depth: int = 2,
        activation_cls: Type[nn.Module] = nn.Tanh,
        dtype: torch.dtype = torch.float64,
        device: torch.device = None,
    ):
        super().__init__()
        if n_states < 1:
            raise ValueError("n_states must be positive")
        if hidden_depth < 0:
            raise ValueError("hidden_depth must be non-negative")
        if hidden_depth > 0 and hidden_width < 1:
            raise ValueError("hidden_width must be positive when hidden_depth > 0")

        self.n_states = n_states

        layers = []
        in_features = n_states
        for _ in range(hidden_depth):
            layers.append(
                nn.Linear(
                    in_features,
                    hidden_width,
                    dtype=dtype,
                    device=device,
                )
            )
            layers.append(activation_cls())
            in_features = hidden_width

        layers.append(
            nn.Linear(
                in_features,
                n_states,
                dtype=dtype,
                device=device,
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute dx/dt from the current state.

        t is accepted for torchdiffeq compatibility. The default Neural ODE is
        autonomous, so time is not used.
        """
        if x.shape[-1] != self.n_states:
            raise ValueError(
                f"Expected last dimension {self.n_states}, got {x.shape[-1]}"
            )
        return self.net(x)

    def predict_derivative(self, x: Tensor) -> Tensor:
        """Convenience: compute dx/dt without a meaningful time argument."""
        return self.forward(torch.tensor(0.0, device=x.device, dtype=x.dtype), x)
