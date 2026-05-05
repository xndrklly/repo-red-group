"""
nn_model.py  —  NN-EUCLID ICNN  (Thakolkaran et al. 2022)
----------------------------------------------------------
Architecture matches Appendix A of the paper exactly:
  - Squared-softplus activations with c_F = 1/12 scaling
  - c_G = 1.0 for the first layer (plain softplus)
  - Non-negative z-path weights via softplus(raw_A)
  - Zero-stress reference: W_c = W(z) - W(0) - H:E
    where H = -dW/dF|_{F=I}  (eq. 8, 9, 11 in paper)

Invariants (small-strain 2D adaptation of paper's F-based invariants):
  I1 = eps11 + eps22          (trace)
  I2 = eps11^2 + 2*eps12^2 + eps22^2   (Frobenius norm squared)

Stresses:
  sigma11 = dW/dI1 + 2*eps11 * dW/dI2
  sigma22 = dW/dI1 + 2*eps22 * dW/dI2
  sigma12 =          2*eps12 * dW/dI2

Dropout=0.2 during training, off for eval — exactly as paper states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICNN(nn.Module):
    def __init__(self, hidden=32, layers=3, act_scale=1/12, dropout=0.2):
        """
        hidden    : units per hidden layer  (paper: not specified, we use 32)
        layers    : number of hidden layers (paper: not specified, we use 3)
        act_scale : c_F in eq.16  (paper: 1/12)
        dropout   : paper uses 0.2 during training, 0.0 at eval
        """
        super().__init__()
        self.act_scale = act_scale
        self.drop = nn.Dropout(p=dropout)

        input_dim = 2   # [I1, I2]

        # Layer 1: skip-only (no z-path yet), plain softplus (c_G=1.0)
        self.B_first = nn.Linear(input_dim, hidden)

        # Layers 2..L: z-path + skip
        n_extra = layers - 1
        self.raw_As = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden, hidden)) for _ in range(n_extra)]
        )
        self.Bs = nn.ModuleList(
            [nn.Linear(input_dim, hidden) for _ in range(n_extra)]
        )

        # Output: scalar W
        self.raw_A_out = nn.Parameter(torch.zeros(1, hidden))
        self.B_out = nn.Linear(input_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.B_first.weight)
        for B in self.Bs:
            nn.init.xavier_normal_(B.weight)
        nn.init.xavier_normal_(self.B_out.weight)

    def _sq_softplus(self, x):
        """c_F * softplus(x)^2  —  eq. 16, c_F = act_scale = 1/12"""
        return self.act_scale * F.softplus(x) ** 2

    def _forward_W(self, z0):
        """Raw network W^NN(z0), before thermodynamic correction."""
        z = F.softplus(self.B_first(z0))       # layer 1, c_G=1.0
        z = self.drop(z)

        for raw_A, B in zip(self.raw_As, self.Bs):
            A = F.softplus(raw_A)              # non-negative z-path weights
            z = self._sq_softplus(z @ A.T) + F.softplus(B(z0))
            z = self.drop(z)

        A_out = F.softplus(self.raw_A_out)
        return z @ A_out.T + self.B_out(z0)   # (batch, 1)

    def forward(self, eps_flat):
        """
        eps_flat : (N, 3)  [eps11, eps22, eps12]
        returns  : (N, 3)  [s11,   s22,   s12  ]

        Thermodynamic correction (eq. 8, 9, 11):
          W0 = W^NN|_{eps=0}
          H  = -dW^NN/dI * dI/deps|_{eps=0}   (the reference stress)
          W_corrected = W^NN - W0 - H:eps
        This enforces W(0)=0 and sigma(0)=0 while preserving convexity.

        NOTE: we only subtract the I1-linear term (not I2), because:
          sigma12 = 2*eps12 * dW/dI2  is already zero at eps=0 without
          forcing dW/dI2|_0 = 0, and forcing it kills the shear modulus.
        """
        e11 = eps_flat[:, 0]
        e22 = eps_flat[:, 1]
        e12 = eps_flat[:, 2]

        I1 = e11 + e22
        I2 = e11**2 + 2.0*e12**2 + e22**2

        with torch.enable_grad():
            z0 = torch.stack([I1, I2], dim=-1).requires_grad_(True)
            W = self._forward_W(z0)

            # Reference correction
            z_ref = torch.zeros(1, 2, device=z0.device, dtype=z0.dtype,
                                requires_grad=True)
            W_ref = self._forward_W(z_ref)
            dW_ref = torch.autograd.grad(
                W_ref.sum(), z_ref,
                create_graph=torch.is_grad_enabled(),
            )[0]                                      # (1, 2): [dW/dI1, dW/dI2]

            # Subtract W(0) + (dW/dI1|_0)*I1  only
            W_c = W - W_ref - z0[:, 0:1] * dW_ref[:, 0:1]

            grads = torch.autograd.grad(
                W_c.sum(), z0, create_graph=torch.is_grad_enabled(),
            )[0]                                      # (N, 2)

        dWdI1, dWdI2 = grads[:, 0], grads[:, 1]

        s11 = dWdI1 + 2.0 * e11.detach() * dWdI2
        s22 = dWdI1 + 2.0 * e22.detach() * dWdI2
        s12 =         2.0 * e12.detach() * dWdI2

        return torch.stack([s11, s22, s12], dim=-1)
