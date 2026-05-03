"""
nn_model.py  (ICNN)
-------------------
Input Convex Neural Network for hyperelastic constitutive modeling.

Architecture (Amos et al. 2017, adapted for small-strain 2D):
  z0         = [I1 - I1_ref, I2 - I2_ref]           shifted invariants
  z^(1)      = softplus( B^(1) z0 )                  first hidden layer
  z^(k≥2)   = act_scale * sq_softplus( A^(k) z^(k-1) )   z-path
             + softplus( B^(k) z0 )                       skip path
  W          = A_out @ z^(L) + B_out @ z0            scalar output

Convexity: A^(k) = softplus(raw_A^(k)) ≥ 0.
act_scale (default 1/12): prevents gradient explosion in deep nets —
    borrowed directly from nn-EUCLID (config.py: scaling_sftpSq = 1/12).

Stresses via autograd chain rule:
  σ₁₁ = ∂W/∂I₁ + 2 ε₁₁ · ∂W/∂I₂
  σ₂₂ = ∂W/∂I₁ + 2 ε₂₂ · ∂W/∂I₂
  σ₁₂ =          2 ε₁₂ · ∂W/∂I₂

Zero-stress at rest: W_c = W(z0) - W(0) - ∇W(0)·z0
    (subtracting a linear fn preserves convexity; enforces W(0)=0 AND ∇W(0)=0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICNN(nn.Module):
    def __init__(self, hidden=32, layers=3, act_scale=1/12, dropout=0.0):
        """
        hidden    : hidden units per layer
        layers    : number of hidden layers
        act_scale : scale applied after squared softplus (nn-EUCLID: 1/12)
        dropout   : dropout probability during training (0 = disabled)
        """
        super().__init__()

        input_dim = 2
        self.act_scale = act_scale
        self.drop = nn.Dropout(p=dropout)

        # Layer 1: only skip path (no z_prev yet)
        self.B_first = nn.Linear(input_dim, hidden)

        # Layers 2 ... L
        n_extra = layers - 1
        self.raw_As = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden, hidden)) for _ in range(n_extra)]
        )
        self.Bs = nn.ModuleList(
            [nn.Linear(input_dim, hidden) for _ in range(n_extra)]
        )

        # Output layer: scalar W
        self.raw_A_out = nn.Parameter(torch.zeros(1, hidden))
        self.B_out = nn.Linear(input_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.B_first.weight)
        for B in self.Bs:
            nn.init.xavier_normal_(B.weight)
        nn.init.xavier_normal_(self.B_out.weight)

    def _sq_softplus(self, x):
        """act_scale * softplus(x)² — convex, non-decreasing, scaled to prevent explosion."""
        return self.act_scale * F.softplus(x) ** 2

    def _forward_W(self, z0):
        """
        z0 : (batch, 2)  shifted invariants
        Returns W : (batch, 1)
        """
        # Layer 1 — skip only, plain softplus (no z_prev)
        z = F.softplus(self.B_first(z0))
        z = self.drop(z)

        # Layers 2 ... L
        for raw_A, B in zip(self.raw_As, self.Bs):
            A = F.softplus(raw_A)                           # non-negative weights
            z = self._sq_softplus(z @ A.T) + F.softplus(B(z0))
            z = self.drop(z)

        # Scalar output — no activation
        A_out = F.softplus(self.raw_A_out)
        W = z @ A_out.T + self.B_out(z0)                   # (batch, 1)
        return W

    def cache_ref(self):
        """
        Pre-compute the thermodynamic correction at the reference point (z0=0)
        and cache it for use in forward().

        Call this ONCE per optimizer step — before the batch of forward passes.
        For L-BFGS this means at the top of the closure, not inside it, so the
        cached values are reused across all ~20 line-search sub-steps per epoch.
        For Adam, call it once before opt.step().

        This avoids running a full second network forward+backward on every
        closure evaluation, roughly halving per-epoch cost on large grids.
        """
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        with torch.enable_grad():
            z0_ref = torch.zeros(1, 2, device=device, dtype=dtype,
                                 requires_grad=True)
            W_ref = self._forward_W(z0_ref)
            grad_ref = torch.autograd.grad(
                W_ref.sum(), z0_ref, create_graph=self.training
            )[0]
        # Keep graph-connected tensors for training (needed for weight grads),
        # plain detached scalars for eval.
        self._ref_W        = W_ref
        self._ref_dWdI1    = grad_ref[:, 0:1]

    def _get_ref(self, device, dtype):
        """Return (W_ref, dWdI1_ref), computing inline if cache_ref() hasn't been called."""
        if hasattr(self, '_ref_W'):
            return self._ref_W, self._ref_dWdI1
        # Fallback: compute inline (e.g. first inference call before any cache_ref)
        with torch.enable_grad():
            z0_ref = torch.zeros(1, 2, device=device, dtype=dtype, requires_grad=True)
            W_ref = self._forward_W(z0_ref)
            grad_ref = torch.autograd.grad(
                W_ref.sum(), z0_ref, create_graph=self.training
            )[0]
        return W_ref, grad_ref[:, 0:1]

    def forward(self, eps_flat):
        """
        eps_flat : (batch, 3)  — [e11, e22, e12]
        Returns  : (batch, 3)  — [s11, s22, s12]

        Call cache_ref() once before each optimizer step for best performance.
        """
        e11 = eps_flat[:, 0]
        e22 = eps_flat[:, 1]
        e12 = eps_flat[:, 2]

        I1 = e11 + e22
        I2 = e11 ** 2 + 2.0 * e12 ** 2 + e22 ** 2

        with torch.enable_grad():
            z0 = torch.stack([I1, I2], dim=-1).requires_grad_(True)

            W = self._forward_W(z0)

            # Thermodynamic correction: W_c = W(z0) - W(0) - (∂W/∂I1)|₀ · I1
            #
            # Only the I1 term is subtracted, not the I2 term. Reason:
            #   s11 = ∂W/∂I1 + 2ε11·∂W/∂I2  → zero at ε=0 requires ∂W/∂I1|₀=0
            #   s12 = 2ε12·∂W/∂I2            → zero at ε=0 because ε12=0, NOT
            #                                    because ∂W/∂I2|₀=0
            # Subtracting the I2 linear term (as the full ∇W·z0 correction does)
            # forces ∂W/∂I2|₀=0, which kills the linear shear modulus.
            W_ref, dWdI1_ref = self._get_ref(z0.device, z0.dtype)
            W = W - W_ref - z0[:, 0:1] * dWdI1_ref

            grads = torch.autograd.grad(
                W.sum(), z0, create_graph=self.training
            )[0]

        dW_dI1 = grads[:, 0]
        dW_dI2 = grads[:, 1]

        s11 = dW_dI1 + 2.0 * e11.detach() * dW_dI2
        s22 = dW_dI1 + 2.0 * e22.detach() * dW_dI2
        s12 = 2.0 * e12.detach() * dW_dI2

        return torch.stack([s11, s22, s12], dim=-1)
