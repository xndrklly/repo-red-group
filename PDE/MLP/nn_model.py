"""
nn_model.py
-----------
Simple MLP: strain invariants [I1, I2] -> stress [s11, s22, s12]
Zero-stress at zero-strain enforced by subtracting NN(0) in forward.
"""

import torch
import torch.nn as nn


class ConstitutiveNN(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()

        net = [nn.Linear(2, hidden), nn.Softplus()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Softplus()]
        net += [nn.Linear(hidden, 3)]   # output: [s11, s22, s12]

        self.net = nn.Sequential(*net)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inv):
        """
        inv : (..., 2)  strain invariants [I1, I2]
        returns : (..., 3)  stress [s11, s22, s12]
        """
        sigma = self.net(inv)

        # Enforce sigma(0)=0 exactly for zero-strain consistency.
        inv0 = torch.zeros(1, 2, device=inv.device, dtype=inv.dtype)
        sigma0 = self.net(inv0)
        return sigma - sigma0
