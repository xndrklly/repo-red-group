"""
nn_model.py
-----------
Simple MLP: strain invariants [I1, I2] -> stress [s11, s22, s12]
Zero-stress at zero-strain enforced by removing first-layer bias.
"""

import torch
import torch.nn as nn


class ConstitutiveNN(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()

        # First layer: no bias so NN(0) = 0
        net = [nn.Linear(2, hidden, bias=False), nn.Softplus()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Softplus()]
        net += [nn.Linear(hidden, 3)]   # output: [s11, s22, s12]

        self.net = nn.Sequential(*net)

    def forward(self, inv):
        """
        inv : (..., 2)  strain invariants [I1, I2]
        returns : (..., 3)  stress [s11, s22, s12]
        """
        return self.net(inv)
