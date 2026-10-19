"""
Threshold hypernetwork h(s; φ)
==============================
A small MLP shared across all layers: it maps one layer's 4-d statistic
descriptor s^(l) to that layer's threshold pair [τ+, τ-] (in σ units of the
layer's weights; see quantizers.py).

Default 4 → 64 → 64 → 2 with ReLU hidden layers: 4,610 parameters.
Width / depth are configurable for the capacity ablations.

Output parameterization: τ+ = sigmoid(a) ∈ (0, 1),  τ- = −sigmoid(b) ∈ (−1, 0).
The sign constraint matches the definition of τ+ / τ- as the positive /
negative ternary thresholds, so every layer always has a positive, a zero
and a negative region.
"""

import math

import torch
import torch.nn as nn


def thresholds_from_raw(
    raw: torch.Tensor, symmetric: bool = False
) -> torch.Tensor:
    """raw (..., 2) → [τ+, τ-] with 0 < τ+ < 1 and −1 < τ- < 0."""
    if symmetric:
        t = torch.sigmoid(raw.mean(-1))
        return torch.stack([t, -t], dim=-1)
    return torch.stack(
        [torch.sigmoid(raw[..., 0]), -torch.sigmoid(raw[..., 1])], dim=-1
    )


class Hypernetwork(nn.Module):

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        depth: int = 2,
        symmetric: bool = False,
        init_tau: float = 0.5,
    ):
        super().__init__()
        self.symmetric = symmetric
        layers, d = [], input_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        self.out = nn.Linear(d, 2)
        self.body = nn.Sequential(*layers)

        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)
        # Start every layer at ±init_tau (the fixed baseline's default), with a
        # small output weight so statistics can move thresholds away from it.
        nn.init.normal_(self.out.weight, std=1e-3)
        logit = math.log(init_tau / (1 - init_tau))
        with torch.no_grad():
            self.out.bias.fill_(logit)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (L, 4) → thresholds (L, 2) as [τ+, τ-]."""
        return thresholds_from_raw(self.out(self.body(s)), self.symmetric)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
