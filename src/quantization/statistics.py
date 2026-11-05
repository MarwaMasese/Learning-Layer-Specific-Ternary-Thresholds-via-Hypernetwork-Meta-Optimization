"""
Per-layer statistics s^(l) = [μ, σ, γ, ρ]  (paper Sec. 3.1)
==========================================================
    μ  mean of the layer's input activations
    σ  std  of the layer's input activations (the paper's "variance";
       Tab. 4 labels its ablation "weight variance", but the statistic
       ablated by `--stats mu,gamma,rho` is this activation std)
    γ  mean |∂L/∂W| of the layer's weights
    ρ  activation sparsity: fraction of |input| < 0.01

μ, σ, ρ come from forward pre-hooks during inner-loop (training) steps;
γ is read from weight gradients after each inner backward. All four are
exponential moving averages, so the hypernetwork sees a smoothed signal.

Raw magnitudes differ by orders of magnitude (γ ~ 1e-5, σ ~ 1), so the
hypernetwork input is  f = [μ, log σ, log γ, ρ]  standardized per feature
across layers. Features can be masked (set to 0 after standardization) for
the statistic ablations without changing the hypernetwork's shape.
"""

from typing import List, Sequence

import torch
import torch.nn as nn

STAT_NAMES = ("mu", "sigma", "gamma", "rho")


class StatsTracker:

    def __init__(
        self,
        layers: Sequence[nn.Module],
        momentum: float = 0.9,
        use: Sequence[str] = STAT_NAMES,
        sparsity_eps: float = 0.01,
    ):
        self.layers: List[nn.Module] = list(layers)
        self.momentum = momentum
        self.sparsity_eps = sparsity_eps
        device = self.layers[0].weight.device
        self.stats = torch.zeros(len(self.layers), 4, device=device)
        self.mask = torch.tensor(
            [1.0 if n in use else 0.0 for n in STAT_NAMES], device=device
        )
        # Python-side flags: avoid a GPU sync per layer per step.
        self._seen_act = [False] * len(self.layers)
        self._seen_grad = [False] * len(self.layers)
        self.active = False
        self._hooks = [
            l.register_forward_pre_hook(self._make_hook(i))
            for i, l in enumerate(self.layers)
        ]

    def _ema(self, i: int, cols, value: torch.Tensor, seen: list) -> None:
        if seen[i]:
            self.stats[i, cols] = (
                self.momentum * self.stats[i, cols]
                + (1 - self.momentum) * value
            )
        else:
            self.stats[i, cols] = value
            seen[i] = True

    def _make_hook(self, i: int):
        def hook(module, inputs):
            if not self.active:
                return
            with torch.no_grad():
                x = inputs[0].detach().flatten()
                if (
                    x.numel() > 1 << 18
                ):  # strided subsample: cheap, unbiased enough
                    x = x[:: x.numel() >> 18]
                x = x.float()
                value = torch.stack(
                    [
                        x.mean(),
                        x.std(),
                        (x.abs() < self.sparsity_eps).float().mean(),
                    ]
                )
                self._ema(i, [0, 1, 3], value, self._seen_act)

        return hook

    @torch.no_grad()
    def update_gradients(self) -> None:
        """Call after an inner-loop backward, before the optimizer step."""
        for i, l in enumerate(self.layers):
            if l.weight.grad is not None:
                self._ema(
                    i,
                    [2],
                    l.weight.grad.detach().abs().mean().float().reshape(1),
                    self._seen_grad,
                )

    @torch.no_grad()
    def features(self) -> torch.Tensor:
        """Normalized hypernetwork inputs, shape (L, 4)."""
        s = self.stats
        f = torch.stack(
            [
                s[:, 0],
                torch.log(s[:, 1].clamp_min(1e-8)),
                torch.log(s[:, 2].clamp_min(1e-12)),
                s[:, 3],
            ],
            dim=1,
        )
        if f.shape[0] > 1:
            f = (f - f.mean(0)) / (f.std(0) + 1e-6)
        return f * self.mask

    def raw(self) -> torch.Tensor:
        return self.stats.detach().clone()

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
