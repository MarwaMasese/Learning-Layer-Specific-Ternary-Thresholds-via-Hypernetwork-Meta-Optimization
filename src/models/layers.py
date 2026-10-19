"""
Ternary layers
==============
`TernaryLinear` / `TernaryConv2d` subclass `nn.Linear` / `nn.Conv2d`, so they
drop into any timm / torchvision backbone and keep the original `forward(x)`
signature and state-dict keys (`weight` is the latent full-precision tensor).

Each layer reads two things set from outside:
    - `qstate`  : the model-wide `QuantState` (mode + temperature)
    - `tau`     : live thresholds [τ+, τ-] in σ units. During the outer loop
                  this tensor is attached to the hypernetwork graph, so the
                  validation loss back-propagates to φ through every layer.
                  When `tau` is None the `thresholds` buffer is used (this is
                  what a deployed / evaluated model holds).
    - `quantizer` (optional): an `LSQTernary` / `TTQTernary` module that
                  replaces the threshold rule entirely.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantization.quantizers import (
    QuantState,
    hard_ternary,
    ls_scale,
    soft_ste_ternary,
    soft_ternary,
    ste_ternary,
)


class _TernaryMixin:
    """Shared quantization logic; requires `self.weight`."""

    def _init_ternary(self) -> None:
        # Last thresholds in σ units, kept for checkpoints, eval, and logging.
        self.register_buffer("thresholds", torch.tensor([0.5, -0.5]))
        self.tau: Optional[torch.Tensor] = None
        self.quantizer: Optional[nn.Module] = None
        # Plain attribute (not a submodule): shared by all layers of a model.
        object.__setattr__(self, "qstate", None)

    def quantized_weight(self) -> torch.Tensor:
        w = self.weight
        st: Optional[QuantState] = self.qstate
        if st is None or st.mode == "fp":
            return w
        if self.quantizer is not None:
            return self.quantizer(w)

        tau = self.tau if self.tau is not None else self.thresholds
        w_n = w / w.detach().std().clamp_min(1e-8)
        if st.mode == "soft":
            q = soft_ternary(w_n, tau[0], tau[1], st.temperature)
        elif st.mode == "ste":
            q = ste_ternary(w_n, tau[0], tau[1])
        elif st.mode == "hard_soft":
            q = soft_ste_ternary(w_n, tau[0], tau[1], st.temperature)
        elif st.mode == "hard":
            q = hard_ternary(w_n, tau[0], tau[1])
        else:
            raise ValueError(f"unknown quant mode {st.mode!r}")
        return ls_scale(w, q) * q

    @torch.no_grad()
    def ternary_fractions(self) -> dict:
        """Fraction of {-1, 0, +1} under the current hard quantizer
        (or the layer's own quantizer module, for LSQ/TTQ/RTN/...)."""
        if self.quantizer is not None:
            q = self.quantizer(self.weight)
        else:
            w_n = self.weight / self.weight.std().clamp_min(1e-8)
            q = hard_ternary(w_n, self.thresholds[0], self.thresholds[1])
        n = q.numel()
        return {
            "-1": (q < 0).sum().item() / n,
            "0": (q == 0).sum().item() / n,
            "+1": (q > 0).sum().item() / n,
        }


class TernaryLinear(_TernaryMixin, nn.Linear):

    @classmethod
    def from_float(cls, m: nn.Linear) -> "TernaryLinear":
        t = cls(
            m.in_features,
            m.out_features,
            bias=m.bias is not None,
            device=m.weight.device,
            dtype=m.weight.dtype,
        )
        t._init_ternary()
        t.load_state_dict(m.state_dict(), strict=False)
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantized_weight(), self.bias)


class TernaryConv2d(_TernaryMixin, nn.Conv2d):

    @classmethod
    def from_float(cls, m: nn.Conv2d) -> "TernaryConv2d":
        t = cls(
            m.in_channels,
            m.out_channels,
            m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=m.bias is not None,
            padding_mode=m.padding_mode,
            device=m.weight.device,
            dtype=m.weight.dtype,
        )
        t._init_ternary()
        t.load_state_dict(m.state_dict(), strict=False)
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(x, self.quantized_weight(), self.bias)


TERNARY_TYPES = (TernaryLinear, TernaryConv2d)
