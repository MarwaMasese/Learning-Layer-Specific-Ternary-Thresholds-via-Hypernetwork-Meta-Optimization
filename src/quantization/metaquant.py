"""
MetaQuant (Chen, Wang & Pan, NeurIPS 2019) with ternary weights
===============================================================
A shared, coordinate-wise meta network M_φ replaces the straight-through
gradient. With the identity pre-processing A(W) = W (as for BWN, Eq. 2):

    Ŵ = Q(W̃),   g_W = M_φ(g_Ŵ, W̃)                              (Eqs. 3-4)

φ is trained through the "late weight update" (Eqs. 6-8): the weights used at
step t are W̃_t = W_{t-1} − η·M_φ(g_Ŵ^{t-1}, W̃_{t-1}), so

    ∂ℓ_t/∂φ = M(g_Ŵ^t, W̃_t) · ∂W̃_t/∂φ
            = −η · M(g_Ŵ^t, W̃_t) · ∂M_φ(g_Ŵ^{t-1}, W̃_{t-1})/∂φ.

Adaptations for this benchmark (the paper reproduces every baseline under
one W1/A32 protocol, Sec. 4.1):
  * Q is TWN ternarization (Δ = 0.7·E|W|, scale = mean |W| above Δ) instead
    of the binary BWN/DoReFa quantizers of the original paper.
  * Meta network: MultiFC (Eq. 10), M_φ(g, w̃) = g · FCs(w̃), two linear
    layers 1 → 100 → 1 with no non-linearity (Sec. 5.1), PyTorch default
    (random) initialisation.
  * The latent weights are updated by the benchmark's optimizer from g_W, so
    the paper's gradient refinement π(·) is that optimizer; its residual is
    treated as constant in the φ-gradient, which therefore has the SGD form
    above with η = the current learning rate.
  * φ uses Adam with lr = --lr_phi (default 1e-3, the paper's γ).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def twn_quantize(w: torch.Tensor) -> torch.Tensor:
    """TWN: {-α, 0, +α} with Δ = 0.7·E|W| and α = mean |W| over |W| > Δ."""
    delta = 0.7 * w.abs().mean()
    mask = (w.abs() > delta).to(w.dtype)
    alpha = (w.abs() * mask).sum() / mask.sum().clamp_min(1.0)
    return alpha * torch.sign(w) * mask


class MultiFC(nn.Module):
    """M_φ(g, w̃) = g · FC2(FC1(w̃)), 1 → hidden → 1, no non-linearity.

    Without a non-linearity the two layers compose to one affine map
    a·w̃ + c; it is evaluated in that form (identical outputs and gradients
    for every parameter) so no (numel × hidden) tensor is materialised.
    """

    def __init__(self, hidden: int = 100):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        a = (self.fc2.weight @ self.fc1.weight).reshape(())
        c = (self.fc2.weight @ self.fc1.bias + self.fc2.bias).reshape(())
        return g * (a * w + c)


class _MetaQuantFn(torch.autograd.Function):
    """Forward: TWN. Backward: the meta gradient M_φ(g_Ŵ, W̃)."""

    @staticmethod
    def forward(ctx, w, owner):
        ctx.owner = owner
        ctx.save_for_backward(w)
        return twn_quantize(w)

    @staticmethod
    def backward(ctx, g):
        (w,) = ctx.saved_tensors
        return ctx.owner.calibrate(g, w), None


class MetaQuantTernary(nn.Module):
    """Per-layer quantizer; the meta network is shared and owned by the
    method (held as a plain attribute so it is not a parameter of the
    model, and so not updated by the weight optimizer)."""

    def __init__(self, meta: MultiFC):
        super().__init__()
        object.__setattr__(self, "meta", meta)
        # (g_Ŵ, W̃, M) of the current step and (g_Ŵ, W̃) of the previous one
        self.cur: Optional[Tuple[torch.Tensor, ...]] = None
        self.prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and w.requires_grad:
            return _MetaQuantFn.apply(w, self)
        return twn_quantize(w)

    @torch.no_grad()
    def calibrate(self, g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        g, w = g.detach().to(w.dtype), w.detach()
        m = self.meta(g, w)
        self.cur = (g, w, m)
        return m

    @torch.no_grad()
    def thresholds_sigma(self, w: torch.Tensor) -> list:
        t = (0.7 * w.abs().mean() / w.std().clamp_min(1e-8)).item()
        return [t, -t]


def meta_step(
    quantizers, meta: MultiFC, opt: torch.optim.Optimizer, lr: float
) -> Optional[float]:
    """One φ update from the late-update surrogate (module docstring).

    Call once per training step, after the backward pass. Returns the
    surrogate loss, or None on the first step (no previous gradient yet).
    """
    terms = []
    for q in quantizers:
        if q.cur is None:
            continue
        g, w, m = q.cur
        if q.prev is not None:
            g_prev, w_prev = q.prev
            terms.append((m * meta(g_prev, w_prev)).sum())
        q.prev, q.cur = (g, w), None
    if not terms:
        return None
    loss = -lr * torch.stack(terms).sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()
