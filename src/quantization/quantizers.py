"""
Ternary quantizers
==================
Everything that maps a latent full-precision weight tensor W to its ternary
version lives here.

Threshold-based methods (ours, fixed, per-layer, TWN) share one rule:

    W_n   = W / std(W)                          (thresholds live in σ units)
    q     = Q(W_n; τ+, τ-)    ∈ [-1, 1]         (soft, hard, or STE)
    W_q   = α · q,   α = <W, q> / <q, q>        (least-squares per-layer scale)

With hard q, α is exactly the TWN scale (mean |W| over non-zero entries), so
the deployed model is {-α, 0, +α} per layer. With soft q at high temperature,
α·q ≈ W, so annealing moves smoothly from full precision to ternary.

Normalising by std(W) puts thresholds and temperature on the same scale for
every layer; without it, τ ∈ [-1, 1] and T ≥ 0.05 are far larger than typical
weights (std ≈ 0.02) and the soft quantizer never approaches the hard one.

LSQ, TTQ, RTN and FATNN have their own quantization rules and are
implemented as per-layer modules (`LSQTernary`, `TTQTernary`, `RTNTernary`,
`FATNNTernary`); MetaQuant lives in `metaquant.py`. Each module reports its
effective thresholds in σ units (`thresholds_sigma`), so threshold plots
compare every method on one axis.
"""

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Elementwise ternary maps on normalised weights
# ---------------------------------------------------------------------------


def soft_ternary(
    w_n: torch.Tensor, tau_plus, tau_minus, T: float
) -> torch.Tensor:
    """Q_soft (paper Eq. 4): σ((w-τ-)/T) + σ((w-τ+)/T) - 1.

    → +1 for w ≫ τ+, 0 for τ- ≪ w ≪ τ+, -1 for w ≪ τ-, and recovers
    `hard_ternary` as T → 0. Differentiable in w and in both thresholds.
    """
    return (
        torch.sigmoid((w_n - tau_minus) / T)
        + torch.sigmoid((w_n - tau_plus) / T)
        - 1.0
    )


def hard_ternary(w_n: torch.Tensor, tau_plus, tau_minus) -> torch.Tensor:
    """Q_hard: +1 if w > τ+, -1 if w < τ-, else 0 (Q_soft as T → 0)."""
    return (w_n > tau_plus).to(w_n.dtype) - (w_n < tau_minus).to(w_n.dtype)


def ste_ternary(
    w_n: torch.Tensor, tau_plus, tau_minus, tau_grad: bool = False
) -> torch.Tensor:
    """Hard ternary forward, straight-through (identity) backward.

    The weights always get gradient 1. ``tau_grad`` selects what the
    thresholds get, and the two settings are not equivalent:

    ``tau_grad=False`` (default, and what the published tables were run
        with): the thresholds get no gradient at all. For the Tab. 4 "STE
        instead of Q_soft" row this means the hypernetwork receives no
        meta-signal whatsoever and stays at its initialisation, so that row
        measures a *frozen* hypernetwork rather than STE as such.

    ``tau_grad=True`` (``--ste-tau-grad``): q is differentiated as if it
        were w − (τ+ + τ−)/2, so each threshold gets the constant −1/2 and
        the hypernetwork receives the constant-gradient meta-signal that
        Sec. 3.3 actually describes. Note the resulting threshold gradient
        is −0.5·Σ(incoming gradient) over the whole weight tensor: it scales
        with layer size and carries no information about which weights lie
        near a threshold, which is precisely why it is a poor meta-signal.
        This is the more faithful reading of the paper's text, but it does
        not reproduce the published Tab. 4 value.

    For TWN the thresholds are plain buffers, so only w is ever affected.
    """
    q = hard_ternary(
        w_n.detach(),
        tau_plus.detach() if torch.is_tensor(tau_plus) else tau_plus,
        tau_minus.detach() if torch.is_tensor(tau_minus) else tau_minus,
    )
    # exact q forward, identity backward
    g = w_n - 0.5 * (tau_plus + tau_minus) if tau_grad else w_n
    return q + (g - g.detach())


def soft_ste_ternary(
    w_n: torch.Tensor, tau_plus, tau_minus, T: float
) -> torch.Tensor:
    """Exact hard ternary forward, Q_soft gradient backward (to w and both τ).

    Training then sees exactly the weights used at evaluation, while the
    soft relaxation still supplies the gradient path to the thresholds (and
    hence to the hypernetwork).
    """
    s = soft_ternary(w_n, tau_plus, tau_minus, T)
    h = hard_ternary(
        w_n.detach(),
        tau_plus.detach() if torch.is_tensor(tau_plus) else tau_plus,
        tau_minus.detach() if torch.is_tensor(tau_minus) else tau_minus,
    )
    return h + (s - s.detach())


def ls_scale(w: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Least-squares scale α = <w, q> / <q, q>: α·q best approximates w."""
    return (w * q).sum() / (q * q).sum().clamp_min(1e-12)


# ---------------------------------------------------------------------------
# Shared quantizer state (temperature + mode), read by every ternary layer
# ---------------------------------------------------------------------------


class QuantState:
    """Mutable config shared by all ternary layers of one model.

    mode:
        "fp"   – no quantization (latent weights used directly)
        "soft" – Q_soft at the current temperature: the training quantizer
                 of the inner loop (Eq. 3). Matches the paper's text, but
                 trains on smooth weights while we report hard ternary ones
        "ste"  – hard forward, straight-through backward (TWN, STE ablation)
        "hard_soft" – hard forward, Q_soft backward; train and eval weights
                 coincide. This is the setting the published tables used
        "hard" – exact ternary; used for every evaluation number we report

    ``ste_tau_grad`` picks which STE the "ste" mode uses; see
    :func:`ste_ternary`.
    """

    def __init__(
        self,
        T0: float = 5.0,
        T_min: float = 0.05,
        schedule: str = "exp",
        mode: str = "soft",
        ste_tau_grad: bool = False,
    ):
        assert schedule in ("exp", "linear", "const")
        self.T0, self.T_min, self.schedule = T0, T_min, schedule
        self.mode = mode
        self.ste_tau_grad = ste_tau_grad
        self.temperature = T_min if schedule == "const" else T0

    def update_temperature(self, epoch: int, total_epochs: int) -> float:
        """T(p) for training progress p ∈ [0, 1], floored at T_min.

        exp:    T0 · exp(-5p)           (paper Sec. 3.3 default; with
                                         T0 = 5 the T_min = 0.05 floor is
                                         reached at p = ln(100)/5 ≈ 0.92)
        linear: T0 + (T_min - T0) · p   ("linear decay" ablation, Tab. 4)
        const:  T_min throughout        ("no annealing" ablation, Tab. 4)
        """
        p = epoch / max(total_epochs - 1, 1)
        if self.schedule == "exp":
            T = self.T0 * math.exp(-5.0 * p)
        elif self.schedule == "linear":
            T = self.T0 + (self.T_min - self.T0) * p
        else:
            T = self.T_min
        self.temperature = max(self.T_min, T)
        return self.temperature


# ---------------------------------------------------------------------------
# Baselines with their own quantization rule
# ---------------------------------------------------------------------------


def _grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Forward x, backward grad·scale (LSQ's step-size gradient scaling)."""
    return (x - x * scale).detach() + x * scale


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return x.round().detach() + (x - x.detach())


class LSQTernary(nn.Module):
    """LSQ (Esser et al., ICLR 2020) restricted to 3 levels {-s, 0, +s}.

    W_q = round(clamp(W / s, -1, 1)) · s with a learned per-layer step s,
    initialised to 2·mean|W| / sqrt(Q_P) and gradient-scaled by
    1 / sqrt(N · Q_P), with Q_P = 1 for ternary.
    """

    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.step = nn.Parameter(
            2.0 * w.detach().abs().mean().clamp_min(1e-8).reshape(())
        )
        self.g = 1.0 / math.sqrt(w.numel() * 1.0)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        s = _grad_scale(self.step, self.g)
        return _round_ste(torch.clamp(w / s, -1.0, 1.0)) * s

    @torch.no_grad()
    def thresholds_sigma(self, w: torch.Tensor) -> list:
        """[τ+, τ-] in σ units: a weight is non-zero iff |w| ≥ s/2."""
        t = (self.step.abs() / 2 / w.std().clamp_min(1e-8)).item()
        return [t, -t]


class TTQTernary(nn.Module):
    """TTQ (Zhu et al., ICLR 2017): learned asymmetric scales {-W_n, 0, +W_p}.

    Threshold Δ = t · max|W| (t = 0.05). Gradients follow the paper:
    ∂L/∂W_p = Σ_{pos} ∂L/∂W_q,  ∂L/∂W_n = Σ_{neg} ∂L/∂W_q, and the latent
    gradient is scaled by W_p / 1 / W_n in the positive / zero / negative
    region.
    """

    def __init__(self, w: torch.Tensor, t: float = 0.05):
        super().__init__()
        self.t = t
        w = w.detach()
        delta = t * w.abs().max()
        pos, neg = w[w > delta], w[w < -delta]
        self.w_p = nn.Parameter(
            (pos.mean() if pos.numel() else w.abs().mean()).reshape(())
        )
        self.w_n = nn.Parameter(
            (neg.abs().mean() if neg.numel() else w.abs().mean()).reshape(())
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        delta = self.t * w.detach().abs().max()
        pos = (w.detach() > delta).to(w.dtype)
        neg = (w.detach() < -delta).to(w.dtype)
        q = self.w_p * pos - self.w_n * neg
        mult = (
            self.w_p.detach() * pos
            + self.w_n.detach() * neg
            + (1.0 - pos - neg)
        )
        latent = w * mult
        return q + (
            latent - latent.detach()
        )  # exact q forward, scaled latent grad

    @torch.no_grad()
    def thresholds_sigma(self, w: torch.Tensor) -> list:
        """[τ+, τ-] in σ units: Δ = t·max|W|."""
        t = (self.t * w.abs().max() / w.std().clamp_min(1e-8)).item()
        return [t, -t]


class RTNTernary(nn.Module):
    """RTN (Li et al., AAAI 2020), weight quantizer only (Sec. 3.1–3.2).

    Eq. 5: per output filter, a learnable affine map W̄ = k·W + b sets the
    quantization threshold; W̄ is ternarized by the fixed rule of Eq. 1
    (sign(W̄) if |W̄| > 0.5, else 0) with the clipped STE
    ∂Wt/∂W̄ = 1{|W̄| ≤ 1}, and rescaled by a learnable scale α
    (W̄t = α·Wt; RTN uses no offset for weights). We use one α per layer.

    Initialisation (not specified in the paper): k = 0.5 / (0.7·E|W|) per
    filter and b = 0, so the starting thresholds are TWN's, and α is the TWN
    scale. k and b are stored as 1-d tensors so the optimizer applies no
    weight decay to them.
    """

    def __init__(self, w: torch.Tensor):
        super().__init__()
        w = w.detach()
        m = w.abs().flatten(1).mean(1).clamp_min(1e-8)  # (out,)
        self.k = nn.Parameter(0.5 / (0.7 * m))
        self.b = nn.Parameter(torch.zeros_like(m))
        nz = (w.abs() > 0.7 * m.view(self._shape(w))).to(w.dtype)
        alpha = (w.abs() * nz).sum() / nz.sum().clamp_min(1.0)
        self.alpha = nn.Parameter(alpha.reshape(()))

    @staticmethod
    def _shape(w: torch.Tensor) -> tuple:
        return (w.shape[0],) + (1,) * (w.dim() - 1)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        shape = self._shape(w)
        wb = self.k.view(shape) * w + self.b.view(shape)
        hard = torch.sign(wb) * (wb.abs() > 0.5).to(wb.dtype)
        clipped = wb.clamp(-1.0, 1.0)  # gradient 1 on |W̄| ≤ 1, else 0
        return self.alpha * (hard.detach() + (clipped - clipped.detach()))

    @torch.no_grad()
    def thresholds_sigma(self, w: torch.Tensor) -> list:
        """Mean over filters of the effective [τ+, τ-] in σ units of W:
        k·w + b > 0.5  ⇔  w > (0.5 − b) / k   (for k > 0)."""
        sd = w.std().clamp_min(1e-8)
        k = self.k.abs().clamp_min(1e-8)
        tp = ((0.5 - self.b) / k).mean() / sd
        tm = ((-0.5 - self.b) / k).mean() / sd
        return [tp.item(), tm.item()]


class FATNNTernary(nn.Module):
    """FATNN (Chen et al., ICCV 2021), weight quantizer only (Eqs. 8–9).

    With standardized weights ŵ = (W − mean) / std ("weight normalization"
    in the paper; per layer here):
        q = ⌊clip(ŵ/α1, −1, 0)⌉ + ⌊clip(ŵ/α2, 0, 1)⌉  ∈ {−1, 0, +1},
    i.e. thresholds −α1/2 and +α2/2. α1, α2 are learned per layer,
    initialised to 1.0, with the STE through rounding only (no LSQ gradient
    scaling, as in the paper). ⌊·⌉ is round-half-to-even (torch.round),
    matching the paper's banker's rounding. The single high-precision
    coefficient is the least-squares scale used by the other threshold
    methods.
    """

    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.a1 = nn.Parameter(torch.ones((), device=w.device))
        self.a2 = nn.Parameter(torch.ones((), device=w.device))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        w_n = (w - w.mean()) / w.std().clamp_min(1e-8)
        a1 = self.a1.abs().clamp_min(1e-3)
        a2 = self.a2.abs().clamp_min(1e-3)
        q = _round_ste(torch.clamp(w_n / a1, -1.0, 0.0)) + _round_ste(
            torch.clamp(w_n / a2, 0.0, 1.0)
        )
        return ls_scale(w, q) * q

    @torch.no_grad()
    def thresholds_sigma(self, w: torch.Tensor) -> list:
        """[τ+, τ-] = [α2/2, −α1/2] (in σ units, relative to the mean)."""
        return [self.a2.abs().item() / 2, -self.a1.abs().item() / 2]
