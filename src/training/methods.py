"""
Threshold methods
=================
Every ternary method is one `Method` object exposing the same hooks, so a
single training loop runs all of them under an identical protocol (same
backbone init, data, schedule, optimizer, epochs, evaluation).

    method          thresholds                      train quantizer   φ update
    -------------   ------------------------------  -----------  ------------
    ours            hypernetwork(stats), per layer  Q_soft*      bilevel
    ours_joint      hypernetwork(stats), per layer  Q_soft*      single-level
    perlayer        free learnable [τ+,τ-] / layer  Q_soft*      single-level
    fixed           one global ±τ for all layers    Q_soft            —
    twn             0.7·E|W| per layer (symmetric)  STE               —
    lsq             learned step s, 3 levels        STE (LSQ)         —
    ttq             0.05·max|W|, learned W_p / W_n  STE (TTQ)         —
    rtn             learned k·W+b per filter, ±0.5  clipped STE       —
    fatnn           learned −α1/2, +α2/2 per layer  STE (round)       —
    metaquant       TWN rule                        meta gradient  M_φ, late
                                                                   update

`ours`/`fixed`/`perlayer` share the quantizer, scale rule and annealing, so
ours − fixed and ours − perlayer isolate the threshold policy itself.
Every reported number is evaluated with exact hard ternary weights.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from src.quantization.hypernetwork import Hypernetwork, thresholds_from_raw
from src.quantization.metaquant import MetaQuantTernary, MultiFC, meta_step
from src.quantization.quantizers import (
    FATNNTernary,
    LSQTernary,
    RTNTernary,
    TTQTernary,
)
from src.quantization.statistics import StatsTracker

METHODS = (
    "ours",
    "ours_joint",
    "perlayer",
    "fixed",
    "twn",
    "lsq",
    "ttq",
    "rtn",
    "fatnn",
    "metaquant",
)


def threshold_regularizers(tau: torch.Tensor):
    """R_smooth = Σ_l ||τ_l||², R_consistency = Σ_l ||τ_{l+1} − τ_l||².

    tau has shape (L, 2), layers in forward order.
    """
    r_smooth = (tau**2).sum()
    r_cons = (
        ((tau[1:] - tau[:-1]) ** 2).sum()
        if tau.shape[0] > 1
        else tau.new_zeros(())
    )
    return r_smooth, r_cons


class Method(nn.Module):
    """Default: no learned thresholds, nothing to regularize."""

    bilevel = False  # φ updated on held-out batches every K inner steps
    joint = False  # φ updated together with θ on the training loss
    train_mode = "soft"  # QuantState.mode used during training

    def __init__(self, layers: List[nn.Module], args):
        super().__init__()
        object.__setattr__(self, "layers", list(layers))  # not submodules
        self.args = args

    # -- hooks used by the training loop ----------------------------------
    def phi_params(self) -> List[nn.Parameter]:
        return []

    def set_thresholds(self, connected: bool) -> Optional[torch.Tensor]:
        """Install thresholds on every layer before a forward pass.

        Returns the (L, 2) threshold tensor when it carries gradient to φ."""
        return None

    def regularizer(self, tau: Optional[torch.Tensor]) -> torch.Tensor:
        if tau is None:
            return torch.zeros((), device=self.layers[0].weight.device)
        r_s, r_c = threshold_regularizers(tau)
        return self.args.lambda1 * r_s + self.args.lambda2 * r_c

    def on_train_forward(self, active: bool) -> None:
        pass

    def after_inner_backward(self) -> None:
        pass

    @torch.no_grad()
    def finalize(self) -> None:
        """Freeze current thresholds into each layer's `thresholds` buffer
        (what a deployed model holds) and drop live tensors."""
        tau = self.set_thresholds(connected=False)
        for i, layer in enumerate(self.layers):
            if tau is not None:
                layer.thresholds.copy_(tau[i].detach())
            elif layer.tau is not None:
                layer.thresholds.copy_(layer.tau.detach())
            layer.tau = None

    def extra_state(self) -> dict:
        return {}

    def load_extra_state(self, state: dict) -> None:
        pass

    def threshold_table(self) -> List[List[float]]:
        return [layer.thresholds.tolist() for layer in self.layers]


# ---------------------------------------------------------------------------


class Fixed(Method):
    """One global symmetric threshold ±τ (σ units) for every layer."""

    def __init__(self, layers, args):
        super().__init__(layers, args)
        self.train_mode = args.train_mode
        for layer in self.layers:
            t = args.fixed_tau
            layer.thresholds.copy_(torch.tensor([t, -t]))


class TWN(Method):
    """Li et al. 2016: τ_l = 0.7·E|W_l| per layer, hard ternary + STE."""

    train_mode = "ste"

    @torch.no_grad()
    def set_thresholds(self, connected):
        for layer in self.layers:
            w = layer.weight
            t = 0.7 * w.abs().mean() / w.std().clamp_min(1e-8)
            layer.thresholds.copy_(torch.stack([t, -t]))
        return None


class _ModuleQuantizer(Method):
    """Baselines that replace the threshold rule with their own module."""

    train_mode = "ste"
    factory = None

    def __init__(self, layers, args):
        super().__init__(layers, args)
        for layer in self.layers:
            layer.quantizer = type(self).factory(layer.weight)

    def finalize(self):
        pass

    def threshold_table(self):
        """Effective [τ+, τ-] per layer in σ units (comparable to ours)."""
        return [
            layer.quantizer.thresholds_sigma(layer.weight)
            for layer in self.layers
        ]


class LSQ(_ModuleQuantizer):
    factory = LSQTernary


class TTQ(_ModuleQuantizer):
    factory = TTQTernary


class RTN(_ModuleQuantizer):
    """Li et al. 2020, weight quantizer (see RTNTernary)."""

    factory = RTNTernary


class FATNN(_ModuleQuantizer):
    """Chen et al. 2021, weight quantizer (see FATNNTernary)."""

    factory = FATNNTernary


class MetaQuant(_ModuleQuantizer):
    """Chen et al. 2019: TWN forward, meta-learned gradient (metaquant.py).

    The shared meta network has its own Adam optimizer and is stepped here,
    once per training step, so it is not part of φ for the bilevel loop.
    """

    def __init__(self, layers, args):
        Method.__init__(self, layers, args)
        self.meta = MultiFC(hidden=100)
        self._meta_opt = None  # built lazily, after .to(device)
        self.current_lr = args.lr
        for layer in self.layers:
            layer.quantizer = MetaQuantTernary(self.meta)

    @property
    def meta_opt(self):
        if self._meta_opt is None:
            self._meta_opt = torch.optim.Adam(
                self.meta.parameters(), lr=self.args.lr_phi
            )
        return self._meta_opt

    def after_inner_backward(self):
        meta_step(
            [layer.quantizer for layer in self.layers],
            self.meta,
            self.meta_opt,
            self.current_lr,
        )

    def extra_state(self):
        return {"meta_opt": self.meta_opt.state_dict()}

    def load_extra_state(self, state):
        if state.get("meta_opt"):
            self.meta_opt.load_state_dict(state["meta_opt"])


class PerLayer(Method):
    """Free per-layer [τ+, τ-] trained jointly with θ (no hypernetwork).
    Same sign-constrained sigmoid parameterization as the hypernetwork."""

    joint = True

    def __init__(self, layers, args):
        super().__init__(layers, args)
        self.train_mode = args.train_mode
        self.raw = nn.Parameter(
            torch.zeros(len(self.layers), 2)
        )  # sigmoid(0) = 0.5

    def phi_params(self):
        return [self.raw]

    def set_thresholds(self, connected):
        tau = thresholds_from_raw(self.raw)
        if not connected:
            tau = tau.detach()
        for i, layer in enumerate(self.layers):
            layer.tau = tau[i]
        return tau


class HyperNet(Method):
    """Ours: shared hypernetwork maps per-layer statistics to thresholds."""

    bilevel = True

    def __init__(self, layers, args):
        super().__init__(layers, args)
        self.hn = Hypernetwork(
            hidden_dim=args.hn_width,
            depth=args.hn_depth,
            symmetric=args.symmetric,
        )
        object.__setattr__(
            self,
            "tracker",
            StatsTracker(self.layers, use=args.stats.split(",")),
        )
        self.train_mode = args.train_mode

    def phi_params(self):
        return list(self.hn.parameters())

    def set_thresholds(self, connected):
        feats = self.tracker.features().to(next(self.hn.parameters()).device)
        if connected:
            tau = self.hn(feats)
        else:
            with torch.no_grad():
                tau = self.hn(feats)
        for i, layer in enumerate(self.layers):
            layer.tau = tau[i]
        return tau

    def on_train_forward(self, active):
        self.tracker.active = active

    def after_inner_backward(self):
        self.tracker.update_gradients()

    def extra_state(self):
        return {
            "stats": self.tracker.stats.clone(),
            "seen": (self.tracker._seen_act, self.tracker._seen_grad),
        }

    def load_extra_state(self, state):
        self.tracker.stats.copy_(state["stats"])
        self.tracker._seen_act, self.tracker._seen_grad = map(
            list, state["seen"]
        )


class HyperNetJoint(HyperNet):
    """Ablation: same hypernetwork, but φ trained jointly with θ on the
    training loss (single-level, no held-out split)."""

    bilevel = False
    joint = True


def build_method(name: str, layers, args) -> Method:
    return {
        "ours": HyperNet,
        "ours_joint": HyperNetJoint,
        "perlayer": PerLayer,
        "fixed": Fixed,
        "twn": TWN,
        "lsq": LSQ,
        "ttq": TTQ,
        "rtn": RTN,
        "fatnn": FATNN,
        "metaquant": MetaQuant,
    }[name](layers, args)
