"""
Backbone zoo + conversion to ternary
====================================
All backbones come from timm with ImageNet-1k pretrained weights, so every
architecture starts from a standard, citable full-precision checkpoint.

Ternarization replaces every nn.Conv2d / nn.Linear with its ternary
counterpart EXCEPT the first layer (stem / patch embedding) and the final
classifier, which stay full precision — the standard convention in the
ternary/low-bit literature (TWN, TTQ, LSQ). Norm layers, biases, position
embeddings and relative-position tables also stay full precision.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from src.models.layers import TERNARY_TYPES, TernaryConv2d, TernaryLinear

# name -> (timm model id, default inner optimizer)
BACKBONES = {
    "resnet18": ("resnet18.tv_in1k", "sgd"),
    "resnet50": ("resnet50.tv_in1k", "sgd"),
    "mobilenetv2": ("mobilenetv2_100.ra_in1k", "sgd"),
    "convnext_t": ("convnext_tiny.fb_in1k", "adamw"),
    "vit_s16": ("deit_small_patch16_224.fb_in1k", "adamw"),
    "swin_t": ("swin_tiny_patch4_window7_224.ms_in1k", "adamw"),
}


class InputResize(nn.Module):
    """Upsample low-resolution inputs (CIFAR 32×32) on the GPU to the
    backbone's pretraining resolution; much cheaper than resizing in the
    data loader."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.size and x.shape[-2] == self.size:
            return x
        return nn.functional.interpolate(
            x,
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False,
        )


def build_model(
    name: str, num_classes: int, img_size: int = 224, pretrained: bool = True
) -> nn.Module:
    import timm

    timm_id, _ = BACKBONES[name]
    net = timm.create_model(
        timm_id, pretrained=pretrained, num_classes=num_classes
    )
    return nn.Sequential(InputResize(img_size), net)


def default_optimizer(name: str) -> str:
    return BACKBONES[name][1]


def _classifier(model: nn.Module) -> nn.Module:
    net = model[1] if isinstance(model, nn.Sequential) else model
    return net.get_classifier()


def ternarize(model: nn.Module) -> List[nn.Module]:
    """Swap Conv2d/Linear for ternary layers in place.

    Returns the ternary layers in forward (module-registration) order, which
    defines the layer ordering used by R_consistency.
    """
    candidates = [
        (n, m)
        for n, m in model.named_modules()
        if type(m) in (nn.Conv2d, nn.Linear)
    ]
    if not candidates:
        raise ValueError("no Conv2d/Linear layers found")
    skip = {id(candidates[0][1]), id(_classifier(model))}

    for name, m in candidates:
        if id(m) in skip:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        new = (
            TernaryConv2d if isinstance(m, nn.Conv2d) else TernaryLinear
        ).from_float(m)
        setattr(parent, child, new)

    return [m for m in model.modules() if isinstance(m, TERNARY_TYPES)]


def ternary_layers(model: nn.Module) -> List[nn.Module]:
    return [m for m in model.modules() if isinstance(m, TERNARY_TYPES)]


def set_qstate(model: nn.Module, qstate) -> None:
    for m in ternary_layers(model):
        object.__setattr__(m, "qstate", qstate)


@torch.no_grad()
def layer_weight_profile(model: nn.Module) -> List[dict]:
    """Per-layer weight-distribution descriptors of the (FP) ternary layers.

    Used for the heterogeneity analysis: how much these vary across layers is
    measured, not assumed. `twn_tau_sigma` is the TWN threshold 0.7·E|W|
    expressed in σ units — a layer-wise "natural" threshold.
    """
    out = []
    for name, m in model.named_modules():
        if not isinstance(m, TERNARY_TYPES):
            continue
        w = m.weight.detach().float().flatten()
        mu, sd = w.mean(), w.std().clamp_min(1e-12)
        z = (w - mu) / sd
        out.append(
            {
                "name": name,
                "type": "conv" if isinstance(m, TernaryConv2d) else "linear",
                "depthwise": bool(
                    isinstance(m, TernaryConv2d) and m.groups > 1
                ),
                "numel": w.numel(),
                "std": sd.item(),
                "mean_over_std": (mu / sd).item(),
                "kurtosis": (z**4).mean().item(),
                "twn_tau_sigma": (0.7 * w.abs().mean() / sd).item(),
            }
        )
    return out


def count_params(model: nn.Module) -> Tuple[int, int]:
    """(total params, params inside ternary layers' weights)."""
    total = sum(p.numel() for p in model.parameters())
    tern = sum(m.weight.numel() for m in ternary_layers(model))
    return total, tern
