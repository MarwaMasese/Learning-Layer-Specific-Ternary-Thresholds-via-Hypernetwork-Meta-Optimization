# Paper ↔ code alignment (WACV 2027 camera-ready)

The camera-ready paper is the specification for this repository. This file
maps each part of the paper to the code that implements it. It also lists
the few paper statements that no code change can reproduce.

Section, table and equation numbers refer to the camera-ready PDF.

## 1. Method (Sec. 3)

| Paper | Code |
|---|---|
| Descriptor s⁽ˡ⁾ = [μ, σ, γ, ρ]: activation mean, variance, gradient magnitude, sparsity (Sec. 3.1) | `src/quantization/statistics.py`. The hypernetwork input is [μ, log σ, log γ, ρ], standardised across layers. Log-std and log-variance differ only by a factor of 2, and standardisation removes that factor, so the features are identical either way. |
| Hypernetwork 4→64→64→2, ReLU, sigmoid output, 4,610 params, shared across layers | `src/quantization/hypernetwork.py` (320 + 4,160 + 130 = 4,610) |
| τ⁺ = σ(a) ∈ (0,1), τ⁻ = −σ(b) ∈ (−1,0), in units of the layer's weight std | `thresholds_from_raw`; `w / std(w)` in `src/models/layers.py` |
| Outer objective L_val + λ₁R_smooth + λ₂R_consistency (Eq. 2); R_consistency over adjacent layers | `threshold_regularizers`. Layer order is the forward execution order, verified on all six backbones. |
| Inner objective L_train(Q_soft(θ, τ, T)) (Eq. 3) | `train_mode: "soft"` in `configs/*.json` (also the CLI default) |
| Disjoint inner/outer data; held-out reserved for policy updates (Sec. 3.2) | `src/utils/data.py`. The outer forward pass runs under `frozen_bn_stats`, so BatchNorm running statistics never see held-out data. No checkpoint is selected on held-out accuracy. |
| Q_soft (Eq. 4) | `soft_ternary` |
| T(p) = T₀·exp(−5p), T₀ = 5, floor T_min = 0.05 | `QuantState` (`exp`). The Tab. 4 ablations use `linear` and `const`. |
| K = 5 inner steps per outer update; first-order meta-gradient that omits second-order terms (Secs. 2.4, 3.3) | `train_ternary` in `src/training/loop.py` |
| STE "treating the hard step as an identity in the backward pass" (Sec. 2.2), Tab. 4 ablation | `ste_ternary`: identity gradient to w and a constant gradient (−½) to each threshold, so φ still receives the STE's constant-gradient meta-signal |

## 2. Experimental protocol (Sec. 4)

| Paper | Code |
|---|---|
| Baselines TWN, TTQ, RTN, MetaQuant, LSQ, FATNN under W1/A32 | `src/training/methods.py`, `src/quantization/quantizers.py`, `src/quantization/metaquant.py` |
| SGD-Nesterov 1e−2 / WD 5e−4 (ResNet, MobileNetV2); AdamW 1e−4 / WD 5e−2 (ConvNeXt, ViT, Swin); no WD on biases/norms; cosine + 3 % warmup; clip 1.0; bf16 | `run.py`, `make_optimizer`, `cosine_with_warmup` |
| CIFAR: 15 FP epochs, then 30 ternary epochs, batch size 128. ImageNet: pretrained init, then 10 epochs, batch size 256 | `configs/*.json` |
| Augmentation as in Sec. 4.1; no distillation | `src/utils/data.py` |
| φ: Adam 1e−3, no WD, no schedule; λ₁ = 1e−4, λ₂ = 1e−3 | configs, `train_ternary` |
| Held-out: 10 % of CIFAR train, 10 images per class for ImageNet | `src/utils/data.py` |
| CIFAR: mean over 3 seeds; ImageNet: single seed | `scripts/make_jobs.py` |
| Tab. 1: ResNet-18 on CIFAR-10/100 | `make_jobs.py main` (ResNet-18, the six methods, ours) |
| Tabs. 1–3: ImageNet, six backbones | `make_jobs.py imagenet` |
| Tabs. 4–5: CIFAR-10 / ResNet-18 ablations and sensitivity | `make_jobs.py ablation` |
| Tabs. 1–5 | `scripts/aggregate.py --latex` → `paper_table{1..5}_*.tex` (paper layout) |
| Figs. 2–3 | `scripts/plot_paper_figures.py` |
| Params and FLOPs (Tab. 3) | `results.json` → `params`, `macs` (`count_macs`) |

Nothing is tuned on held-out or test data. The repo previously had a
held-out-based selection phase and a held-out-tuned `fixed` baseline;
neither is in the paper, so both were removed. `fixed` (τ = 0.5, no
hypernetwork) remains only as the "without the hypernetwork" reference for
the Tab. 3 overhead.

## 3. Paper statements the code cannot reproduce

These need a correction in the paper (erratum or arXiv version). Changing
the code cannot fix them.

1. **Memory O(K·|θ|) and "backpropagate across K unrolled inner steps"
   (Secs. 2.2, 3.3, 4.3, 5).** The paper also says the meta-gradient is
   first-order with second-order terms omitted (Secs. 2.4, 3.3). Under that
   approximation no gradient passes through the inner updates, so memory
   is O(|θ|) and the cost per epoch falls as K *grows*. Likewise, the
   abstract's "reducible to approximately half with fewer inner-loop steps"
   holds here only in the sense of fewer outer updates per inner step.
2. **"w/o Weight variance (σ⁽ˡ⁾)" (Tab. 4, Fig. 3, Sec. 4.3).** Sec. 3.1
   defines σ⁽ˡ⁾ as an *activation* statistic, and the code follows Sec. 3.1.
   The ablation therefore removes the activation variance.
3. **HN overhead column (Tab. 3).** 4,610 ÷ backbone params is 0.039 % for
   ResNet-18, 0.018 % for ResNet-50, 0.132 % for MobileNetV2, 0.016 % for
   ConvNeXt, 0.021 % for ViT-S/16 and 0.016 % for Swin-T. The paper prints
   values about 2× higher (0.08 % … 0.27 %). The "< 0.3 %" claim still holds.
4. **"Mem. Cost = ratio of total model memory with and without the
   hypernetwork" (Tab. 3).** Stored-model memory grows by at most 0.13 %
   (MobileNetV2). The 1.06–1.10× range can only be peak *training* memory,
   which is what the code measures (ours vs `fixed`).
5. **ViT-S/16 FLOPs.** The measured value is 4.24 G MACs; the paper gives
   4.6 G, the figure DeiT reports. The two use different counting
   conventions for attention operations. All other backbones match.
6. **ViT-S/16 weights.** The code uses DeiT-S ImageNet-1k weights
   (`deit_small_patch16_224.fb_in1k`). The paper does not say so.

## 4. Release items the paper promises

* **"Code and pretrained weights are available" (Sec. 4.1).** No weights
  are in the repo yet. Attach `checkpoints/last.pt` and `results.json` of
  each reported run to a GitHub Release or the Hugging Face Hub.
* **LICENSE.** There is none yet. The authors must choose one.
* **The paper PDF** is not tracked, because of the publisher's copyright.
  Link the CVF open-access page or arXiv from the README once either exists.
