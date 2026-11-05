# Paper ↔ code alignment (WACV 2027 camera-ready)

This document compares the camera-ready paper with this repository, item by
item. It records what already matches, what this branch changes, and which
gaps are left for the authors to decide. Some of those gaps are best fixed
in the code. Others are best fixed in the paper text or an erratum.

Section, table and equation numbers refer to the camera-ready PDF.

## 1. What already matches

| Paper | Code | Status |
|---|---|---|
| Hypernetwork 4→64→64→2, ReLU, sigmoid output, **4,610 params**, shared across layers (Sec. 3.1) | `src/quantization/hypernetwork.py` (verified: 320 + 4,160 + 130 = 4,610) | ✅ |
| τ⁺ = σ(a) ∈ (0,1), τ⁻ = −σ(b) ∈ (−1,0), in units of the layer's weight std (Sec. 3.1) | `thresholds_from_raw`, `w / std(w)` in `layers.py` | ✅ |
| Q_soft = σ((θ−τ⁻)/T) + σ((θ−τ⁺)/T) − 1 (Eq. 4) | `soft_ternary` | ✅ |
| T(p) = T₀·exp(−5p), T₀ = 5, floor T_min = 0.05 (Sec. 3.3) | `QuantState.update_temperature`, `configs/*.json` | ✅ |
| R_smooth = Σ‖τ⁽ˡ⁾‖², R_consistency = Σ‖τ⁽ˡ⁺¹⁾ − τ⁽ˡ⁾‖², λ₁ = 1e−4, λ₂ = 1e−3 (Eq. 2, Sec. 4.1) | `threshold_regularizers`, configs | ✅ |
| K = 5, φ: Adam lr 1e−3, no WD, no schedule (Sec. 4.1) | `train_ternary`, configs | ✅ |
| SGD-Nesterov 1e−2 / WD 5e−4 (ResNet, MobileNetV2); AdamW 1e−4 / WD 5e−2 (ConvNeXt, ViT, Swin); no WD on biases/norms; cosine + 3 % warmup; clip 1.0; bf16 | `run.py`, `make_optimizer`, `cosine_with_warmup` | ✅ |
| CIFAR: 15 FP epochs, then 30 ternary epochs, batch size 128; ImageNet: pretrained init, then 10 ternary epochs, batch size 256 | `DATASET_DEFAULTS`, configs | ✅ |
| Held-out split: 10 % of CIFAR train, 10 images per class for ImageNet, disjoint from the train split | `src/utils/data.py` | ✅ |
| Augmentation (crop 32 pad 4 + flip; RRC-224 + flip; resize 256 / center crop 224 at eval) | `src/utils/data.py` | ✅ |
| Stem and classifier kept full precision; W1/A32 for every method | `ternarize` | ✅ |
| Baselines TWN, TTQ, RTN, MetaQuant, LSQ, FATNN, all under the same protocol | `methods.py`, `quantizers.py`, `metaquant.py` | ✅ |
| Ablation rows in Tab. 4 and sensitivity rows in Tab. 5 | `make_jobs.py ablation` (each row has a tag) | ✅ |
| Backbone params in Tab. 3 (11.7M / 25.6M / 3.5M / 28.6M / 22.1M / 28.3M) | timm models, verified | ✅ |
| Backbone FLOPs in Tab. 3 (1.8 / 4.1 / 0.3 / 4.5 / 4.6 / 4.5 G) | measured MACs: 1.81 / 4.09 / 0.30 / 4.46 / **4.24** / 4.49 G | ✅, except ViT-S/16 (see 3.6) |
| CIFAR: mean of 3 seeds; ImageNet: 1 seed | `make_jobs.py` | ✅ |

## 2. What this branch changes

1. **Paper-layout LaTeX tables.** `aggregate.py --latex <dir>` now also
   writes `paper_table1_resnet18.tex` … `paper_table5_sensitivity.tex`.
   They use the camera-ready rows, columns, *Pub. W/A* column, *vs. LSQ*
   row, ablation groups and λ₁/λ₂/K columns, so each table can be compared
   with the PDF directly. Cells show the mean to one decimal, as in the
   paper; `--paper_std` adds ±std. The existing working tables are unchanged.
2. **FLOPs in `results.json`.** `run.py` now records `macs`, counted with
   `torch.utils.flop_counter` (`count_macs` in `src/models/backbones.py`).
   Tab. 3 can then be regenerated with nothing typed by hand.
3. **Docstrings now match the paper.** Q_soft is Eq. 4. The old text said
   "Eq. 9", and the hard quantizer had no equation number in the paper.
   Two notes were added: the meta-gradient is first-order, and σ is the
   *activation* std (see 3.1–3.2).
4. **README.** The reproduction table now maps to Tabs. 1–5. It states that
   the paper's CIFAR results are ResNet-18 only, describes the training
   quantizer and the use of held-out data precisely, and adds citation
   information (`CITATION.cff`).
5. **Smoke test.** A new test checks the ResNet-18 MAC count (≈ 1.8G) and
   that all five paper tables are written.

No training behaviour or default value changed. Existing run directories
remain valid; they only lack `macs`, so Tab. 3 shows "--" in the FLOPs
column until those runs are repeated.

## 3. Gaps that need an author decision

### 3.1 The meta-gradient is first-order with no unrolling, but the paper describes unrolling ⚠️ (paper text)

The paper says the meta-gradient "must backpropagate through the quantizer
across K unrolled inner steps" (Secs. 2.2, 3.3, 4.3). It gives memory as
O(K·|θ|) (Sec. 3.3, Limitations) and suggests "first-order meta-learning
approximations [26] as a drop-in replacement for the bilevel unrolling" as
*future* work (Sec. 5).

`src/training/loop.py` already uses first-order updates. Every K inner
steps it runs one forward pass on a held-out batch at the **current θ**.
The gradient reaches φ only through the soft-quantizer thresholds of that
one pass. Nothing is backpropagated through the K inner updates. This
has three consequences:

* Extra memory is O(|θ|), independent of K.
* Outer-loop overhead scales with **1/K**. A *larger* K means *fewer* outer
  steps and is cheaper. The abstract says overhead is "reducible to
  approximately half with fewer inner-loop steps". In this code, that
  reduction comes from *more* inner steps per outer step.
* The argument that the STE fails because it is "composed across K unrolled
  steps" does not describe this code (see 3.2).

**Recommendation:** treat this as an erratum or arXiv-version fix. Describe
the outer step as a first-order, one-step meta-gradient with θ held fixed.
Adding real K-step unrolling (e.g. with `torch.func`) would change every
number in the paper.

### 3.2 What the "STE instead of Q_soft" ablation does in the code (paper text)

With `--train-mode ste`, `ste_ternary` detaches τ. φ then gets **no
task-loss gradient at all**, only the gradients of R_smooth and
R_consistency. The −3.3 % in Tab. 4 therefore measures "the hypernetwork
receives no task signal". It does not measure "the STE corrupts the
unrolled signal". Sec. 4.3 should say so.

### 3.3 σ: activation statistic or weight variance? (paper text)

Sec. 3.1 defines σ⁽ˡ⁾ as an *activation* statistic, and the code uses the
input-activation std (log-transformed). Tab. 4, Fig. 3 and Sec. 4.3 call it
"weight variance" and explain the ablation in terms of weight
distributions. The code never feeds weight statistics to the hypernetwork.
Weights enter only through the σ(W) normalisation of the thresholds.
**Recommendation:** rename the ablation row "w/o activation std σ".
Alternatively, if a weight-variance feature was meant, add it as a fifth
statistic, which would change the hypernetwork's parameter count.

The code also applies two input transforms that the paper does not
describe: an EMA (momentum 0.9) and the features
`[μ, log σ, log γ, ρ]`, each standardised across layers. Both affect
reproducibility, so a sentence in Sec. 3.1 or the supplement would help.

### 3.4 Training quantizer: Q_soft (paper) vs `hard_soft` (configs)

Eq. 3 trains θ through Q_soft. The released configs set
`"train_mode": "hard_soft"`: the forward pass uses exact ternary weights,
and Q_soft supplies only the backward gradient. That setting was chosen in
the `make_jobs.py select` phase, which the paper does not mention.
Running `python run.py` *without* `--config` uses plain `soft`. The paper
or supplement should state which mode produced the reported numbers. The
README already describes both modes.

### 3.5 Where held-out data is used

The paper (Sec. 4.1) says the held-out split is "never used for model
selection". In the code it is used for:

* the outer loop (as the paper says);
* **hyperparameter selection**: the training-quantizer setting (`select`
  phase) and the τ of the `fixed (tuned)` baseline;
* choosing `checkpoints/best.pt`. Reported test accuracy always comes from
  the **final-epoch** model, not from `best.pt`, so no checkpoint selection
  enters the results;
* **BatchNorm running statistics.** The outer forward pass runs in
  `model.train()`, so held-out batches update BN running means and
  variances. This affects ResNet and MobileNetV2; the LayerNorm backbones
  are unaffected. It is a small leak against the paper's statement that
  the split is "reserved exclusively for policy updates".

**Recommendation:** in the paper, write "never used for checkpoint selection
or test reporting; used for the outer loop and for choosing the training
quantizer". In the code, a possible follow-up is an opt-in flag that
freezes BN statistics during the outer step, e.g. by switching BN modules
to `eval()` around the meta forward. That flag is not included here,
because the default should stay the setting that produced the paper's
numbers.

### 3.6 The numbers in Tab. 3 do not follow from the code ⚠️

* **HN overhead.** 4,610 ÷ backbone params gives 0.039 % (ResNet-18),
  0.018 %, 0.132 % (MobileNetV2), 0.016 %, 0.021 % and 0.016 %. The paper
  reports 0.08 %, 0.04 %, 0.27 %, 0.03 %, 0.04 % and 0.03 %, about 2×
  higher in every column. `paper_table3_efficiency.tex` shows the ratio
  that `aggregate.py` computes. Either the paper's column uses a different
  definition (if so, state it), or it needs an erratum. The claim "< 0.3 %"
  holds either way.
* **Mem. Cost.** The paper says "ratio of total model memory with and
  without the hypernetwork". The code measures **peak training memory**,
  ours versus the `fixed` baseline. A 4,610-parameter MLP cannot add 6–10 %
  to the stored model, so peak training memory must be what was measured.
  The caption should say so.
* **ViT-S/16 FLOPs.** The measured value is 4.24 G MACs; the paper gives
  4.6 G, the figure DeiT reports. The two methods count attention
  operations differently. Pick one convention and state it.
* **~30 % time per epoch.** This overhead is not in any table.
  `paper_table3_efficiency.tex` now carries the measured ours/fixed time
  ratio as a LaTeX comment per backbone.

### 3.7 Experiments the code runs but the paper does not report

* `make_jobs.py main` runs **all six backbones** on CIFAR-10/100. The paper
  reports CIFAR results for ResNet-18 only (Tab. 1). To reproduce only what
  is published, use `make_jobs.py main --models resnet18`. The README now
  says so.
* The `fixed` (tuned global ±τ) and `perlayer` baselines appear in the main
  sweep. Only `perlayer` appears in the paper (Tab. 4). The `fixed`
  baseline is the most direct test of the "quantization compromise paradox"
  (ours − fixed (tuned), paired), so it may be worth adding to the
  supplement.
* ViT-S/16 uses **DeiT-S** weights (`deit_small_patch16_224.fb_in1k`). The
  README says so; the paper does not.

### 3.8 Release items the paper promises

* **"Code and pretrained weights are available"** (Sec. 4.1). The repo
  contains no weights and no download script. Suggestion: attach the
  `best.pt`/`last.pt` files of the reported runs to a GitHub Release (or
  the Hugging Face Hub), together with their `results.json`. Then add a
  `scripts/download_weights.py` and an `--eval_only` path to `run.py`.
* **No LICENSE file.** Without one, others have no legal permission to
  reuse the code. The authors need to choose a license (MIT, Apache-2.0
  and BSD-3 are common for research code).
* **Figures 2 and 3** have no generating script. `scripts/plot_thresholds.py`
  plots threshold evolution, which is not a paper figure. A
  `scripts/plot_paper_figures.py` that reads the same run groups as
  `aggregate.py` would complete the rule that every number comes from this
  code.
* **The paper PDF** is not tracked (correctly, given the publisher's
  copyright). Link the CVF open-access page or arXiv from the README once
  either exists.

### 3.9 Minor points

* `R_consistency` uses module-registration order. In timm ResNets this puts
  a block's `downsample` conv after `conv2`, which is close to, but not
  exactly, forward order.
* `pyproject.toml` still has the internal name `experimentswacv`.
* `Method.theta_extra_params` is never called. θ is
  `model.parameters()`, which already includes the quantizer submodules.
  The method can be removed.
