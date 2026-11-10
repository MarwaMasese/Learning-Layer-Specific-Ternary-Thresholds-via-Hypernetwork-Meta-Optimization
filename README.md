# Learning Layer-Specific Ternary Thresholds via Hypernetwork Meta-Optimization

Official code for the WACV 2027 paper *Learning Layer-Specific Ternary
Thresholds via Hypernetwork Meta-Optimization* (Cornelius Maroa Masese,
Ali Hussein, Ashery Mbilinyi). A compact hypernetwork, shared across
all layers, reads each layer's activation and gradient statistics and outputs
that layer's asymmetric ternary thresholds. It is trained by first-order
bilevel optimization through a temperature-annealed soft quantizer. Every
number in the paper comes from a run directory produced by this code and
summarized by `scripts/aggregate.py`.

<p align="center">
  <img src="docs/figures/bilevel_pipeline.png" width="760"
       alt="Bilevel training: the inner loop updates weights with thresholds from the hypernetwork; the outer loop updates the hypernetwork on held-out data">
</p>

## Method

**Why per-layer thresholds.** Layers differ in how their weights are
distributed, so one threshold can zero out too much of one layer and too
little of another. The hypernetwork sets each layer's thresholds from that
layer's own statistics.

<p align="center">
  <img src="docs/figures/threshold_illustration.png" width="760"
       alt="Schematic: one static threshold versus layer-specific thresholds on two idealized weight distributions">
  <br><em>Schematic with two idealized weight distributions.</em>
</p>

- **Ternary layers** (`src/models/layers.py`): every Conv2d/Linear except the
  stem and the classifier computes `W_q = α_l · Q(W / σ(W); τ+, τ−)`, with a
  per-layer least-squares scale `α_l` (the TWN scale for hard ternary
  weights). Thresholds are in units of the layer's weight std, with
  `0 < τ+ < 1` and `−1 < τ− < 0`.
- **Statistics** `[μ, σ, γ, ρ]` (`src/quantization/statistics.py`):
  input-activation mean/std/sparsity and weight-gradient magnitude,
  EMA-smoothed, fed as `[μ, log σ, log γ, ρ]` standardized across layers.
- **Hypernetwork** (`src/quantization/hypernetwork.py`): shared
  4→64→64→2 MLP (4,610 parameters), sigmoid-parameterized thresholds.
- **Bilevel training** (`src/training/loop.py`): θ is trained on the train
  split; every K = 5 inner steps, φ takes one step on a held-out batch (10% of
  the CIFAR train set, or 10 images per class for ImageNet), using a
  first-order meta-gradient (Secs. 3.2 and 3.3: second-order terms are
  omitted, as in first-order MAML). Held-out data is used only for this
  update: it never updates BatchNorm statistics and never selects a
  checkpoint.
- **Soft quantizer** (`src/quantization/quantizers.py`): Q_soft, Eq. 4 of the
  paper, with T(p) = 5·exp(−5p) floored at 0.05. The default
  `--train-mode hard_soft` takes the forward pass from the exact ternary
  weights and the gradients from Q_soft, so training and evaluation use the
  same weights; `--train-mode soft` uses Q_soft for both, which is the
  literal reading of the inner loop (Eq. 3). See
  [Which setting reproduces the paper](#which-setting-reproduces-the-paper).
  The STE ablation (`--train-mode ste`) uses a hard forward pass with an
  identity backward pass for the weights and, by default, no gradient at all
  to the thresholds; `--ste-tau-grad` instead gives each threshold the
  constant −1/2 gradient of Sec. 3.3.
  **Every reported accuracy uses exact hard ternary weights.**

<p align="center">
  <img src="docs/figures/soft_quantizer.png" width="480"
       alt="Soft ternary quantizer for several temperatures; it approaches the hard quantizer as T decreases">
</p>

**Baselines** (`--method`), as in the paper's Tabs. 1 and 2: `twn`, `ttq`,
`rtn`, `metaquant`, `lsq` (three levels) and `fatnn`, plus `fp`. The Tab. 4
ablations use `perlayer` (free per-layer τ, no hypernetwork) and
`ours_joint` (hypernetwork trained in a single level). Tab. 3 uses `fixed`
(same pipeline, one global ±τ = 0.5, no hypernetwork) as the "without
hypernetwork" reference for memory and time. Every baseline runs under the same protocol as ours
(initialisation, data, epochs, optimizer, schedule) with ternary weights and
full-precision activations. RTN, FATNN and MetaQuant use their published
weight quantizers; the adaptations needed for this setting (MetaQuant with a
ternary TWN forward instead of binary, FATNN's weight normalisation per layer,
RTN's initialisation) are documented in `src/quantization/quantizers.py` and
`src/quantization/metaquant.py`.
**Backbones** (`--model`): `resnet18 resnet50 mobilenetv2 convnext_t vit_s16
swin_t` (timm ImageNet-1k weights; `vit_s16` uses DeiT-S weights).

## Setup
Python ≥ 3.10.
```bash
# After extracting the source archive, enter its directory.
cd <source-directory>
python -m venv ~/envs/ternary && source ~/envs/ternary/bin/activate
pip install -r requirements.txt   # Alliance: pip install --no-index torch torchvision numpy timm
python scripts/prefetch.py        # CIFAR-10/100 + all weights, then an offline check
make test                         # CPU smoke tests, about 1-2 min: ALL PASSED
```
The exact package versions used by each run are recorded in
`runs/<name>/env.txt`. Run outputs, downloaded data, and checkpoints are not
included in the source archive; they are created locally when you run the code.

## Running one experiment
```bash
# full-precision fine-tune (also the init for every ternary run)
python run.py --dataset cifar10 --model resnet18 --method fp --seed 0
# a ternary method, initialised from runs/cifar10_resnet18_fp_s0
python run.py --dataset cifar10 --model resnet18 --method ours --seed 0
# the paper settings from a config file, with any CLI override
python run.py --config configs/cifar.json --dataset cifar10 \
    --model vit_s16 --method ours --seed 1
python run.py --help              # every argument, with defaults
```
No experiment needs a code edit: every setting is a command-line argument or a
`--config` JSON key (the command line wins). `--deterministic` gives identical
reruns at some cost in speed.

Each run writes `runs/<dataset>_<model>_<method><tag>_s<seed>/`:

| file | content |
|---|---|
| `cmd.txt` | exact launch command |
| `config.json` | fully resolved configuration |
| `env.txt` | Python/torch/CUDA/timm versions, GPU, git commit, SLURM ids |
| `logs/metrics.jsonl` | one JSON line per epoch (loss, held-out accuracy, T, thresholds) |
| `checkpoints/last.pt` | resumable state and final weights; rerunning the same command resumes |
| `results.json` | final test accuracy, thresholds, per-layer profile, timing |

## Running sweeps
```bash
python scripts/make_jobs.py main --data_dir '$DATA' --config configs/cifar.json
# SLURM (e.g. Alliance clusters): one array per wave, chained; logs in slurm_logs/
NOTIFY_EMAIL=you@uni.ca bash scripts/slurm_submit.sh jobs/main.txt def-yourprof
# machines without SLURM: one job per GPU, logs in queue_logs/
python scripts/gpu_queue.py jobs/main.txt --gpus 0,1
```
Resubmitting a jobs file skips finished runs and resumes unfinished ones.

## Reproducing the paper
Run the FP wave of each jobs file first (the generated files do this, with a
`---` barrier), then:

| paper item | jobs | output |
|---|---|---|
| Tab. 1, CIFAR-10/100 columns (ResNet-18, 3 seeds) | `make_jobs.py main --config configs/cifar.json` | `paper_table1_resnet18.tex` |
| Tab. 1 ImageNet column, Tab. 2 (6 backbones, 1 seed) | `make_jobs.py imagenet --config configs/imagenet.json` | `paper_table1_resnet18.tex`, `paper_table2_imagenet.tex` |
| Tab. 3 (params, FLOPs, HN overhead, memory) | the same ImageNet runs (`ours` vs `fixed`) | `paper_table3_efficiency.tex` |
| Fig. 2 | the same ImageNet runs | `fig2_imagenet_efficiency.pdf` |
| Tab. 4 ablations, Tab. 5 sensitivity (CIFAR-10, ResNet-18, 3 seeds) | `make_jobs.py ablation --config configs/cifar.json` | `paper_table4_ablation.tex`, `paper_table5_sensitivity.tex` |
| Fig. 3 | the same ablation runs | `fig3_ablations.pdf` |

```bash
python scripts/aggregate.py --latex tables              # Tabs. 1-5 (mean)
python scripts/aggregate.py --latex tables --paper_std  # mean ± std
python scripts/plot_paper_figures.py --out plots        # Figs. 2-3
```

Every setting is fixed in advance in `configs/*.json` (paper Sec. 4.1).
The held-out split is used only by the outer loop and for per-epoch
monitoring. Reported accuracy is the test accuracy of the final-epoch model
with exact ternary weights. Nothing is tuned on held-out or test data.

### Which setting reproduces the paper

The shipped defaults are the ones the published tables were run with. Two
options depart from them, and both are off by default:

| option | default | effect |
|---|---|---|
| `--train-mode soft` | `hard_soft` | Trains on the smooth weights while accuracy is still reported on hard ternary ones. This is the literal reading of Eq. 3, but train and eval weights then differ, and it does **not** reproduce the published tables. Affects `ours`, `fixed` and `perlayer`, i.e. the "Ours" row of Tabs. 1–2, Tab. 3's accuracy column and all of Tabs. 4–5. |
| `--ste-tau-grad` | off | Gives each threshold the constant −1/2 gradient Sec. 3.3 describes. Without it the thresholds get no gradient, so in `--train-mode ste` the hypernetwork stays at its initialisation and Tab. 4's "STE instead of Q_soft" row measures a *frozen* hypernetwork rather than STE as such. The flag is the more faithful reading of the text; the published −3.3% was measured without it. |

Two further notes on reading the paper against this code:

- Tab. 3's **Mem. Cost** column is the ratio of peak *training* memory, ours
  against the `fixed` baseline — not model memory. The caption's "total model
  memory with and without the hypernetwork" cannot be the measured quantity:
  the hypernetwork is 4,610 parameters, far too few to account for 1.06–1.10×.
- The paper's tables and Figs. 2–3 are typeset by hand in the paper source.
  `aggregate.py --latex` and `plot_paper_figures.py` regenerate the same rows,
  columns and grouping from run directories as a cross-check; they are not the
  source the paper was built from.

## Citation
```bibtex
@inproceedings{masese2027ternary,
  title     = {Learning Layer-Specific Ternary Thresholds via Hypernetwork
               Meta-Optimization},
  author    = {Masese, Cornelius Maroa and Hussein, Ali and
               Mbilinyi, Ashery},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on
               Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Development
```bash
pip install -r requirements-dev.txt && pre-commit install
make format   # isort + black (79 columns)
make lint     # isort/black check + flake8
make test     # CPU smoke tests
```

## Repository layout
```
run.py                     single entry point (argparse + --config)
configs/                   paper settings (JSON)
src/models/                backbones, ternary layers
src/quantization/          quantizers, hypernetwork, statistics
src/training/              training loops, methods
src/utils/data.py          CIFAR / ImageNet loaders, held-out split
scripts/                   job generation, SLURM, queue, aggregation, prefetch
tests/smoke_test.py        CPU checks (make test)
docs/                      figures
```

```
