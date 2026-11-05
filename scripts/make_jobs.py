"""
Generate job files for scripts/gpu_queue.py, one per experiment group of the
paper (camera-ready, WACV 2027):

    python scripts/make_jobs.py main --data_dir '$DATA' \
        --config configs/cifar.json          # Tab. 1, CIFAR-10/100 columns
    python scripts/make_jobs.py imagenet --data_dir '$DATA' \
        --config configs/imagenet.json    # Tab. 1 ImageNet column, Tabs. 2-3
    python scripts/make_jobs.py ablation --data_dir '$DATA' \
        --config configs/cifar.json          # Tabs. 4-5

    main      ResNet-18 on CIFAR-10/100, 3 seeds, the methods of Tab. 1
    imagenet  all six backbones on ImageNet, 1 seed, the methods of Tab. 2,
              plus `fixed` (same quantizer, no hypernetwork) as the reference
              for the Tab. 3 memory / time overhead
    ablation  CIFAR-10 / ResNet-18, 3 seeds, every row of Tabs. 4 and 5

Writes jobs/<phase>.txt. FP fine-tunes come first, then a `---` barrier, then
the ternary runs that start from them. All settings are fixed in advance
(configs/*.json, paper Sec. 4.1); nothing is tuned on held-out or test data.
"""

import argparse
import os

ALL_MODELS = [
    "resnet18",
    "resnet50",
    "mobilenetv2",
    "convnext_t",
    "vit_s16",
    "swin_t",
]
# Tabs. 1 and 2: the evaluated baselines (Sec. 4) and ours
MAIN_METHODS = [
    "twn",
    "ttq",
    "rtn",
    "metaquant",
    "lsq",
    "fatnn",
    "ours",
]
# Tab. 3 compares ours with the same pipeline without the hypernetwork
IMAGENET_METHODS = MAIN_METHODS + ["fixed"]

ABLATIONS = {
    # Tab. 4 -- statistic conditioning (features masked at the HN input)
    "_no_gamma": "--stats mu,sigma,rho",
    "_no_rho": "--stats mu,sigma,gamma",
    "_no_sigma": "--stats mu,gamma,rho",
    "_mean_only": "--stats mu",
    # Tab. 4 -- hypernetwork capacity
    "_w32": "--hn-width 32",
    "_w128": "--hn-width 128",
    "_d1": "--hn-depth 1",
    "_d3": "--hn-depth 3",
    # Tab. 4 -- quantization design
    "_symmetric": "--symmetric",
    "_linear_T": "--temp-schedule linear",
    "_const_T": "--temp-schedule const",
    "_ste": "--train-mode ste",
    # Tab. 5 -- sensitivity (each swept with the others at their defaults)
    "_l1_1e-5": "--lambda1 1e-5",
    "_l1_1e-3": "--lambda1 1e-3",
    "_l2_1e-4": "--lambda2 1e-4",
    "_l2_1e-2": "--lambda2 1e-2",
    "_k1": "--k-inner 1",
    "_k3": "--k-inner 3",
    "_k10": "--k-inner 10",
}


def cmd(ds, model, method, seed, common, tag="", extra=""):
    c = (
        f"python run.py --dataset {ds} --model {model} "
        f"--method {method} --seed {seed} {common}"
    )
    if tag:
        c += f" --tag {tag}"
    if extra:
        c += f" {extra}"
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "phase",
        choices=["main", "ablation", "imagenet"],
    )
    ap.add_argument(
        "--data_dir", "--data-root", dest="data_dir", required=True
    )
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument(
        "--config",
        default=None,
        help="JSON config for every run (required)",
    )
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--models", default=None, help="override model list")
    ap.add_argument("--datasets", default="cifar10,cifar100")
    ap.add_argument("--extra", default="", help="extra flags for every run")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    if not a.config:
        ap.error(
            f"{a.phase} needs --config (the frozen settings used in the paper)"
        )
    common = f"--data_dir {a.data_dir} --out_dir {a.out_dir}"
    if a.config:
        common += f" --config {a.config}"
    common = f"{common} {a.extra}".strip()
    fp, tern = [], []

    if a.phase == "main":
        # the paper's CIFAR results (Tab. 1) are ResNet-18 only
        models = a.models.split(",") if a.models else ["resnet18"]
        for ds in a.datasets.split(","):
            for m in models:
                for s in seeds:
                    fp.append(cmd(ds, m, "fp", s, common))
                    for meth in MAIN_METHODS:
                        tern.append(cmd(ds, m, meth, s, common))

    elif a.phase == "ablation":
        models = a.models.split(",") if a.models else ["resnet18"]
        for m in models:
            for s in seeds:
                fp.append(cmd("cifar10", m, "fp", s, common))
                tern.append(cmd("cifar10", m, "ours", s, common))
                tern.append(cmd("cifar10", m, "ours_joint", s, common))
                tern.append(cmd("cifar10", m, "perlayer", s, common))
                for tag, flags in ABLATIONS.items():
                    tern.append(
                        cmd(
                            "cifar10",
                            m,
                            "ours",
                            s,
                            common,
                            tag=tag,
                            extra=flags,
                        )
                    )

    elif a.phase == "imagenet":
        models = a.models.split(",") if a.models else ALL_MODELS
        seeds = seeds[:1]
        for m in models:
            fp.append(cmd("imagenet", m, "fp", seeds[0], common))
            for meth in IMAGENET_METHODS:
                tern.append(cmd("imagenet", m, meth, seeds[0], common))

    os.makedirs("jobs", exist_ok=True)
    path = os.path.join("jobs", f"{a.phase}.txt")
    with open(path, "w", newline="\n") as f:
        f.write(f"# generated by scripts/make_jobs.py {a.phase}\n")
        if fp:
            f.write("\n".join(dict.fromkeys(fp)) + "\n---\n")
        f.write("\n".join(dict.fromkeys(tern)) + "\n")
    print(f"wrote {path}: {len(set(fp))} FP + {len(set(tern))} ternary jobs")


if __name__ == "__main__":
    main()
