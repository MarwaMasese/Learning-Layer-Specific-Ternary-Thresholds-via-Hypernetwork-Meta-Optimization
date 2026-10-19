"""
Generate job files for scripts/gpu_queue.py.

    python scripts/make_jobs.py select --data_dir '$DATA'
    python scripts/make_jobs.py main --data_dir '$DATA'
        --config configs/cifar.json
    python scripts/make_jobs.py ablation --data_dir '$DATA'
        --config configs/cifar.json
    python scripts/make_jobs.py imagenet --data_dir '$DATA'
        --config configs/imagenet.json

`select` chooses the training-quantizer setting (TRAIN_SETTINGS) before the
sweeps: ours and the fixed baseline under each candidate setting, CIFAR-10,
ResNet-18 and ViT-S/16, seed 0. The setting with the best HELD-OUT accuracy
of ours is then written into configs/*.json.

Writes jobs/<phase>.txt. FP fine-tunes come first, then a `---` barrier, then
the ternary runs that start from them.

Protocol rule: any choice made from these runs (e.g. the fixed baseline's τ,
learning rates) is made on HELD-OUT accuracy (`heldout_acc_hard` in the
history), never on test accuracy.
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
MAIN_METHODS = [
    "ours",
    "fixed",
    "perlayer",
    "twn",
    "ttq",
    "lsq",
    "rtn",
    "fatnn",
    "metaquant",
]
# ImageNet (Table 2): every baseline except per-layer, one seed
IMAGENET_METHODS = [m for m in MAIN_METHODS if m != "perlayer"]
FIXED_TAU_GRID = ("0.3", "0.7")  # plus the default 0.5 (`fixed`)

# candidate training-quantizer settings compared by the `select` phase
TRAIN_SETTINGS = {
    "_soft_geom005": "--temp-schedule geom --T-min 0.005",
    "_hs_005": "--train-mode hard_soft --T-min 0.05",
    "_hs_03": "--train-mode hard_soft --T-min 0.3",
}

ABLATIONS = {
    # statistic conditioning (features masked at the hypernetwork input)
    "_no_gamma": "--stats mu,sigma,rho",
    "_no_rho": "--stats mu,sigma,gamma",
    "_no_sigma": "--stats mu,gamma,rho",
    "_mean_only": "--stats mu",
    # hypernetwork capacity
    "_w32": "--hn-width 32",
    "_w128": "--hn-width 128",
    "_d1": "--hn-depth 1",
    "_d3": "--hn-depth 3",
    # quantizer design
    "_symmetric": "--symmetric",
    "_linear_T": "--temp-schedule linear",
    "_const_T": "--temp-schedule const",
    "_ste": "--train-mode ste",
    # sensitivity
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
        choices=["select", "main", "ablation", "imagenet"],
    )
    ap.add_argument(
        "--data_dir", "--data-root", dest="data_dir", required=True
    )
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument(
        "--config",
        default=None,
        help="JSON config for every run (required: main/ablation/imagenet)",
    )
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--models", default=None, help="override model list")
    ap.add_argument("--datasets", default="cifar10,cifar100")
    ap.add_argument("--extra", default="", help="extra flags for every run")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    if a.phase in ("main", "ablation", "imagenet") and not a.config:
        ap.error(
            f"{a.phase} needs --config (the frozen settings used in the paper)"
        )
    common = f"--data_dir {a.data_dir} --out_dir {a.out_dir}"
    if a.config:
        common += f" --config {a.config}"
    common = f"{common} {a.extra}".strip()
    fp, tern = [], []

    if a.phase == "select":
        models = a.models.split(",") if a.models else ["resnet18", "vit_s16"]
        for m in models:
            fp.append(cmd("cifar10", m, "fp", seeds[0], common))
            for tag, flags in TRAIN_SETTINGS.items():
                for meth in ("ours", "fixed"):
                    tern.append(
                        cmd(
                            "cifar10",
                            m,
                            meth,
                            seeds[0],
                            common,
                            tag=tag,
                            extra=flags,
                        )
                    )

    elif a.phase == "main":
        models = a.models.split(",") if a.models else ALL_MODELS
        for ds in a.datasets.split(","):
            for m in models:
                for s in seeds:
                    fp.append(cmd(ds, m, "fp", s, common))
                    for meth in MAIN_METHODS:
                        tern.append(cmd(ds, m, meth, s, common))
                    # fixed-tau grid: aggregate.py reports the tau with the
                    # best HELD-OUT accuracy as "fixed (tuned)"
                    for t in FIXED_TAU_GRID:
                        tern.append(
                            cmd(
                                ds,
                                m,
                                "fixed",
                                s,
                                common,
                                tag=f"_tau{t}",
                                extra=f"--fixed_tau {t}",
                            )
                        )

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
            for t in FIXED_TAU_GRID:
                tern.append(
                    cmd(
                        "imagenet",
                        m,
                        "fixed",
                        seeds[0],
                        common,
                        tag=f"_tau{t}",
                        extra=f"--fixed_tau {t}",
                    )
                )

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
