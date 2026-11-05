#!/usr/bin/env python3
"""Single experiment entry point.

Every run writes one self-contained directory::

    runs/<run_name>/
        cmd.txt               exact launch command
        config.json           fully resolved configuration
        env.txt               Python/torch/CUDA, GPU, git commit, SLURM ids
        results.json          final metrics (what scripts/aggregate.py reads)
        logs/metrics.jsonl    one JSON line per epoch
        checkpoints/last.pt   resumable state and final weights (resumed
                              automatically)

Examples::

    # 1) full-precision fine-tune: the shared init for all ternary runs
    python run.py --dataset cifar10 --model resnet18 --method fp --seed 0

    # 2) a ternary method, initialised from that FP run
    python run.py --dataset cifar10 --model resnet18 --method ours --seed 0

    # 3) the same settings from a stored config, with a CLI override
    python run.py --config configs/cifar.json --model vit_s16 --seed 1

    # CPU smoke test (random data, random init, 2 batches per epoch)
    python run.py --dataset smoketest --model resnet18 --method ours \\
        --no-pretrained --limit_batches 2
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

from src.models.backbones import (  # noqa: E402
    BACKBONES,
    build_model,
    count_macs,
    count_params,
    default_optimizer,
    layer_weight_profile,
    set_qstate,
    ternarize,
)
from src.quantization.quantizers import QuantState  # noqa: E402
from src.training.loop import train_fp, train_ternary  # noqa: E402
from src.training.methods import METHODS, build_method  # noqa: E402
from src.utils.data import NUM_CLASSES, get_loaders  # noqa: E402

# Per-dataset defaults, used only when neither --config nor the CLI sets them.
DATASET_DEFAULTS = {
    "cifar10": dict(fp_epochs=15, epochs=30, batch_size=128),
    "cifar100": dict(fp_epochs=15, epochs=30, batch_size=128),
    "imagenet": dict(fp_epochs=0, epochs=10, batch_size=256),
    "smoketest": dict(fp_epochs=1, epochs=2, batch_size=16),
}
LR = {"sgd": 1e-2, "adamw": 1e-4}
WEIGHT_DECAY = {"sgd": 5e-4, "adamw": 5e-2}


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser. Dashed spellings are kept as aliases."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add = p.add_argument

    g = p.add_argument_group("experiment")
    add = g.add_argument
    add("--config", default=None, help="JSON file of defaults; CLI wins")
    add(
        "--run_name",
        default="",
        help="auto: <ds>_<model>_<method><tag>_s<seed>",
    )
    add("--out_dir", default="runs", help="parent of all run directories")
    add("--method", choices=("fp",) + METHODS, help="fp or a ternary method")
    add("--tag", default="", help="suffix naming an ablation variant")
    add(
        "--init_from",
        default=None,
        help="FP checkpoint for ternary runs (auto: the matching fp run)",
    )

    g = p.add_argument_group("data")
    add = g.add_argument
    add("--dataset", choices=list(DATASET_DEFAULTS))
    add("--data_dir", "--data-root", dest="data_dir", default="./data")
    add("--num_workers", "--workers", dest="num_workers", type=int, default=8)
    add(
        "--img_size",
        "--img-size",
        dest="img_size",
        type=int,
        default=224,
        help="input resolution fed to the backbone",
    )
    add("--download", action="store_true", help="download CIFAR if missing")

    g = p.add_argument_group("model")
    add = g.add_argument
    add("--model", choices=list(BACKBONES))
    add(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ImageNet-pretrained init (fp runs only)",
    )

    g = p.add_argument_group("optimization")
    add = g.add_argument
    add("--epochs", type=int, default=None, help="ternary epochs")
    add("--fp_epochs", "--fp-epochs", dest="fp_epochs", type=int, default=None)
    add(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
    )
    add(
        "--optimizer",
        "--opt",
        dest="optimizer",
        default="auto",
        choices=("auto", "sgd", "adamw"),
        help="auto: per backbone family",
    )
    add(
        "--scheduler",
        default="cosine",
        choices=("cosine",),
        help="cosine decay with linear warmup",
    )
    add("--lr", type=float, default=None, help="auto: 1e-2 sgd / 1e-4 adamw")
    add(
        "--weight_decay",
        "--wd",
        dest="weight_decay",
        type=float,
        default=None,
        help="auto: 5e-4 sgd / 5e-2 adamw",
    )
    add(
        "--warmup_frac",
        "--warmup-frac",
        dest="warmup_frac",
        type=float,
        default=0.03,
    )
    add("--clip", type=float, default=1.0, help="gradient-norm clip")

    g = p.add_argument_group("reproducibility and hardware")
    add = g.add_argument
    add("--seed", type=int, default=0)
    add(
        "--deterministic",
        action="store_true",
        help="deterministic cuDNN/algorithms (slower)",
    )
    add("--device", default="auto", choices=("auto", "cuda", "cpu"))
    add(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bf16 autocast on CUDA",
    )
    add(
        "--resume",
        default="auto",
        help="'auto' (runs/<name>/checkpoints/last.pt), 'none', or a path",
    )

    g = p.add_argument_group("hypernetwork and bilevel (ours)")
    add = g.add_argument
    add("--lr_phi", "--lr-phi", dest="lr_phi", type=float, default=1e-3)
    add("--lambda1", type=float, default=1e-4, help="R_smooth weight")
    add("--lambda2", type=float, default=1e-3, help="R_consistency weight")
    add(
        "--k_inner",
        "--k-inner",
        dest="k_inner",
        type=int,
        default=5,
        help="inner steps per outer step",
    )
    add("--hn_width", "--hn-width", dest="hn_width", type=int, default=64)
    add("--hn_depth", "--hn-depth", dest="hn_depth", type=int, default=2)
    add("--symmetric", action="store_true", help="force tau- = -tau+")
    add(
        "--stats",
        default="mu,sigma,gamma,rho",
        help="statistics fed to the hypernetwork",
    )
    add(
        "--train_mode",
        "--train-mode",
        dest="train_mode",
        default="soft",
        choices=("soft", "ste", "hard_soft"),
        help="training quantizer for ours/fixed/perlayer: soft = Q_soft "
        "(Eq. 3); ste = STE ablation (Tab. 4); hard_soft = hard forward, "
        "Q_soft backward",
    )

    g = p.add_argument_group("soft quantizer")
    add = g.add_argument
    add("--T0", type=float, default=5.0, help="initial temperature")
    add("--T_min", "--T-min", dest="T_min", type=float, default=0.05)
    add(
        "--temp_schedule",
        "--temp-schedule",
        dest="temp_schedule",
        default="exp",
        choices=("exp", "linear", "const"),
    )

    g = p.add_argument_group("baselines")
    add = g.add_argument
    add(
        "--fixed_tau",
        "--fixed-tau",
        dest="fixed_tau",
        type=float,
        default=0.5,
        help="global +-tau of the fixed baseline (sigma units)",
    )

    g = p.add_argument_group("debugging")
    add = g.add_argument
    add(
        "--limit_batches",
        "--limit-batches",
        dest="limit_batches",
        type=int,
        default=0,
        help="cap batches per epoch (0 = all)",
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI over an optional JSON config, then resolve auto values."""
    parser = build_parser()
    partial, _ = parser.parse_known_args(argv)
    if partial.config is not None:
        with open(partial.config, encoding="utf-8") as f:
            parser.set_defaults(**json.load(f))
    a = parser.parse_args(argv)
    for required in ("dataset", "model", "method"):
        if getattr(a, required) is None:
            parser.error(f"--{required} is required (CLI or --config)")

    for k, v in DATASET_DEFAULTS[a.dataset].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.optimizer == "auto":
        a.optimizer = default_optimizer(a.model)
    if a.lr is None:
        a.lr = LR[a.optimizer]
    if a.weight_decay is None:
        a.weight_decay = WEIGHT_DECAY[a.optimizer]
    if a.dataset == "smoketest" and a.model not in ("vit_s16", "swin_t"):
        a.img_size = 32
    if not a.run_name.strip():
        a.run_name = default_run_name(
            a.dataset, a.model, a.method, a.tag, a.seed
        )
    return a


def default_run_name(dataset, model, method, tag, seed) -> str:
    """Canonical run name; scripts/gpu_queue.py relies on this format."""
    return f"{dataset}_{model}_{method}{tag}_s{seed}"


def set_seed(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy and torch; optionally force determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a few upsampling kernels have no deterministic version.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def _git(*cmd: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *cmd], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info(device: torch.device) -> dict:
    """Software/hardware facts needed to reproduce or audit a run."""
    import timm
    import torchvision

    info = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(
            _git("status", "--porcelain", "--untracked-files=no")
        ),
    }
    for k in (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_CLUSTER_NAME",
    ):
        if k in os.environ:
            info[k.lower()] = os.environ[k]
    return info


def write_json(path: Path, obj: dict) -> None:
    """Atomic JSON write (a crash never leaves a half-written file)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def resolve_init(args: argparse.Namespace) -> Path:
    """FP checkpoint a ternary run starts from."""
    if args.init_from:
        p = Path(args.init_from)
        return p / "checkpoints" / "last.pt" if p.is_dir() else p
    fp_name = default_run_name(args.dataset, args.model, "fp", "", args.seed)
    return Path(args.out_dir) / fp_name / "checkpoints" / "last.pt"


def _macs(model, img_size: int) -> int | None:
    """MACs per image (paper Tab. 3 "FLOPs"); None if counting fails."""
    try:
        return count_macs(model, img_size)
    except Exception as e:  # never fail a training run over a statistic
        print(f"warning: MAC count failed ({e})")
        return None


def main(argv: list[str] | None = None) -> int:
    """Run one experiment; returns a process exit code."""
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    set_seed(args.seed, args.deterministic)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    run_dir = Path(args.out_dir) / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "cmd.txt").write_text(
        "python run.py " + shlex.join(argv) + "\n", encoding="utf-8"
    )
    write_json(run_dir / "config.json", vars(args))
    env = environment_info(device)
    (run_dir / "env.txt").write_text(
        "".join(f"{k}: {v}\n" for k, v in env.items()), encoding="utf-8"
    )
    print(
        f"== {args.run_name} on {env['gpu'] or env['platform']} "
        f"-> {run_dir}"
    )

    loaders = get_loaders(
        args.dataset,
        args.data_dir,
        args.batch_size,
        args.num_workers,
        download=args.download,
        seed=args.seed,
    )
    model = build_model(
        args.model,
        NUM_CLASSES[args.dataset],
        args.img_size,
        pretrained=args.pretrained and args.method == "fp",
    )
    result = {
        "run_name": args.run_name,
        "config": vars(args),
        "env": env,
        "started": datetime.now().isoformat(),
    }
    t0 = time.time()

    if args.method == "fp":
        result["macs"] = _macs(model, args.img_size)
        model.to(device)
        res = train_fp(model, loaders, args, device, run_dir=run_dir)
        result.update(
            test_acc=res["test_acc"],
            history=res["history"],
            params=count_params(model)[0],
        )
    else:
        init = resolve_init(args)
        if not init.exists():
            print(
                f"missing FP checkpoint {init}: run --method fp "
                f"--seed {args.seed} first, or pass --init_from"
            )
            return 2
        state = torch.load(init, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        result["macs"] = _macs(model, args.img_size)
        layers = ternarize(model)
        profile = layer_weight_profile(model)
        model.to(device)
        qstate = QuantState(args.T0, args.T_min, args.temp_schedule)
        set_qstate(model, qstate)
        method = build_method(args.method, layers, args).to(device)

        total, tern = count_params(model)
        n_method = sum(p.numel() for p in method.parameters())
        print(
            f"ternary layers: {len(layers)} | params {total:,} "
            f"(ternary weights {tern:,}) | method params {n_method:,}"
        )

        res = train_ternary(
            model,
            method,
            qstate,
            loaders,
            args,
            device,
            run_dir=run_dir,
            resume=args.resume,
        )
        epoch_times = [h["epoch_time_s"] for h in res["history"]]
        result.update(
            init_from=str(init),
            test_acc=res["test_acc"],
            history=res["history"],
            params=total,
            ternary_weight_params=tern,
            num_ternary_layers=len(layers),
            method_params=n_method,
            peak_mem_mb=res["peak_mem_mb"],
            final_thresholds=res["final_thresholds"],
            ternary_fractions=res["ternary_fractions"],
            layer_profile=profile,
            mean_epoch_time_s=(
                float(np.mean(epoch_times)) if epoch_times else None
            ),
        )

    result["wall_time_s"] = time.time() - t0
    result["finished"] = datetime.now().isoformat()
    write_json(run_dir / "results.json", result)
    print(
        f"saved {run_dir / 'results.json'} | "
        f"test acc {result['test_acc']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
