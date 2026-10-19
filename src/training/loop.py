"""Training loops.

``train_fp``
    Fine-tune the pretrained full-precision backbone (the upper bound and the
    shared starting point of every ternary method).
``train_ternary``
    One loop for all ternary methods::

        for each training batch:                          (inner loop, theta)
            install thresholds (detached; connected for joint methods)
            loss = CE(model(x), y)   [+ lambda-regularizers if joint]
            backward, update statistics, clip, step theta (and phi if joint)
            every K inner steps, if bilevel:              (outer loop, phi)
                install thresholds connected to phi
                meta = CE(model(x_heldout), y_heldout)
                       + lambda1 R_smooth + lambda2 R_consistency
                backward into phi only, step phi

The meta-gradient is first-order (theta is treated as fixed during the phi
step, as in first-order DARTS / MAML); the K inner steps use K distinct
batches. Every reported accuracy comes from ``evaluate(..., mode="hard")``:
exact ternary weights {-alpha_l, 0, +alpha_l}.

Run-directory artefacts written here (see run.py for the full layout):
``logs/metrics.jsonl``, ``checkpoints/last.pt`` and ``checkpoints/best.pt``.
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from src.models.backbones import ternary_layers

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def autocast(device: torch.device, enabled: bool):
    """bf16 autocast on CUDA, no-op elsewhere."""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_optimizer(params, kind: str, lr: float, weight_decay: float):
    """SGD (Nesterov) or AdamW; no weight decay on 0/1-d parameters."""
    params = [p for p in params if p.requires_grad]
    groups = [
        {
            "params": [p for p in params if p.ndim > 1],
            "weight_decay": weight_decay,
        },
        {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0},
    ]
    if kind == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))


def cosine_with_warmup(opt, total_steps: int, warmup_steps: int):
    """Per-step LR multiplier: linear warmup, then cosine decay to zero."""

    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    amp=True,
    qstate=None,
    mode="hard",
    limit_batches: int = 0,
) -> float:
    """Top-1 accuracy (%). Ternary models are evaluated in ``mode``."""
    prev = qstate.mode if qstate is not None else None
    if qstate is not None:
        qstate.mode = mode
    model.eval()
    correct = total = 0
    for b, (x, y) in enumerate(loader):
        if limit_batches and b >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device, amp):
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    if qstate is not None:
        qstate.mode = prev
    return 100.0 * correct / max(total, 1)


def _cycle(loader):
    while True:
        yield from loader


def _peak_mem_mb(device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return None


def _num_batches(loader, args) -> int:
    n = len(loader)
    return min(n, args.limit_batches) if args.limit_batches else n


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _log_metrics(run_dir: Path | None, rec: dict) -> None:
    if run_dir is not None:
        with open(
            run_dir / "logs" / "metrics.jsonl", "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# full precision
# ---------------------------------------------------------------------------


def train_fp(
    model, loaders, args, device, run_dir: Path | None = None, log=print
) -> dict:
    """Fine-tune in full precision; saves last.pt (the ternary init)."""
    train_loader, held_loader, test_loader = loaders
    history, best = [], -1.0
    ck = run_dir / "checkpoints" if run_dir is not None else None
    if args.fp_epochs > 0:
        opt = make_optimizer(
            model.parameters(), args.optimizer, args.lr, args.weight_decay
        )
        steps = args.fp_epochs * _num_batches(train_loader, args)
        sched = cosine_with_warmup(opt, steps, int(args.warmup_frac * steps))
        crit = nn.CrossEntropyLoss()
        for epoch in range(args.fp_epochs):
            model.train()
            t0, loss_sum, n = time.time(), 0.0, 0
            for b, (x, y) in enumerate(train_loader):
                if args.limit_batches and b >= args.limit_batches:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(device, args.amp):
                    loss = crit(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                sched.step()
                loss_sum += loss.item()
                n += 1
            held = evaluate(
                model,
                held_loader,
                device,
                args.amp,
                limit_batches=args.limit_batches,
            )
            rec = {
                "epoch": epoch,
                "train_loss": loss_sum / max(n, 1),
                "heldout_acc": held,
                "epoch_time_s": time.time() - t0,
            }
            history.append(rec)
            _log_metrics(run_dir, rec)
            log(
                f"[fp] epoch {epoch + 1}/{args.fp_epochs} "
                f"loss {rec['train_loss']:.4f} heldout {held:.2f}%"
            )
            if ck is not None and held > best:
                best = held
                _atomic_save(
                    {"model": model.state_dict(), "epoch": epoch},
                    ck / "best.pt",
                )
    if ck is not None:
        _atomic_save(
            {"model": model.state_dict(), "epoch": args.fp_epochs},
            ck / "last.pt",
        )
    test = evaluate(
        model, test_loader, device, args.amp, limit_batches=args.limit_batches
    )
    log(f"[fp] test acc {test:.2f}%")
    return {"test_acc": test, "history": history}


# ---------------------------------------------------------------------------
# ternary
# ---------------------------------------------------------------------------


def _resume_path(run_dir: Path | None, resume: str) -> Path | None:
    if resume == "none" or (run_dir is None and resume == "auto"):
        return None
    if resume == "auto":
        return run_dir / "checkpoints" / "last.pt"
    return Path(resume)


def train_ternary(
    model,
    method,
    qstate,
    loaders,
    args,
    device,
    run_dir: Path | None = None,
    resume: str = "auto",
    log=print,
) -> dict:
    """Train any ternary method; resumable from checkpoints/last.pt."""
    train_loader, held_loader, test_loader = loaders
    crit = nn.CrossEntropyLoss()

    theta = list(model.parameters())
    opt_theta = make_optimizer(
        theta, args.optimizer, args.lr, args.weight_decay
    )
    phi = method.phi_params()
    opt_phi = torch.optim.Adam(phi, lr=args.lr_phi) if phi else None
    total_steps = args.epochs * _num_batches(train_loader, args)
    sched = cosine_with_warmup(
        opt_theta, total_steps, int(args.warmup_frac * total_steps)
    )

    ck_dir = run_dir / "checkpoints" if run_dir is not None else None
    start_epoch, history, best = 0, [], -1.0
    resume_from = _resume_path(run_dir, resume)
    if resume_from is not None and resume_from.exists():
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        method.load_state_dict(ck["method"])
        method.load_extra_state(ck["method_extra"])
        opt_theta.load_state_dict(ck["opt_theta"])
        if opt_phi is not None:
            opt_phi.load_state_dict(ck["opt_phi"])
        sched.load_state_dict(ck["sched"])
        start_epoch, history = ck["epoch"] + 1, ck["history"]
        best = max((h["heldout_acc_hard"] for h in history), default=-1.0)
        log(f"[resume] {resume_from} from epoch {start_epoch}")
    elif method.bilevel or method.joint:
        _warm_start_stats(
            model, method, qstate, train_loader, crit, device, args
        )

    held_iter = _cycle(held_loader) if method.bilevel else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, args.epochs):
        T = qstate.update_temperature(epoch, args.epochs)
        qstate.mode = method.train_mode
        model.train()
        t0 = time.time()
        loss_sum = meta_sum = 0.0
        n = n_meta = 0

        for b, (x, y) in enumerate(train_loader):
            if args.limit_batches and b >= args.limit_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---- inner step (theta; plus phi for joint methods) ----
            tau = method.set_thresholds(connected=method.joint)
            method.on_train_forward(True)
            with autocast(device, args.amp):
                loss = crit(model(x), y)
            method.on_train_forward(False)
            total = loss + method.regularizer(tau) if method.joint else loss

            opt_theta.zero_grad(set_to_none=True)
            if opt_phi is not None:
                opt_phi.zero_grad(set_to_none=True)
            total.backward()
            method.current_lr = opt_theta.param_groups[0]["lr"]
            method.after_inner_backward()
            nn.utils.clip_grad_norm_(theta, args.clip)
            opt_theta.step()
            sched.step()
            if method.joint:
                opt_phi.step()
            loss_sum += loss.item()
            n += 1

            # ---- outer step (phi), every K inner steps ----
            if method.bilevel and (b + 1) % args.k_inner == 0:
                xv, yv = next(held_iter)
                xv = xv.to(device, non_blocking=True)
                yv = yv.to(device, non_blocking=True)
                tau = method.set_thresholds(connected=True)
                with autocast(device, args.amp):
                    meta = crit(model(xv), yv)
                meta = meta + method.regularizer(tau)
                opt_phi.zero_grad(set_to_none=True)
                meta.backward(inputs=phi)
                opt_phi.step()
                meta_sum += meta.item()
                n_meta += 1

        epoch_time = time.time() - t0
        method.finalize()
        held = evaluate(
            model,
            held_loader,
            device,
            args.amp,
            qstate,
            "hard",
            limit_batches=args.limit_batches,
        )
        rec = {
            "epoch": epoch,
            "temperature": T,
            "train_loss": loss_sum / max(n, 1),
            "meta_loss": meta_sum / n_meta if n_meta else None,
            "heldout_acc_hard": held,
            "epoch_time_s": epoch_time,
            "thresholds": method.threshold_table(),
            "zero_fraction": [
                round(m.ternary_fractions()["0"], 5)
                for m in ternary_layers(model)
            ],
        }
        history.append(rec)
        _log_metrics(run_dir, rec)
        log(
            f"[{args.method}] epoch {epoch + 1}/{args.epochs} T={T:.3f} "
            f"loss {rec['train_loss']:.4f} heldout(hard) {held:.2f}% "
            f"{epoch_time:.0f}s"
        )

        if ck_dir is not None:
            state = {
                "model": model.state_dict(),
                "method": method.state_dict(),
                "method_extra": method.extra_state(),
                "opt_theta": opt_theta.state_dict(),
                "opt_phi": opt_phi.state_dict() if opt_phi else None,
                "sched": sched.state_dict(),
                "epoch": epoch,
                "history": history,
            }
            _atomic_save(state, ck_dir / "last.pt")
            if held > best:
                best = held
                _atomic_save(
                    {
                        "model": model.state_dict(),
                        "method": method.state_dict(),
                        "epoch": epoch,
                    },
                    ck_dir / "best.pt",
                )

    method.finalize()
    test = evaluate(
        model,
        test_loader,
        device,
        args.amp,
        qstate,
        "hard",
        limit_batches=args.limit_batches,
    )
    log(f"[{args.method}] test acc (hard ternary) {test:.2f}%")
    fractions = [m.ternary_fractions() for m in ternary_layers(model)]
    return {
        "test_acc": test,
        "history": history,
        "peak_mem_mb": _peak_mem_mb(device),
        "final_thresholds": method.threshold_table(),
        "ternary_fractions": fractions,
    }


@torch.no_grad()
def _warm_start_stats(model, method, qstate, loader, crit, device, args):
    """One forward/backward with the initial thresholds, so the statistics
    (including gamma) exist before the hypernetwork is first queried."""
    if not hasattr(method, "tracker"):
        return
    qstate.mode = method.train_mode
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    model.train()
    method.on_train_forward(True)
    with torch.enable_grad(), autocast(device, args.amp):
        loss = crit(model(x), y)
    method.on_train_forward(False)
    with torch.enable_grad():
        loss.backward()
    method.after_inner_backward()
    model.zero_grad(set_to_none=True)
