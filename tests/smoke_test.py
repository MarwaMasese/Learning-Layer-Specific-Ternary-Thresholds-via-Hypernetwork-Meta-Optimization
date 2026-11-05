"""
CPU sanity checks — run before launching anything on GPUs:

    python tests/smoke_test.py        # unit checks + resnet18, all methods
    python tests/smoke_test.py --all  # also all six backbones
"""

import argparse
import os
import shutil
import sys
import tempfile

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.backbones import (  # noqa: E402
    BACKBONES,
    build_model,
    set_qstate,
    ternarize,
)
from src.quantization.hypernetwork import Hypernetwork  # noqa: E402
from src.quantization.quantizers import (  # noqa: E402
    LSQTernary,
    QuantState,
    TTQTernary,
    hard_ternary,
    soft_ternary,
)


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        check.failed += 1


check.failed = 0


def test_quantizer():
    print("quantizer")
    x = torch.linspace(-3, 3, 601)
    tp, tm = torch.tensor(0.4), torch.tensor(-0.7)
    hard = hard_ternary(x, tp, tm)
    check(
        set(hard.unique().tolist()) == {-1.0, 0.0, 1.0},
        "hard ternary takes exactly {-1,0,+1}",
    )
    check(
        bool((hard[x > 0.4] == 1).all() and (hard[x < -0.7] == -1).all()),
        "hard respects thresholds",
    )
    soft = soft_ternary(x, tp, tm, 1e-3)
    away = ((x - 0.4).abs() > 0.05) & ((x + 0.7).abs() > 0.05)
    check(
        torch.allclose(soft[away], hard[away], atol=1e-3),
        "Q_soft → Q_hard as T → 0",
    )
    check(
        bool((soft_ternary(x, tp, tm, 0.5).diff() >= -1e-6).all()),
        "Q_soft monotone in w",
    )
    tp_r = tp.clone().requires_grad_()
    soft_ternary(x, tp_r, tm, 0.3).sum().backward()
    check(
        tp_r.grad is not None and tp_r.grad.abs() > 0,
        "Q_soft differentiable in τ+",
    )
    from src.quantization.quantizers import soft_ste_ternary

    tp_r, tm_r = tp.clone().requires_grad_(), tm.clone().requires_grad_()
    hs = soft_ste_ternary(x, tp_r, tm_r, 0.3)
    check(torch.equal(hs, hard), "hard_soft forward is exactly Q_hard")
    hs.sum().backward()
    check(
        tp_r.grad.abs() > 0 and tm_r.grad.abs() > 0,
        "hard_soft passes gradient to both thresholds",
    )
    from src.quantization.quantizers import ste_ternary

    tp_r, tm_r = tp.clone().requires_grad_(), tm.clone().requires_grad_()
    x_r = x.clone().requires_grad_()
    q = ste_ternary(x_r, tp_r, tm_r)
    check(torch.equal(q, hard), "STE forward is exactly Q_hard")
    q.sum().backward()
    check(
        bool((x_r.grad == 1).all())
        and abs(tp_r.grad.item() + 0.5 * x.numel()) < 1e-3
        and abs(tm_r.grad.item() + 0.5 * x.numel()) < 1e-3,
        "STE: identity gradient to w, constant gradient to both τ",
    )
    lin = QuantState(5.0, 0.05, "linear")
    check(
        abs(lin.update_temperature(29, 30) - 0.05) < 1e-9,
        "linear schedule reaches T_min",
    )
    st = QuantState(5.0, 0.05, "exp")
    Ts = [st.update_temperature(e, 30) for e in range(30)]
    check(
        Ts[0] == 5.0
        and abs(Ts[-1] - 0.05) < 1e-9
        and Ts == sorted(Ts, reverse=True),
        f"temperature anneals 5.0 → 0.05 (got {Ts[0]:.2f} → {Ts[-1]:.3f})",
    )
    w = torch.randn(64, 32) * 0.02
    for Q in (LSQTernary, TTQTernary):
        q = Q(w)(w.clone().requires_grad_())
        check(q.unique().numel() <= 3, f"{Q.__name__} output has ≤3 levels")


def test_new_baselines():
    print("RTN / FATNN / MetaQuant quantizers")
    from src.quantization.metaquant import (
        MetaQuantTernary,
        MultiFC,
        meta_step,
        twn_quantize,
    )
    from src.quantization.quantizers import FATNNTernary, RTNTernary

    w = torch.randn(16, 8, 3, 3) * 0.02
    rtn = RTNTernary(w)
    q = rtn(w)
    check(q.unique().numel() <= 3, "RTN output has ≤3 levels")
    per_filter = 0.7 * w.abs().flatten(1).mean(1).view(-1, 1, 1, 1)
    check(
        torch.equal(q != 0, w.abs() > per_filter),
        "RTN starts at the TWN threshold 0.7·E|W| of each filter",
    )
    wr = w.clone().requires_grad_()
    rtn(wr).sum().backward()
    check(
        rtn.k.grad is not None and rtn.b.grad is not None and rtn.k.ndim == 1,
        "RTN k, b (1-d, no weight decay) and α receive gradients",
    )

    fat = FATNNTernary(w)
    q = fat(w)
    check(q.unique().numel() <= 3, "FATNN output has ≤3 levels")
    w_n = (w - w.mean()) / w.std()
    check(
        torch.equal(q != 0, w_n.abs() > 0.5),
        "FATNN init α1 = α2 = 1 gives thresholds ±0.5σ",
    )
    wf = w.clone().requires_grad_()
    fat(wf).sum().backward()
    check(
        fat.a1.grad is not None and fat.a2.grad is not None,
        "FATNN step sizes α1, α2 receive gradients",
    )

    meta = MultiFC()
    mq = MetaQuantTernary(meta)
    opt = torch.optim.Adam(meta.parameters(), lr=1e-3)
    wm = w.clone().requires_grad_()
    out = mq(wm)
    check(torch.equal(out, twn_quantize(w)), "MetaQuant forward is TWN")
    out.sum().backward()
    g = torch.ones_like(w)
    a = (meta.fc2.weight @ meta.fc1.weight).item()
    c = (meta.fc2.weight @ meta.fc1.bias + meta.fc2.bias).item()
    check(
        torch.allclose(wm.grad, g * (a * w + c), atol=1e-6),
        "latent-weight gradient is the meta gradient M(g, W)",
    )
    h = torch.randn(1000, 1)
    full = meta.fc2(meta.fc1(h))
    check(
        torch.allclose(full.squeeze(1), meta(torch.ones(1000), h[:, 0])),
        "MultiFC composed form equals the two-layer network",
    )
    check(
        meta_step([mq], meta, opt, 0.01) is None,
        "no φ update on the first step (no previous gradient)",
    )
    mq(w.clone().requires_grad_()).pow(2).sum().backward()
    before = [p.detach().clone() for p in meta.parameters()]
    meta_step([mq], meta, opt, 0.01)
    check(
        any(not torch.equal(b, p) for b, p in zip(before, meta.parameters())),
        "φ updated from the late-update surrogate on the second step",
    )


def test_hypernetwork():
    print("hypernetwork")
    hn = Hypernetwork()
    check(
        hn.num_params() == 4610,
        f"default HN has 4,610 params (got {hn.num_params()})",
    )
    tau = hn(torch.randn(10, 4))
    check(
        bool((tau[:, 0] > 0).all() and (tau[:, 1] < 0).all()),
        "τ+ > 0 > τ- for every layer",
    )
    check(
        torch.allclose(
            tau, torch.tensor([0.5, -0.5]).expand(10, 2), atol=0.02
        ),
        f"initial thresholds ≈ ±0.5 (got {tau[:, 0].mean():.3f}, "
        f"{tau[:, 1].mean():.3f})",
    )
    big = Hypernetwork()
    big.out.bias.data = torch.tensor([20.0, -20.0])
    tb = big(torch.randn(10, 4))
    check(
        bool((tb[:, 0] > 0).all() and (tb[:, 1] < 0).all()),
        "signs hold even at extreme outputs",
    )
    sym = Hypernetwork(symmetric=True)(torch.randn(5, 4))
    check(
        torch.allclose(sym[:, 0], -sym[:, 1]), "symmetric mode gives τ- = -τ+"
    )


def test_backbone(name, img=32):
    model = build_model(
        name,
        10,
        img if name not in ("vit_s16", "swin_t") else 224,
        pretrained=False,
    )
    n_before = sum(
        1
        for m in model.modules()
        if type(m) in (torch.nn.Conv2d, torch.nn.Linear)
    )
    layers = ternarize(model)
    st = QuantState()
    set_qstate(model, st)
    x = torch.randn(2, 3, 32, 32)
    for layer in layers:
        layer.tau = None
    st.mode = "soft"
    model(x).sum().backward()
    grads = all(m.weight.grad is not None for m in layers)
    st.mode = "hard"
    with torch.no_grad():
        levels = max(m.quantized_weight().unique().numel() for m in layers)
    check(
        len(layers) == n_before - 2 and grads and levels <= 3,
        f"{name}: {len(layers)} ternary layers (stem+head FP), grads flow, "
        "hard weights ≤3 levels",
    )


def test_phi_gradient():
    print("bilevel gradient path")
    import argparse as ap

    from src.training.methods import build_method

    model = build_model("resnet18", 10, 32, pretrained=False)
    # timm zero-inits the last BN of each residual block; with random
    # weights that silences residual branches (zero grads). Pretrained
    # weights don't have this.
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(mod.weight)
    layers = ternarize(model)
    st = QuantState()
    st.mode = "soft"
    set_qstate(model, st)
    args = ap.Namespace(
        hn_width=64,
        hn_depth=2,
        symmetric=False,
        stats="mu,sigma,gamma,rho",
        train_mode="soft",
        lambda1=1e-4,
        lambda2=1e-3,
    )
    m = build_method("ours", layers, args)
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
    m.on_train_forward(True)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    m.on_train_forward(False)
    loss.backward()
    m.after_inner_backward()
    check(
        bool(
            (m.tracker.stats[:, 1] > 0).all()
            and (m.tracker.stats[:, 2] > 0).all()
        ),
        "stats tracker filled σ and γ for every layer",
    )
    tau = m.set_thresholds(connected=False)
    check(not tau.requires_grad, "inner-loop thresholds detached from φ")
    tau = m.set_thresholds(connected=True)
    meta = torch.nn.functional.cross_entropy(model(x), y) + m.regularizer(tau)
    phi = m.phi_params()
    meta.backward(inputs=phi)
    check(
        all(p.grad is not None and p.grad.abs().sum() > 0 for p in phi),
        "outer loss back-propagates to every hypernetwork parameter",
    )
    from src.training.loop import frozen_bn_stats

    bn = next(
        mod for mod in model.modules() if isinstance(mod, torch.nn.BatchNorm2d)
    )
    before = bn.running_mean.clone()
    model.train()
    with frozen_bn_stats(model):
        model(x)
    check(
        torch.equal(bn.running_mean, before),
        "outer (held-out) forward leaves BN running stats unchanged",
    )
    f = m.tracker.features()
    m2 = build_method(
        "ours", layers, ap.Namespace(**{**vars(args), "stats": "mu"})
    )
    m2.tracker.stats.copy_(m.tracker.stats)
    check(
        bool((m2.tracker.features()[:, 1:] == 0).all())
        and torch.equal(m2.tracker.features()[:, 0], f[:, 0]),
        "stat masking zeroes the ablated features only",
    )


def test_paper_tables():
    """MAC count matches paper Tab. 3; paper-layout tables are written."""
    print("paper tables")
    from src.models.backbones import count_macs

    macs = count_macs(build_model("resnet18", 1000, 224, pretrained=False))
    check(
        abs(macs / 1e9 - 1.8) < 0.05,
        f"resnet18 MACs {macs / 1e9:.2f}G ≈ 1.8G (paper Tab. 3)",
    )

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from aggregate import paired, stat
    from latex_tables import write_paper_tables

    def run(ds, model, acc, seed=0):
        return {
            "config": {"seed": seed},
            "test_acc": acc,
            "params": 11_700_000,
            "method_params": 4610,
            "macs": macs,
            "peak_mem_mb": 1000.0,
            "mean_epoch_time_s": 10.0,
        }

    groups = {}
    for ds in ("cifar10", "imagenet"):
        for v, acc in (("fp", 95.0), ("lsq", 90.0), ("ours", 92.0)):
            groups[(ds, "resnet18", v)] = [run(ds, "resnet18", acc)]
        groups[(ds, "resnet18", "fixed")] = [run(ds, "resnet18", 89.0)]
    groups[("cifar10", "resnet18", "ours_k1")] = [run("", "", 91.0)]
    tmp = tempfile.mkdtemp()
    try:
        names = write_paper_tables(tmp, groups, stat, paired)
        check(
            len(names) == 5,
            f"all five paper tables written ({len(names)})",
        )
        t1 = open(os.path.join(tmp, "paper_table1_resnet18.tex")).read()
        check("+2.0" in t1 and "W2/A2" in t1, "Tab. 1 has vs. LSQ and W/A")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_end_to_end(models):
    """Every method trains on random smoke-test data and writes a complete
    run dir."""
    print("end-to-end (random smoke-test data, CPU)")
    import json

    import run
    from scripts.gpu_queue import result_path

    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        for model in models:
            base = [
                "--dataset",
                "smoketest",
                "--model",
                model,
                "--no-pretrained",
                "--limit_batches",
                "2",
                "--no-amp",
                "--num_workers",
                "0",
                "--device",
                "cpu",
            ]
            run.main(base + ["--method", "fp"])
            fp_dir = os.path.join("runs", f"smoketest_{model}_fp_s0")
            need = [
                "cmd.txt",
                "config.json",
                "env.txt",
                "results.json",
                "logs/metrics.jsonl",
                "checkpoints/last.pt",
            ]
            missing = [
                f for f in need if not os.path.exists(os.path.join(fp_dir, f))
            ]
            check(
                not missing,
                f"{model}/fp: run dir has all required artefacts "
                f"{'' if not missing else missing}",
            )
            methods = (
                [
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
                ]
                if model == "resnet18"
                else ["ours"]
            )
            for meth in methods:
                run.main(base + ["--method", meth])
                out = os.path.join(
                    "runs", f"smoketest_{model}_{meth}_s0", "results.json"
                )
                r = json.load(open(out))
                check(
                    0 <= r["test_acc"] <= 100 and len(r["history"]) == 2,
                    f"{model}/{meth}: 2 epochs, results written "
                    f"({r['test_acc']:.1f}%)",
                )
            cmd = "python run.py " + " ".join(base + ["--method", "ours"])
            expect = os.path.join(
                "runs", f"smoketest_{model}_ours_s0", "results.json"
            )
            check(
                os.path.normpath(result_path(cmd)) == os.path.normpath(expect),
                "queue derives the same run dir as run.py",
            )
            # resume: drop results, keep last.pt, rerun with 3 epochs
            run_dir = os.path.join("runs", f"smoketest_{model}_ours_s0")
            os.remove(os.path.join(run_dir, "results.json"))
            run.main(base + ["--method", "ours", "--epochs", "3"])
            r = json.load(open(os.path.join(run_dir, "results.json")))
            check(
                len(r["history"]) == 3,
                f"{model}/ours resumes from checkpoints/last.pt",
            )
            # --config supplies defaults; the CLI overrides them
            with open("cfg.json", "w") as f:
                json.dump(
                    {
                        "dataset": "smoketest",
                        "model": model,
                        "method": "fixed",
                        "fixed_tau": 0.7,
                        "epochs": 1,
                    },
                    f,
                )
            run.main(
                ["--config", "cfg.json", "--fixed_tau", "0.3", "--tag", "_cfg"]
                + base[4:]
            )
            c = json.load(
                open(
                    os.path.join(
                        "runs",
                        f"smoketest_{model}_fixed_cfg_s0",
                        "config.json",
                    )
                )
            )
            check(
                c["fixed_tau"] == 0.3 and c["epochs"] == 1,
                "--config defaults applied, CLI override wins",
            )
        # same seed + --deterministic -> identical training curves
        losses = []
        for out in ("det_a", "det_b"):
            run.main(
                base[:3]
                + ["resnet18"]
                + base[4:]
                + [
                    "--method",
                    "ours",
                    "--deterministic",
                    "--out_dir",
                    out,
                    "--init_from",
                    os.path.join("runs", "smoketest_resnet18_fp_s0"),
                ]
            )
            r = json.load(
                open(
                    os.path.join(
                        out, "smoketest_resnet18_ours_s0", "results.json"
                    )
                )
            )
            losses.append([h["train_loss"] for h in r["history"]])
        check(
            losses[0] == losses[1],
            "--deterministic: two runs, same seed -> identical losses",
        )
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    test_quantizer()
    test_new_baselines()
    test_hypernetwork()
    print("backbones")
    for name in (BACKBONES if a.all else ["resnet18", "mobilenetv2"]):
        test_backbone(name)
    test_phi_gradient()
    test_paper_tables()
    test_end_to_end(["resnet18", "vit_s16"] if a.all else ["resnet18"])
    print(
        f"\n{'ALL PASSED' if not check.failed else f'{check.failed} FAILED'}"
    )
    sys.exit(1 if check.failed else 0)
