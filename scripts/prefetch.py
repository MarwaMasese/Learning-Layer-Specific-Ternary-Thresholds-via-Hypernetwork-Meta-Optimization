"""
Download everything compute nodes need (they have no internet), then verify
it loads OFFLINE exactly as a job will. Run on a login node:

    python scripts/prefetch.py     # CIFAR-10/100 into ./data + timm weights
    python scripts/prefetch.py --models resnet18,vit_s16

Ends with "PREFETCH OK" only if every item loads with HF_HUB_OFFLINE=1.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.backbones import BACKBONES  # noqa: E402

VERIFY = """
import sys, timm, torchvision as tv
sys.path.insert(0, {root!r})
from src.models.backbones import BACKBONES
for name in {models!r}:
    timm.create_model(BACKBONES[name][0], pretrained=True)
    print("  offline ok:", name, flush=True)
tv.datasets.CIFAR10({data!r}, download=False)
tv.datasets.CIFAR100({data!r}, download=False)
print("  offline ok: cifar10, cifar100")
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir", "--data-root", dest="data_dir", default="./data"
    )
    ap.add_argument("--models", default=",".join(BACKBONES))
    a = ap.parse_args()
    models = a.models.split(",")

    import timm
    import torchvision as tv

    print("downloading CIFAR-10/100 ...")
    tv.datasets.CIFAR10(a.data_dir, download=True)
    tv.datasets.CIFAR100(a.data_dir, download=True)
    for name in models:
        print(f"downloading {name} ({BACKBONES[name][0]}) ...", flush=True)
        timm.create_model(BACKBONES[name][0], pretrained=True)

    print("verifying offline loading ...")
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    code = VERIFY.format(root=ROOT, models=models, data=a.data_dir)
    rc = subprocess.call([sys.executable, "-c", code], env=env)
    if rc != 0:
        sys.exit(
            "PREFETCH FAILED: something is not in the local cache "
            "(see error above)"
        )
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"weights cache: {hf}\nPREFETCH OK")


if __name__ == "__main__":
    main()
