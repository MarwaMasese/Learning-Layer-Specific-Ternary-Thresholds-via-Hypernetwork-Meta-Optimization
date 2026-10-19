"""
Minimal multi-GPU job queue for plain (non-SLURM) lab machines.

Runs one command per GPU at a time from a jobs file, so several experiments
share a box without stepping on each other (or on other users' GPUs: pass only
the GPU ids agreed on with them).

    python scripts/gpu_queue.py jobs/main.txt --gpus 0,1,2,3
    # inside tmux so it survives logout:
    tmux new -s queue 'python scripts/gpu_queue.py jobs/main.txt --gpus 2,3'

Jobs file:
    - one shell command per line; blank lines and lines starting with # ignored
    - a line containing only `---` is a barrier: everything above must finish
      before anything below starts (e.g. FP fine-tunes before ternary runs)
    - a job whose runs/<run_name>/results.json already exists is skipped, so
      re-running the same file after a crash resumes the sweep; runs
      themselves resume from checkpoints/last.pt

Logs: queue_logs/<run_name>.log
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def result_path(cmd: str) -> str | None:
    """runs/<run_name>/results.json for a `python run.py ...` command.

    Uses run.py's own parser (including --config files), so naming can
    never drift between the runner and the queue.
    """
    words = shlex.split(cmd)
    if not any(w.endswith("run.py") for w in words):
        return None
    argv = words[
        [i for i, w in enumerate(words) if w.endswith("run.py")][0] + 1 :
    ]
    sys.path.insert(0, ROOT)
    from run import parse_args

    try:
        a = parse_args(argv)
    except SystemExit:
        return None
    return os.path.join(a.out_dir, a.run_name, "results.json")


def load_waves(path):
    waves, cur = [], []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "---":
            if cur:
                waves.append(cur)
            cur = []
        else:
            cur.append(line)
    if cur:
        waves.append(cur)
    return waves


def log(msg):
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def main():
    if (
        len(sys.argv) == 3 and sys.argv[1] == "--print-result-path"
    ):  # used by narval_job.sh
        print(result_path(sys.argv[2]) or "")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument(
        "--gpus", required=True, help="comma-separated GPU ids to use"
    )
    ap.add_argument(
        "--per-gpu", type=int, default=1, help="concurrent jobs per GPU"
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    slots = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    os.makedirs("queue_logs", exist_ok=True)
    failed = []

    for w, wave in enumerate(load_waves(a.jobs)):
        todo = []
        for cmd in wave:
            out = result_path(cmd)
            if out and os.path.exists(out):
                log(f"skip (done): {out}")
            else:
                todo.append(cmd)
        log(
            f"wave {w}: {len(todo)} jobs to run ({len(wave) - len(todo)} done)"
        )
        if a.dry_run:
            for c in todo:
                print("   ", c)
            continue

        running = {}  # slot index -> (Popen, cmd, logfile, t0)
        while todo or running:
            for s, (proc, cmd, lf, t0) in list(running.items()):
                if proc.poll() is not None:
                    dt = (time.time() - t0) / 60
                    if proc.returncode == 0:
                        log(f"done  gpu{slots[s]} {dt:.0f}m  {lf}")
                    else:
                        log(f"FAIL  gpu{slots[s]} rc={proc.returncode}  {lf}")
                        failed.append(cmd)
                    del running[s]
            for s in range(len(slots)):
                if s not in running and todo:
                    cmd = todo.pop(0)
                    out = result_path(cmd)
                    name = (
                        os.path.basename(os.path.dirname(out))
                        if out
                        else f"job{int(time.time())}"
                    )
                    lf = os.path.join("queue_logs", name + ".log")
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=slots[s],
                        PYTHONUNBUFFERED="1",
                    )
                    fh = open(lf, "a")
                    fh.write(
                        f"\n### {datetime.now().isoformat()} "
                        f"gpu{slots[s]}\n### {cmd}\n"
                    )
                    fh.flush()
                    proc = subprocess.Popen(
                        cmd,
                        shell=True,
                        env=env,
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                    )
                    running[s] = (proc, cmd, lf, time.time())
                    log(f"start gpu{slots[s]}  {cmd}")
            time.sleep(10)

    if failed:
        log(f"{len(failed)} job(s) failed:")
        for c in failed:
            print("   ", c)
        sys.exit(1)
    log("all done")


if __name__ == "__main__":
    main()
