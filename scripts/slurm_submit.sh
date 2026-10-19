#!/bin/bash
# ======================================================================
# Submit a whole jobs file to SLURM (Fir, Narval, or any Alliance cluster).
# Each wave (separated by `---`) becomes one job array; each wave starts
# after the previous one has finished (afterany: a ternary run whose FP
# checkpoint is missing exits immediately with a clear message).
#
#   bash scripts/slurm_submit.sh jobs/main.txt def-yourprof
#
# Optional environment variables:
#   TIME=0:20:00        time limit per task (default: 8 h, from the script)
#   GPU=a100            GPU type (default h100, as Fir requires a type)
#   NOTIFY_EMAIL=you@uni.ca  email when each wave ends or fails
#                       (not MAIL: Linux already sets MAIL to a mailbox path)
#
# Finished runs are skipped (runs/<name>/results.json exists), so it is safe
# to resubmit the same file after failures or time-outs; unfinished runs
# resume from runs/<name>/checkpoints/last.pt.
# ======================================================================
set -euo pipefail
JOBS=$1
ACCOUNT=$2
mkdir -p slurm_logs

n_waves=$(awk '/^#/ || NF==0 {next} $0=="---" {w++} END {print w+1}' "$JOBS")
dep=""
for ((w = 0; w < n_waves; w++)); do
  n=$(awk -v w="$w" '/^#/ || NF==0 {next} $0=="---" {wave++; next}
                     wave==w {i++} END {print i+0}' "$JOBS")
  [ "$n" -eq 0 ] && continue
  opts=(--parsable --account="$ACCOUNT" --array=1-"$n"
        --gpus-per-node="${GPU:-h100}:1")
  [ -n "${TIME:-}" ] && opts+=(--time="$TIME")
  [ -n "${NOTIFY_EMAIL:-}" ] && opts+=(--mail-user="$NOTIFY_EMAIL" --mail-type=END,FAIL)
  [ -n "$dep" ] && opts+=("$dep")
  id=$(sbatch "${opts[@]}" scripts/slurm_array.slurm "$JOBS" "$w")
  echo "wave $w: $n tasks -> job $id"
  dep="--dependency=afterany:$id"
done
echo "check progress with: sq   (or squeue -u \$USER)"
