#!/bin/bash
# ======================================================================
# One SLURM array task = one `python run.py ...` line from one wave of a
# jobs file. Submitted by scripts/slurm_submit.sh, not by hand:
#
#   sbatch --account=def-xxx --array=1-N scripts/slurm_array.slurm \
#          jobs/main.txt <wave>
#
# Works on any Alliance cluster; the GPU type is set by slurm_submit.sh.
# Outputs: runs/<run_name>/ (see run.py), SLURM logs in slurm_logs/.
# ======================================================================
#SBATCH --job-name=ternary
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
JOBS=$1
WAVE=$2

echo "JOB_ID: ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} (${SLURM_JOB_ID})"
echo "CLUSTER: ${SLURM_CLUSTER_NAME:-} NODELIST: ${SLURM_JOB_NODELIST:-}"
echo "CPUS: ${SLURM_CPUS_PER_TASK:-} GPUS: ${SLURM_GPUS_ON_NODE:-}"
echo "START_TIME: $(date)"

CMD=$(awk -v w="$WAVE" -v n="$SLURM_ARRAY_TASK_ID" '
  /^#/ || NF==0 {next}
  $0=="---" {wave++; next}
  wave==w {i++; if (i==n) {print; exit}}' "$JOBS")
if [ -z "$CMD" ]; then
  echo "no command $SLURM_ARRAY_TASK_ID in wave $WAVE"
  exit 0
fi

module load StdEnv/2023 python/3.11
source ~/envs/ternary/bin/activate
cd "$SLURM_SUBMIT_DIR"

# Skip runs that already finished: resubmitting a jobs file is safe.
OUT=$(python scripts/gpu_queue.py --print-result-path "$CMD")
if [ -n "$OUT" ] && [ -f "$OUT" ]; then
  echo "done already: $OUT"
  exit 0
fi

# Compute nodes have no internet: data from fast node-local disk,
# pretrained weights from the cache filled by scripts/prefetch.py.
mkdir -p "$SLURM_TMPDIR/data"
cp -r ./data/. "$SLURM_TMPDIR/data/"
export HF_HUB_OFFLINE=1
export DATA="$SLURM_TMPDIR/data"

echo "CMD: $CMD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
eval "$CMD --num_workers $SLURM_CPUS_PER_TASK"
echo "END_TIME: $(date)"
