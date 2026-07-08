#!/bin/bash
#SBATCH --job-name=nasim_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
# NOTE: no --gres/--gpus directive — evaluation is CPU-only

# ── Usage ───────────────────────────────────────────────────────────────────
# sbatch hpc/jobs/eval.sh <RUNS_ROOT> [EVAL_EPISODES]
#
# Positional arguments
#   $1  RUNS_ROOT    — directory tree that contains the run directories
#                      (default: runs/)
#   $2  EVAL_EPISODES — episodes per run (default: 200)
#
# Outputs written into each run directory:
#   eval.csv, eval_summary.json          (trained policy)
#   eval_random.csv, eval_summary_random.json (random baseline)
# Then the aggregator is called twice, producing:
#   results.csv           (trained-policy rows)
#   results_random.csv    (random-baseline rows)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RUNS_ROOT="${1:-runs}"
EVAL_EPISODES="${2:-200}"
EVAL_SEED=1000

echo "============================================================"
echo " NASimLLM held-out evaluation"
echo " RUNS_ROOT      : ${RUNS_ROOT}"
echo " EVAL_EPISODES  : ${EVAL_EPISODES}"
echo " EVAL_SEED      : ${EVAL_SEED}"
echo " SLURM_JOB_ID   : ${SLURM_JOB_ID:-local}"
echo " Date           : $(date)"
echo "============================================================"

# Load cluster modules + activate the project virtual environment.
# Matches hpc/setup.sh (venv at $VSC_DATA/.venvs/nasim) and the training job scripts.
# Falls back to a local .venv/venv when run outside SLURM (e.g. on a laptop).
if command -v module >/dev/null 2>&1; then
    module purge
    module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
      || module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
fi

if [ -n "${VSC_DATA:-}" ] && [ -f "$VSC_DATA/.venvs/nasim/bin/activate" ]; then
    source "$VSC_DATA/.venvs/nasim/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "WARNING: no virtualenv found (looked for \$VSC_DATA/.venvs/nasim, .venv, venv)."
    echo "         Using the current 'python' on PATH."
fi

mkdir -p logs

# ── Step 1: evaluate trained policies (checkpoint) ────────────────────────────
echo ""
echo "--- Step 1: Evaluating trained policies (checkpoint) ---"
python -m nasim.scripts.eval_policy \
    --root "${RUNS_ROOT}" \
    --episodes "${EVAL_EPISODES}" \
    --eval-seed "${EVAL_SEED}" \
    --device cpu \
    --policy checkpoint \
    --summary-name eval_summary.json

echo "--- Checkpoint evaluation complete ---"

# ── Step 2: evaluate random baseline ─────────────────────────────────────────
echo ""
echo "--- Step 2: Evaluating random baseline ---"
python -m nasim.scripts.eval_policy \
    --root "${RUNS_ROOT}" \
    --episodes "${EVAL_EPISODES}" \
    --eval-seed "${EVAL_SEED}" \
    --device cpu \
    --policy random \
    --summary-name eval_summary_random.json

echo "--- Random baseline evaluation complete ---"

# ── Step 3: aggregate into results.csv ───────────────────────────────────────
echo ""
echo "--- Step 3: Aggregating results (checkpoint) ---"
python -m nasim.scripts.build_results_table \
    --root "${RUNS_ROOT}" \
    --output results.csv \
    --summary-name eval_summary.json

echo "--- Aggregating results (random baseline) ---"
python -m nasim.scripts.build_results_table \
    --root "${RUNS_ROOT}" \
    --output results_random.csv \
    --summary-name eval_summary_random.json

echo ""
echo "============================================================"
echo " All done. Outputs:"
echo "   results.csv"
echo "   results_random.csv"
echo "============================================================"
