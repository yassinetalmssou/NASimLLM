#!/bin/bash
#SBATCH --job-name=nasim_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
# Evaluation is CPU-only, so no GPU directive.

# Held-out evaluation: score trained policies and a random baseline, then aggregate.
# Usage: sbatch hpc/jobs/eval.sh <RUNS_ROOT> [EVAL_EPISODES]

set -euo pipefail

RUNS_ROOT="${1:-runs}"
EVAL_EPISODES="${2:-200}"
EVAL_SEED=1000

# Load modules and activate the project venv (fall back to a local venv off-cluster).
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

# Step 1: evaluate trained policies (checkpoint).
python -m nasim.scripts.eval_policy \
    --root "${RUNS_ROOT}" \
    --episodes "${EVAL_EPISODES}" \
    --eval-seed "${EVAL_SEED}" \
    --device cpu \
    --policy checkpoint \
    --summary-name eval_summary.json

# Step 2: evaluate random baseline.
python -m nasim.scripts.eval_policy \
    --root "${RUNS_ROOT}" \
    --episodes "${EVAL_EPISODES}" \
    --eval-seed "${EVAL_SEED}" \
    --device cpu \
    --policy random \
    --summary-name eval_summary_random.json

# Step 3: aggregate into results.csv and results_random.csv.
python -m nasim.scripts.build_results_table \
    --root "${RUNS_ROOT}" \
    --output results.csv \
    --summary-name eval_summary.json

python -m nasim.scripts.build_results_table \
    --root "${RUNS_ROOT}" \
    --output results_random.csv \
    --summary-name eval_summary_random.json

echo "Done. Wrote results.csv and results_random.csv"
