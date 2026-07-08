#!/bin/bash
#SBATCH --job-name=nasim_qwen4B_test
#SBATCH --output=logs/qwen4B_test_%j.out
#SBATCH --error=logs/qwen4B_test_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=ampere_gpu
################################################################################
# Single end-to-end test job: all 3 RQs with Qwen 4B
#
# Reduced parameters for pipeline validation:
#   - scenario : tiny  (smallest network)
#   - episodes : 100   (fast sanity check)
#   - seed     : 0     (single run)
#
# Submit from $VSC_DATA/NASimLLM:
#   sbatch submit_qwen4B_test.sh
################################################################################

set -e

echo "========================================================================"
echo "NASimLLM — Qwen 4B End-to-End Test Job"
echo "Started: $(date)"
echo "Node:    ${SLURM_NODELIST:-local}"
echo "========================================================================"

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
  || module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null \
  || { echo "ERROR: No compatible Python module found."; exit 1; }
module load CUDA/12.1.1 2>/dev/null \
  || module load CUDA/11.8.0 2>/dev/null \
  || echo "WARNING: No CUDA module found — GPU may not work."

cd "$VSC_DATA/NASimLLM"
source "$VSC_DATA/.venvs/nasim/bin/activate"

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME="qwen-4B"
LLM_MODEL="$VSC_SCRATCH/llm_weights/Qwen3-4B"
SCENARIO="tiny"
EPISODES=100
SEED=0
DEVICE="cuda"
OUT_ROOT="$VSC_SCRATCH/runs/${MODEL_NAME}_test"

echo ""
echo "Configuration:"
echo "  Model:    $MODEL_NAME  ($LLM_MODEL)"
echo "  Scenario: $SCENARIO"
echo "  Episodes: $EPISODES"
echo "  Seed:     $SEED"
echo "  Output:   $OUT_ROOT"
echo ""

if [ ! -d "$LLM_MODEL" ]; then
    echo "ERROR: Model weights not found at $LLM_MODEL"
    echo "  Download with: bash hpc_download_model.sh Qwen/Qwen3-4B qwen-4B"
    exit 1
fi

mkdir -p logs \
    "$OUT_ROOT/rq1" \
    "$OUT_ROOT/rq3" \
    "$OUT_ROOT/rq2"

# ── RQ1: LLM guidance vs baselines ───────────────────────────────────────────
echo "========================================================================"
echo "RQ1: LLM guidance vs baselines  [scenario=$SCENARIO, episodes=$EPISODES]"
echo "========================================================================"
python -m nasim.scripts.run_rq1 \
    --scenario   "$SCENARIO" \
    --episodes   "$EPISODES" \
    --seeds      "$SEED" \
    --out-dir    "$OUT_ROOT/rq1" \
    --device     "$DEVICE" \
    --llama-model "$LLM_MODEL"
echo "✓ RQ1 done"

# ── RQ2: Scenario generalization ──────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "RQ2: Scenario generalization  [episodes=$EPISODES]"
echo "========================================================================"
python -m nasim.scripts.run_rq2 \
    --episodes   "$EPISODES" \
    --seeds      "$SEED" \
    --out-dir    "$OUT_ROOT/rq2" \
    --device     "$DEVICE" \
    --llama-model "$LLM_MODEL"
echo "✓ RQ2 done"

# ── RQ3b: Component ablations ─────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "RQ3b: Component ablations  [scenario=$SCENARIO, episodes=$EPISODES]"
echo "========================================================================"
python -m nasim.scripts.run_rq3b \
    --scenario   "$SCENARIO" \
    --episodes   "$EPISODES" \
    --seeds      "$SEED" \
    --out-dir    "$OUT_ROOT/rq3" \
    --device     "$DEVICE" \
    --llama-model "$LLM_MODEL"
echo "✓ RQ3b done"

# ── Sync results to DATA ──────────────────────────────────────────────────────
echo ""
echo "Syncing results to VSC_DATA..."
DATA_OUT="${OUT_ROOT/$VSC_SCRATCH/$VSC_DATA}"
mkdir -p "$DATA_OUT"
rsync --archive --update "$OUT_ROOT/" "$DATA_OUT/"
echo "✓ Results at: $DATA_OUT"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "All RQs completed successfully for Qwen 4B (test run)"
echo "End: $(date)"
echo "========================================================================"
