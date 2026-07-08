#!/bin/bash
#SBATCH --job-name=nasim_gate
#SBATCH --output=logs/gate_%j.out
#SBATCH --error=logs/gate_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=ampere_gpu
################################################################################
# hpc/jobs/gate.sh — fast pre-flight gate before committing GPU-hours.
#
# Purpose: catch a broken environment (venv/torch/transformers/bitsandbytes,
# wrong module, no GPU, missing weights) in ~20 min instead of at hour 20 of a
# 24h array job. submit_experiment.sh chains the real jobs with
# --dependency=afterok on THIS job, so if any gate fails the backbone never runs.
#
# Gate 1 — the harness trains + evaluates WITHOUT the LLM (env + torch + PPO).
# Gate 2 — the LLM actually loads in 4-bit on the GPU (llama-1B = fastest).
#          We assert config.json shows use_llm=true; the trainer silently falls
#          back to no-LLM if the model fails to load, and that flag exposes it.
#
# Submit standalone:  sbatch hpc/jobs/gate.sh
################################################################################
set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
  || module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1 2>/dev/null \
  || module load CUDA/11.8.0

cd "$VSC_DATA/NASimLLM"
source "$VSC_DATA/.venvs/nasim/bin/activate"
mkdir -p logs

echo "======================================================================"
echo " NASimLLM pre-flight gate   |   $(date)   |   node ${SLURM_NODELIST:-local}"
echo "======================================================================"

# ── Gate 1: harness without the LLM ──────────────────────────────────────────
echo ""
echo "--- GATE 1: harness trains + evaluates without LLM ---"
rm -rf runs/_gate/smoke
python -m nasim.train_llm4teach --scenario tiny --no-llm --episodes 5 \
    --episode-length 200 --save-dir runs/_gate/smoke/ckpts --no-step-logs
python -m nasim.scripts.eval_policy --run-dir runs/_gate/smoke --episodes 10 --device cpu
if [ ! -f runs/_gate/smoke/eval_summary.json ]; then
    echo "GATE 1 FAIL: eval_summary.json was not produced."
    exit 1
fi
echo "GATE 1 PASS"

# ── Gate 2: LLM loads in 4-bit on the GPU ────────────────────────────────────
echo ""
echo "--- GATE 2: LLM loads in 4-bit (llama-1B) ---"
GATE2_MODEL="$VSC_SCRATCH/llm_weights/llama-3.2-1B-Instruct"
if [ ! -d "$GATE2_MODEL" ]; then
    echo "GATE 2 FAIL: weights not found at $GATE2_MODEL"
    exit 1
fi
rm -rf runs/_gate/llm
python -m nasim.train_llm4teach --scenario tiny --seed 0 --episodes 5 --episode-length 200 \
    --lambda-mode competence --llm-mix-weight 0.5 \
    --llama-model "$GATE2_MODEL" \
    --device cuda --save-dir runs/_gate/llm/ckpts --no-step-logs
python - "runs/_gate/llm/config.json" <<'PY'
import json, sys
path = sys.argv[1]
try:
    cfg = json.load(open(path, encoding="utf-8"))
except FileNotFoundError:
    print(f"GATE 2 FAIL: {path} not written — training crashed before config."); sys.exit(1)
if not cfg.get("use_llm", False):
    print("GATE 2 FAIL: config use_llm=false → the LLM did NOT initialise.")
    print("            Check bitsandbytes / transformers / CUDA in the venv.")
    sys.exit(1)
print("GATE 2 PASS: config use_llm=true, LLM initialised on GPU.")
PY

echo ""
echo "======================================================================"
echo " ALL GATES PASSED — environment is sound, backbone may run."
echo "======================================================================"
