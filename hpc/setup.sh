#!/usr/bin/env bash
################################################################################
# hpc/setup.sh — One-time environment setup on VUB Hydra HPC
#
# Run this script ONCE on the HPC after uploading the project:
#   ssh vsc11800@login.hpc.vub.be
#   cd $VSC_DATA/NASimLLM
#   bash hpc/setup.sh
#
# What this script does:
#   1. Loads required cluster modules (Python + CUDA)
#   2. Creates a virtual environment at $HOME/.venvs/nasim
#   3. Installs all Python dependencies (CPU first, then GPU torch)
#   4. Installs the nasim package in editable mode
#   5. Verifies LLM weights are present in $VSC_DATA/llm_weights
#
# Re-run anytime requirements.txt changes.
################################################################################

set -e

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
# $HOME has only 6 GB — store the venv in $VSC_DATA (500 TB) instead.
VENV_DIR="$VSC_DATA/.venvs/nasim"
LLM_WEIGHTS_SRC="$VSC_SCRATCH/llm_weights"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           NASimLLM — HPC Environment Setup                   ║"
echo "║                  VUB Hydra (VSC)                             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Project dir : $PROJECT_DIR"
echo "Venv        : $VENV_DIR"
echo "LLM weights : $LLM_WEIGHTS_SRC"
echo ""

# ── 1. Load modules ───────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/5 — Loading cluster modules"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

module purge

# Check which Python versions are available:  module avail Python
# Update the line below if a newer version is listed.
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
  || module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null \
  || { echo "ERROR: No compatible Python module found. Run 'module avail Python' and update this script."; exit 1; }

# Check which CUDA versions are available:  module avail CUDA
module load CUDA/12.1.1 2>/dev/null \
  || module load CUDA/11.8.0 2>/dev/null \
  || { echo "WARNING: No CUDA module found — GPU training will not work. Continuing anyway."; }

echo "✓ Modules loaded"
python --version
echo ""

# ── 2. Create virtual environment ─────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/5 — Creating virtual environment at $VENV_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$(dirname "$VENV_DIR")"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel
echo "✓ Virtual environment ready"
echo ""

# ── 3. Install PyTorch with CUDA ─────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/5 — Installing PyTorch (CUDA 12.1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Adjust --index-url if the loaded CUDA version is 11.8 → use cu118
pip install torch --index-url https://download.pytorch.org/whl/cu121
echo "✓ PyTorch installed"
echo ""

# ── 4. Install project dependencies ──────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4/5 — Installing project dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_DIR"
pip install -r requirements.txt
pip install -e .
echo "✓ Dependencies installed"
echo ""

# ── 5. Verify LLM weights in $VSC_SCRATCH ─────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5/5 — Checking LLM weights in \$VSC_SCRATCH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Weights live permanently in $VSC_SCRATCH for fast I/O and to save DATA quota.
# If they ever disappear (scratch is not guaranteed), re-download with:
#   bash hpc_download_model.sh <hf-repo-id> [local-name]

if [ ! -d "$LLM_WEIGHTS_SRC" ]; then
    echo "WARNING: $LLM_WEIGHTS_SRC does not exist yet."
    echo "  → Download or upload model weights to \$VSC_SCRATCH/llm_weights/ first:"
    echo "    bash hpc_download_model.sh Qwen/Qwen3-4B qwen-4B"
    echo ""
else
    echo "✓ LLM weights found in \$VSC_SCRATCH:"
    ls "$LLM_WEIGHTS_SRC"
    echo ""
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "First, validate the whole pipeline end-to-end with the smoke-test job:"
echo "  sbatch hpc/jobs/test_qwen4B.sh        # tiny, 100 ep, 1 seed — writes runs/qwen-4B_test/"
echo ""
echo "To generate the full experiment command files for a model:"
echo "  cd $PROJECT_DIR"
echo "  bash hpc/launch.sh qwen-4B            # prints the exact sbatch lines to run"
echo ""
echo "Then submit the generated job arrays (paths printed by launch.sh):"
echo "  sbatch --array=1-N hpc/jobs/rq1.sh  \$VSC_SCRATCH/runs/qwen-4B/rq1/commands_tiny.txt"
echo "  sbatch --array=1-N hpc/jobs/rq2.sh  \$VSC_SCRATCH/runs/qwen-4B/rq2/commands.txt"
echo "  sbatch --array=1-N hpc/jobs/rq3b.sh \$VSC_SCRATCH/runs/qwen-4B/rq3/commands_tiny.txt"
echo "  # …or submit everything at once:  bash hpc/submit_all.sh qwen-4B"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo "  tail -f logs/rq1_<jobid>_<taskid>.out"
