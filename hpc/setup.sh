#!/usr/bin/env bash
# One-time environment setup on VUB Hydra HPC: loads modules, creates the venv at
# $VSC_DATA/.venvs/nasim, installs deps + the nasim package, and checks LLM weights.
# Run once after uploading the project (re-run when requirements.txt changes):
#   bash hpc/setup.sh

set -e

# Paths.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
VENV_DIR="$VSC_DATA/.venvs/nasim"
LLM_WEIGHTS_SRC="$VSC_SCRATCH/llm_weights"

echo "Project dir : $PROJECT_DIR"
echo "Venv        : $VENV_DIR"
echo "LLM weights : $LLM_WEIGHTS_SRC"
echo ""

# Load cluster modules (Python + CUDA).
module purge
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
  || module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null \
  || { echo "ERROR: No compatible Python module found. Run 'module avail Python' and update this script."; exit 1; }
module load CUDA/12.1.1 2>/dev/null \
  || module load CUDA/11.8.0 2>/dev/null \
  || { echo "WARNING: No CUDA module found - GPU training will not work. Continuing anyway."; }
python --version
echo ""

# Create the virtual environment.
mkdir -p "$(dirname "$VENV_DIR")"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel

# Install PyTorch (CUDA 12.1; use cu118 if the loaded CUDA is 11.8).
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies.
cd "$PROJECT_DIR"
pip install -r requirements.txt
pip install -e .

# Check LLM weights in $VSC_SCRATCH.
if [ ! -d "$LLM_WEIGHTS_SRC" ]; then
    echo "WARNING: $LLM_WEIGHTS_SRC does not exist yet."
    echo "  Download or upload model weights to \$VSC_SCRATCH/llm_weights/ first:"
    echo "    bash hpc_download_model.sh Qwen/Qwen3-4B qwen-4B"
    echo ""
else
    echo "LLM weights found in \$VSC_SCRATCH:"
    ls "$LLM_WEIGHTS_SRC"
    echo ""
fi

# Setup complete - next steps.
echo ""
echo "Setup complete."
echo ""
echo "Submit the main experiment (gate + RQ1/RQ2/RQ3 arrays):"
echo "  cd $PROJECT_DIR"
echo "  bash hpc/submit_experiment.sh --dry-run   # check task counts"
echo "  bash hpc/submit_experiment.sh"
echo ""
echo "Submit the extended RQ3 study (ablations + sensitivity):"
echo "  bash hpc/submit_rq3_extended.sh"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo "  tail -f logs/rq1_<jobid>_<taskid>.out"
