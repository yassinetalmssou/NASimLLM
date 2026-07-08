#!/usr/bin/env bash
################################################################################
# hpc_upload.sh — Upload NASimLLM project to VUB Hydra HPC
#
# Run this script LOCALLY (on Windows use Git Bash / WSL).
# LLM weights are NOT uploaded here — place them manually in $VSC_DATA/llm_weights
# on the HPC after uploading.
#
# Usage (from project root):
#   bash hpc_upload.sh
#
# Prerequisites (local):
#   - rsync installed (Git Bash, WSL, or Cygwin on Windows)
#   - SSH key configured for vsc11800@login.hpc.vub.be
################################################################################

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
HPC_USER="vsc11800"
HPC_HOST="login.hpc.vub.be"
HPC_DEST="\$VSC_DATA/NASimLLM"   # evaluated on the remote side
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          NASimLLM → VUB Hydra HPC Upload                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Local source : $SCRIPT_DIR"
echo "  Remote target: ${HPC_USER}@${HPC_HOST}:${HPC_DEST}"
echo ""
echo "  LLM weights are excluded. Upload them separately:"
echo "    rsync -avz --progress llm_weights/ \\"
echo "      ${HPC_USER}@${HPC_HOST}:\${VSC_DATA}/llm_weights/"
echo ""

# ── Create remote directory ────────────────────────────────────────────────────
ssh "${HPC_USER}@${HPC_HOST}" "mkdir -p \$VSC_DATA/NASimLLM"

# ── Upload project via tar | ssh (no rsync needed) ────────────────────────────
echo "Archiving and uploading (this may take a moment)..."
tar -czf - \
    --exclude=./llm_weights \
    --exclude=./.venv \
    --exclude=./runs \
    --exclude=./__pycache__ \
    --exclude=./nasim.egg-info \
    --exclude=./.ipynb_checkpoints \
    --exclude=./dist \
    --exclude=./build \
    --exclude=./logs \
    --exclude=./.git \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    -C "${SCRIPT_DIR}" . \
    | ssh "${HPC_USER}@${HPC_HOST}" "tar -xzf - -C \$VSC_DATA/NASimLLM"

echo ""
echo "✓ Upload complete."
echo ""
echo "Next steps on the HPC:"
echo "  1. ssh ${HPC_USER}@${HPC_HOST}"
echo "  2. cd \$VSC_DATA/NASimLLM"
echo "  3. bash hpc/setup.sh          # one-time environment setup"
  echo "  4. bash hpc/launch.sh         # generate job commands"
echo "  5. Submit jobs via sbatch (see launch output)"
