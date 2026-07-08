#!/bin/bash
################################################################################
# hpc_download_model.sh — Download LLM model directly from HuggingFace on HPC
#
# Run on the HPC login node (no GPU needed, no job needed).
#
# Usage:
#   bash hpc_download_model.sh <hf-repo-id> [local-name]
#
# Examples:
#   bash hpc_download_model.sh Qwen/Qwen3-14B
#   bash hpc_download_model.sh Qwen/Qwen3-14B qwen-14B
#   bash hpc_download_model.sh meta-llama/Llama-3.2-1B-Instruct llama-1B
#
# The model lands in: $VSC_DATA/llm_weights/<local-name>/
################################################################################

set -e

if [ -z "$1" ]; then
    echo "Usage: bash hpc_download_model.sh <hf-repo-id> [local-name]"
    echo ""
    echo "Common models:"
    echo "  Qwen/Qwen3-4B"
    echo "  Qwen/Qwen3-14B"
    echo "  Qwen/Qwen3-32B"
    echo "  meta-llama/Llama-3.2-1B-Instruct"
    echo "  meta-llama/Llama-3.2-3B-Instruct"
    echo "  meta-llama/Llama-3.1-8B-Instruct"
    exit 1
fi

HF_REPO="$1"
LOCAL_NAME="${2:-$(echo "$HF_REPO" | cut -d'/' -f2)}"
DEST="$VSC_SCRATCH/llm_weights/$LOCAL_NAME"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║        HuggingFace Model Download on HPC                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo "  Repo:        $HF_REPO"
echo "  Destination: $DEST  (VSC_SCRATCH — fast, no DATA quota used)"
echo ""

# Load Python module
module purge
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
  || module load Python/3.10.8-GCCcore-12.2.0

# Activate venv (needs huggingface_hub)
source "$VSC_DATA/.venvs/nasim/bin/activate"

# Check if huggingface_hub is available
python -c "import huggingface_hub" 2>/dev/null || {
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
}

mkdir -p "$DEST"

echo "Downloading $HF_REPO → $DEST"
echo "(This may take a while for large models)"
echo ""

python - <<EOF
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="$HF_REPO",
    local_dir="$DEST",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
)
print(f"\n✓ Downloaded to: $DEST")
EOF

echo ""
echo "✓ Done! Model available at: $DEST"
echo ""
echo "Use in hpc/launch.sh:"
echo "  bash hpc/launch.sh $LOCAL_NAME"
echo ""
echo "Or add to hpc/submit_all.sh MODELS list."
