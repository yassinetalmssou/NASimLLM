#!/bin/bash
# Download an LLM from HuggingFace into $VSC_SCRATCH/llm_weights (run on the HPC login node).
# Usage: bash download_model.sh <hf-repo-id> [local-name]
#   e.g. bash download_model.sh Qwen/Qwen3-14B qwen-14B

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
echo "Repo:        $HF_REPO"
echo "Destination: $DEST"

# Load Python and activate the venv (needs huggingface_hub).
module purge
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
  || module load Python/3.10.8-GCCcore-12.2.0
source "$VSC_DATA/.venvs/nasim/bin/activate"
python -c "import huggingface_hub" 2>/dev/null || {
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
}

# Download the model snapshot.
mkdir -p "$DEST"
echo "Downloading $HF_REPO to $DEST (may take a while for large models)"
python - <<EOF
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="$HF_REPO",
    local_dir="$DEST",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
)
print(f"\nDownloaded to: $DEST")
EOF

echo "Done. Model available at: $DEST"
echo "Add $LOCAL_NAME to a teacher list in hpc/submit_experiment.sh to use it."
