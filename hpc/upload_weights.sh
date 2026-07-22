#!/usr/bin/env bash
# Upload LLM weights from local llm_weights/ to $VSC_SCRATCH/llm_weights on VUB Hydra HPC.
# Run locally in Git Bash from the project root.
# Usage:
#   bash hpc/upload_weights.sh                        # upload all models
#   bash hpc/upload_weights.sh llama-3.2-1B-Instruct  # upload one model

set -e

HPC_USER="vsc11800"
HPC_HOST="login.hpc.vub.be"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/llm_weights"

# Determine which models to upload (all subdirectories by default).
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=()
    for d in "$WEIGHTS_DIR"/*/; do
        MODELS+=("$(basename "$d")")
    done
fi

echo "Models to upload: ${MODELS[*]}"
echo ""

for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="$WEIGHTS_DIR/$MODEL"
    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Not found locally: $MODEL_PATH"
        continue
    fi

    SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    echo "Uploading: $MODEL  ($SIZE)"

    # Create remote directory.
    ssh "${HPC_USER}@${HPC_HOST}" "mkdir -p \$VSC_SCRATCH/llm_weights/${MODEL}"

    # Stream tar over SSH.
    tar -czf - -C "$WEIGHTS_DIR" "$MODEL" \
        | ssh "${HPC_USER}@${HPC_HOST}" \
            "tar -xzf - -C \$VSC_SCRATCH/llm_weights"

    echo "Done: $MODEL"
    echo ""
done

echo "Upload complete."
echo "Verify on HPC:  ls \$VSC_SCRATCH/llm_weights/"
