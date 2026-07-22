#!/bin/bash
#SBATCH --job-name=nasim_rq2
#SBATCH --output=logs/rq2_%A_%a.out
#SBATCH --error=logs/rq2_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=ampere_gpu

# RQ2 array job: run one command per array task from a command file, then sync results.

set -e

module purge
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
  || module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1 2>/dev/null \
  || module load CUDA/11.8.0

cd "$VSC_DATA/NASimLLM"
source $VSC_DATA/.venvs/nasim/bin/activate

if [ -z "$1" ]; then
    echo "ERROR: Missing command file argument"
    echo "Usage: sbatch --array=1-N hpc/jobs/rq2.sh <command_file>"
    exit 1
fi

COMMAND_FILE=$1

if [ ! -f "$COMMAND_FILE" ]; then
    echo "ERROR: Command file not found: $COMMAND_FILE"
    exit 1
fi

TOTAL_COMMANDS=$(wc -l < "$COMMAND_FILE")

if [ $SLURM_ARRAY_TASK_ID -gt $TOTAL_COMMANDS ]; then
    echo "ERROR: Array task ID ($SLURM_ARRAY_TASK_ID) exceeds total commands ($TOTAL_COMMANDS)"
    exit 1
fi

COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMAND_FILE")

# Run the selected command.
echo "Command: $COMMAND"
eval $COMMAND
EXIT_CODE=$?

# Sync results from SCRATCH back to DATA.
RESULTS_SRC="$(dirname "$COMMAND_FILE")"
RESULTS_DST="${RESULTS_SRC/$VSC_SCRATCH/$VSC_DATA}"
if [ -d "$RESULTS_SRC" ]; then
    mkdir -p "$RESULTS_DST"
    rsync -a --info=progress2 "$RESULTS_SRC/" "$RESULTS_DST/"
    echo "Results synced to $RESULTS_DST (also kept on SCRATCH)"
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo "Job failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
