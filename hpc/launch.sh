#!/bin/bash

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
  || module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null \
  || { echo "ERROR: No compatible Python module found. Run 'module avail Python'."; exit 1; }
if [ -f "$VSC_DATA/.venvs/nasim/bin/activate" ]; then
    source "$VSC_DATA/.venvs/nasim/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VSC_DATA/.venvs/nasim"
    echo "       Run hpc/setup.sh first."
    exit 1
fi

MODEL_NAME="${1:-qwen-4B}"
case "$MODEL_NAME" in
    llama-1B)   _MODEL_PATH="${VSC_SCRATCH}/llm_weights/llama-3.2-1B-Instruct" ;;
    llama-3B)   _MODEL_PATH="${VSC_SCRATCH}/llm_weights/llama-3.2-3B-Instruct" ;;
    llama-8B)   _MODEL_PATH="${VSC_SCRATCH}/llm_weights/llama-8B"               ;;
    llama-70B)  _MODEL_PATH="${VSC_SCRATCH}/llm_weights/llama-70B"              ;;
    qwen-4B)    _MODEL_PATH="${VSC_SCRATCH}/llm_weights/Qwen3-4B"               ;;
    qwen-72B)   _MODEL_PATH="${VSC_SCRATCH}/llm_weights/qwen-72B"               ;;
    qwen-8B)    _MODEL_PATH="${VSC_SCRATCH}/llm_weights/qwen-8B"                ;;
    *)          _MODEL_PATH="$MODEL_NAME"; MODEL_NAME=$(basename "$MODEL_NAME") ;;
esac
LLM_MODEL="${LLM_MODEL:-$_MODEL_PATH}"

EPISODES=2000
SEEDS="0 1 2"
DEVICE="cuda"
SCENARIOS=(tiny small medium)
OUT_ROOT="${OUT_ROOT:-${VSC_SCRATCH:+$VSC_SCRATCH/runs/$MODEL_NAME}}"
OUT_ROOT="${OUT_ROOT:-${SCRATCH:-$HOME/scratch}/runs/$MODEL_NAME}"

echo "NASimLLM Experiment Launcher"
echo ""
echo "Configuration:"
echo "  Model Name:  $MODEL_NAME"
echo "  LLM Model:   $LLM_MODEL"
echo "  Scenarios:   ${SCENARIOS[*]}"
echo "  Episodes:    $EPISODES"
echo "  Seeds:       $SEEDS"
echo "  Device:      $DEVICE"
echo "  Output Root: $OUT_ROOT"
echo ""

mkdir -p "$OUT_ROOT/rq2"
for SCENARIO in "${SCENARIOS[@]}"; do
    mkdir -p "$OUT_ROOT/$SCENARIO/rq1"
    mkdir -p "$OUT_ROOT/$SCENARIO/rq3b"
done
mkdir -p logs

TOTAL_RQ1_ALL=0
for SCENARIO in "${SCENARIOS[@]}"; do
echo "======================================================================"
echo "RQ1 [$SCENARIO]: Baseline Comparisons"
echo "======================================================================"
python -m nasim.scripts.run_rq1 \
  --scenario $SCENARIO \
  --episodes $EPISODES \
  --seeds $SEEDS \
  --out-dir "$OUT_ROOT/$SCENARIO/rq1" \
  --device $DEVICE \
  --llama-model $LLM_MODEL \
  --parallel

if [ -f "$OUT_ROOT/$SCENARIO/rq1/commands.txt" ]; then
    N=$(wc -l < "$OUT_ROOT/$SCENARIO/rq1/commands.txt")
    TOTAL_RQ1_ALL=$((TOTAL_RQ1_ALL + N))
    echo "Generated $N commands for RQ1[$SCENARIO]"
    echo "  Submit: sbatch --array=1-$N hpc/jobs/rq1.sh $OUT_ROOT/$SCENARIO/rq1/commands.txt"
else
    echo "ERROR: Failed to generate RQ1[$SCENARIO] commands"
    exit 1
fi
done

echo ""
echo "======================================================================"
echo "RQ2: Scenario Generalization"
echo "======================================================================"
python -m nasim.scripts.run_rq2 \
  --episodes $EPISODES \
  --seeds $SEEDS \
  --out-dir "$OUT_ROOT/rq2" \
  --device $DEVICE \
  --llama-model $LLM_MODEL \
  --parallel

if [ -f "$OUT_ROOT/rq2/commands.txt" ]; then
    TOTAL_RQ2=$(wc -l < "$OUT_ROOT/rq2/commands.txt")
    echo "Generated $TOTAL_RQ2 commands for RQ2"
    echo "  Submit: sbatch --array=1-$TOTAL_RQ2 hpc/jobs/rq2.sh $OUT_ROOT/rq2/commands.txt"
else
    echo "ERROR: Failed to generate RQ2 commands"
    exit 1
fi

echo ""
TOTAL_RQ3B_ALL=0
for SCENARIO in "${SCENARIOS[@]}"; do
echo "======================================================================"
echo "RQ3b [$SCENARIO]: Component Ablations"
echo "======================================================================"
python -m nasim.scripts.run_rq3b \
  --scenario $SCENARIO \
  --episodes $EPISODES \
  --seeds $SEEDS \
  --out-dir "$OUT_ROOT/$SCENARIO/rq3b" \
  --device $DEVICE \
  --llama-model $LLM_MODEL \
  --parallel

if [ -f "$OUT_ROOT/$SCENARIO/rq3b/commands.txt" ]; then
    N=$(wc -l < "$OUT_ROOT/$SCENARIO/rq3b/commands.txt")
    TOTAL_RQ3B_ALL=$((TOTAL_RQ3B_ALL + N))
    echo "Generated $N commands for RQ3b[$SCENARIO]"
    echo "  Submit: sbatch --array=1-$N hpc/jobs/rq3b.sh $OUT_ROOT/$SCENARIO/rq3b/commands.txt"
else
    echo "ERROR: Failed to generate RQ3b[$SCENARIO] commands"
    exit 1
fi
done

echo ""
echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
echo "Model:  $MODEL_NAME"
echo "Total experiment tasks: $((TOTAL_RQ1_ALL + TOTAL_RQ2 + TOTAL_RQ3B_ALL))"
echo "  - RQ1:  $TOTAL_RQ1_ALL tasks (4 conditions x 3 seeds x ${#SCENARIOS[@]} scenarios)"
echo "  - RQ2:  $TOTAL_RQ2 tasks  (4 scenarios x 3 seeds)"
echo "  - RQ3b: $TOTAL_RQ3B_ALL tasks (5 ablations x 3 seeds x ${#SCENARIOS[@]} scenarios)"
echo ""
echo "Command files:"
for SCENARIO in "${SCENARIOS[@]}"; do
echo "  $OUT_ROOT/$SCENARIO/rq1/commands.txt"
echo "  $OUT_ROOT/$SCENARIO/rq3b/commands.txt"
done
echo "  $OUT_ROOT/rq2/commands.txt"
echo ""
echo "Submit commands:"
for SCENARIO in "${SCENARIOS[@]}"; do
    N1=$(wc -l < "$OUT_ROOT/$SCENARIO/rq1/commands.txt" 2>/dev/null || echo 0)
    N3=$(wc -l < "$OUT_ROOT/$SCENARIO/rq3b/commands.txt" 2>/dev/null || echo 0)
    echo "  sbatch --array=1-$N1  hpc/jobs/rq1.sh  $OUT_ROOT/$SCENARIO/rq1/commands.txt"
    echo "  sbatch --array=1-$N3  hpc/jobs/rq3b.sh $OUT_ROOT/$SCENARIO/rq3b/commands.txt"
done
echo "  sbatch --array=1-$TOTAL_RQ2 hpc/jobs/rq2.sh  $OUT_ROOT/rq2/commands.txt"
echo ""
echo "Or use: bash hpc/submit_all.sh $MODEL_NAME"