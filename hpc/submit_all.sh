#!/bin/bash
################################################################################
# Submit all 3 RQs × 3 scenarios for each LLM model
#
# Usage (on HPC login node, from $VSC_DATA/NASimLLM):
#   bash submit_all_models.sh                        # all 5 thesis models
#   bash submit_all_models.sh llama-1B qwen-4B       # specific models only
#
# Models (aliases defined in launch_all_experiments.sh):
#   llama-1B  →  llm_weights/llama-3.2-1B-Instruct
#   llama-3B  →  llm_weights/llama-3.2-3B-Instruct
#   llama-8B  →  llm_weights/llama-8B
#   qwen-4B   →  llm_weights/Qwen3-4B
#   qwen-8B   →  llm_weights/qwen-8B
#
# Per model: 7 array jobs
#   RQ1: tiny + small + medium  (3 jobs × ~12 tasks each)
#   RQ2: all scenarios internal (1 job  × ~12 tasks)
#   RQ3b: tiny + small + medium (3 jobs × ~15 tasks each)
################################################################################
set -e

cd "$VSC_DATA/NASimLLM"
mkdir -p logs

SCENARIOS=(tiny small medium)

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(llama-1B llama-3B llama-8B qwen-4B qwen-8B)
fi

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║        NASimLLM Multi-Model Job Submission                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo "Models:    ${MODELS[*]}"
echo "Scenarios: ${SCENARIOS[*]}"
echo ""

TOTAL_JOBS=0

for MODEL in "${MODELS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Generating commands for: $MODEL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    bash hpc/launch.sh "$MODEL"

    OUT="$VSC_SCRATCH/runs/$MODEL"
    MODEL_JOBS=0

    # RQ1 + RQ3b: one array job per scenario
    for SCENARIO in "${SCENARIOS[@]}"; do
        if [ -f "$OUT/$SCENARIO/rq1/commands.txt" ]; then
            N=$(wc -l < "$OUT/$SCENARIO/rq1/commands.txt")
            sbatch --job-name="rq1_${MODEL}_${SCENARIO}" \
                   --output="logs/${MODEL}_rq1_${SCENARIO}_%A_%a.out" \
                   --error="logs/${MODEL}_rq1_${SCENARIO}_%A_%a.err" \
                   --array="1-${N}" \
                   hpc/jobs/rq1.sh "$OUT/$SCENARIO/rq1/commands.txt"
            echo "  ✓ rq1[$SCENARIO] → $N tasks"
            MODEL_JOBS=$((MODEL_JOBS + N))
        else
            echo "  ✗ rq1[$SCENARIO] — commands.txt missing, skipping"
        fi

        if [ -f "$OUT/$SCENARIO/rq3b/commands.txt" ]; then
            N=$(wc -l < "$OUT/$SCENARIO/rq3b/commands.txt")
            sbatch --job-name="rq3b_${MODEL}_${SCENARIO}" \
                   --output="logs/${MODEL}_rq3b_${SCENARIO}_%A_%a.out" \
                   --error="logs/${MODEL}_rq3b_${SCENARIO}_%A_%a.err" \
                   --array="1-${N}" \
                   hpc/jobs/rq3b.sh "$OUT/$SCENARIO/rq3b/commands.txt"
            echo "  ✓ rq3b[$SCENARIO] → $N tasks"
            MODEL_JOBS=$((MODEL_JOBS + N))
        else
            echo "  ✗ rq3b[$SCENARIO] — commands.txt missing, skipping"
        fi
    done

    # RQ2: single job (handles scenarios internally)
    if [ -f "$OUT/rq2/commands.txt" ]; then
        N=$(wc -l < "$OUT/rq2/commands.txt")
        sbatch --job-name="rq2_${MODEL}" \
               --output="logs/${MODEL}_rq2_%A_%a.out" \
               --error="logs/${MODEL}_rq2_%A_%a.err" \
               --array="1-${N}" \
               submit_rq2.sh "$OUT/rq2/commands.txt"
        echo "  ✓ rq2[all scenarios] → $N tasks"
        MODEL_JOBS=$((MODEL_JOBS + N))
    else
        echo "  ✗ rq2 — commands.txt missing, skipping"
    fi

    TOTAL_JOBS=$((TOTAL_JOBS + MODEL_JOBS))
    echo "  → $MODEL: $MODEL_JOBS tasks submitted"
    echo ""
done

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    All jobs submitted!                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo "Total tasks submitted: $TOTAL_JOBS"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Results at:    \$VSC_DATA/runs/<model>/{tiny,small,medium}/rq{1,3b}/"
echo "               \$VSC_DATA/runs/<model>/rq2/"
