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
# Per model: 9 array jobs
#   RQ1:  tiny + small + small-linear + medium  (4 jobs × ~12 tasks each)
#   RQ2:  all scenarios in one command file       (1 job  × ~24 tasks)
#   RQ3b: tiny + small + small-linear + medium  (4 jobs × ~15 tasks each)
################################################################################
set -e

cd "$VSC_DATA/NASimLLM"
mkdir -p logs

SCENARIOS=(tiny small small-linear medium)

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

    # RQ1 + RQ3b: one array job per scenario.
    # launch.sh writes scenario-suffixed command files: rq1/commands_<scenario>.txt
    # and rq3/commands_<scenario>.txt (note: dir is 'rq3', job script is rq3b.sh).
    for SCENARIO in "${SCENARIOS[@]}"; do
        if [ -f "$OUT/rq1/commands_${SCENARIO}.txt" ]; then
            N=$(wc -l < "$OUT/rq1/commands_${SCENARIO}.txt")
            sbatch --job-name="rq1_${MODEL}_${SCENARIO}" \
                   --output="logs/${MODEL}_rq1_${SCENARIO}_%A_%a.out" \
                   --error="logs/${MODEL}_rq1_${SCENARIO}_%A_%a.err" \
                   --array="1-${N}" \
                   hpc/jobs/rq1.sh "$OUT/rq1/commands_${SCENARIO}.txt"
            echo "  ✓ rq1[$SCENARIO] → $N tasks"
            MODEL_JOBS=$((MODEL_JOBS + N))
        else
            echo "  ✗ rq1[$SCENARIO] — commands_${SCENARIO}.txt missing, skipping"
        fi

        if [ -f "$OUT/rq3/commands_${SCENARIO}.txt" ]; then
            N=$(wc -l < "$OUT/rq3/commands_${SCENARIO}.txt")
            sbatch --job-name="rq3b_${MODEL}_${SCENARIO}" \
                   --output="logs/${MODEL}_rq3b_${SCENARIO}_%A_%a.out" \
                   --error="logs/${MODEL}_rq3b_${SCENARIO}_%A_%a.err" \
                   --array="1-${N}" \
                   hpc/jobs/rq3b.sh "$OUT/rq3/commands_${SCENARIO}.txt"
            echo "  ✓ rq3b[$SCENARIO] → $N tasks"
            MODEL_JOBS=$((MODEL_JOBS + N))
        else
            echo "  ✗ rq3b[$SCENARIO] — commands_${SCENARIO}.txt missing, skipping"
        fi
    done

    # RQ2: single command file (all scenarios handled internally by run_rq2)
    if [ -f "$OUT/rq2/commands.txt" ]; then
        N=$(wc -l < "$OUT/rq2/commands.txt")
        sbatch --job-name="rq2_${MODEL}" \
               --output="logs/${MODEL}_rq2_%A_%a.out" \
               --error="logs/${MODEL}_rq2_%A_%a.err" \
               --array="1-${N}" \
               hpc/jobs/rq2.sh "$OUT/rq2/commands.txt"
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
echo "Results at:    \$VSC_DATA/runs/<model>/rq1/{scenario}/{condition}/seed{N}/"
echo "               \$VSC_DATA/runs/<model>/rq3/{scenario}/{condition}/seed{N}/"
echo "               \$VSC_DATA/runs/<model>/rq2/{scenario}/{condition}/seed{N}/"
