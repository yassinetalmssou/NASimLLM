#!/bin/bash
# POMDP validation: qwen-4B on small with the LLM teacher, across three observability
# settings x 2 seeds (6 jobs). Confirms that belief-state aggregation recovers
# full-obs-like performance under partial observability before the full POMDP re-run.
#   full            fully observable (reference, reproduces the current setup)
#   partial_naive   partial observability, memoryless (expected to fail)
#   partial_belief  partial observability + belief-state aggregation (expected ~full)
# Usage (on HPC, from $VSC_DATA/NASimLLM):
#   bash hpc/validate_pomdp.sh            # gate + submit
#   bash hpc/validate_pomdp.sh --dry-run  # write command file, print, submit nothing
set -e

TEACHER="qwen-4B"
SEEDS="0 1"
EPISODES=500
HP="--episode-length 500 --lambda-mode competence --lambda-start 1.0 --lambda-decay 0.97 --lambda-min 0.05 --llm-mix-weight 0.5 --teacher-temp 2.0 --batch-size 64 --num-epochs 4"

GATE=1; DRYRUN=0
for arg in "$@"; do
    case "$arg" in
        --no-gate) GATE=0 ;;
        --dry-run) DRYRUN=1; GATE=0 ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
      || module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
fi
if [ -n "${VSC_DATA:-}" ] && [ -f "$VSC_DATA/.venvs/nasim/bin/activate" ]; then
    source "$VSC_DATA/.venvs/nasim/bin/activate"
    cd "$VSC_DATA/NASimLLM" 2>/dev/null || true
fi
mkdir -p logs

RUNS_ROOT="${VSC_SCRATCH:-runs_root}/runs"
MP="${VSC_SCRATCH:-runs_root}/llm_weights/Qwen3-4B"
OUT="$RUNS_ROOT/$TEACHER/valpomdp"
mkdir -p "$OUT"
PY=$(command -v python)

CMDFILE="$OUT/commands_val.txt"
: > "$CMDFILE"
for cond in full partial_naive partial_belief; do
    case $cond in
        full)           OBS="" ;;
        partial_naive)  OBS="--partial-obs" ;;
        partial_belief) OBS="--partial-obs --belief-accum" ;;
    esac
    for s in $SEEDS; do
        SAVE="$OUT/small/$cond/seed$s/ckpts"
        echo "$PY -m nasim.train_llm4teach --scenario small --episodes $EPISODES --seed $s --device cuda --llama-model $MP $HP --save-dir $SAVE --condition $cond --no-step-logs $OBS" >> "$CMDFILE"
    done
done

N=$(wc -l < "$CMDFILE")
echo "Validation command file: $CMDFILE ($N tasks)"
if [ $DRYRUN -eq 1 ]; then
    cat "$CMDFILE"
    echo "(dry-run - nothing submitted)"
    exit 0
fi

DEP=""
if [ $GATE -eq 1 ]; then
    GATEID=$(sbatch --parsable hpc/jobs/gate.sh)
    echo "Pre-flight gate: job $GATEID (validation array waits on afterok)"
    DEP="--dependency=afterok:$GATEID"
fi
sbatch $DEP --export="ALL,SYNC_TO_DATA=${SYNC_TO_DATA:-0}" --job-name valPOMDP --array="1-$N" \
    --output="logs/valPOMDP_%A_%a.out" --error="logs/valPOMDP_%A_%a.err" \
    hpc/jobs/rq3b.sh "$CMDFILE"
echo "Submitted POMDP validation array (1-$N)."
echo "When done, compare teacher-free deployment:"
echo "  python -m nasim.scripts.eval_policy --root \$VSC_DATA/runs/$TEACHER/valpomdp --episodes 200 --sample"