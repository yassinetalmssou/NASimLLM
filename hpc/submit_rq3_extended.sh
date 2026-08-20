#!/bin/bash
# Extended RQ3 in one shot: Set A component ablations (runs/<teacher>/rq3/small/),
# Set B hyperparameter sensitivity, and Set C action-history-length sweep (runs/<teacher>/sens/small/).
# 6 seeds, 2000 episodes, gated by hpc/jobs/gate.sh (afterok).
# Usage (on HPC, from $VSC_DATA/NASimLLM):
#   bash hpc/submit_rq3_extended.sh              # gate + submit everything
#   bash hpc/submit_rq3_extended.sh --dry-run    # generate command files, show counts
#   bash hpc/submit_rq3_extended.sh --no-gate    # submit without the pre-flight gate
set -e

A_TEACHERS=(qwen-4B llama-1B llama-3B llama-8B qwen-8B)
B_TEACHERS=(qwen-4B llama-1B qwen-8B)
C_TEACHERS=(qwen-4B)
SEEDS="0 1 2 3 4 5"
EPISODES=2000
# Observability: partial observability + belief-state aggregation (POMDP).
# competence_window defaults to 50 in the trainer (raised from 10 for POMDP stability;
# recorded in every run's config.json).
OBS="--partial-obs --belief-accum"

GATE=1; DRYRUN=0
for arg in "$@"; do
    case "$arg" in
        --no-gate) GATE=0 ;;
        --dry-run) DRYRUN=1; GATE=0 ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# Environment: activate the venv (the generator imports nasim).
if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
      || module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
fi
if [ -n "${VSC_DATA:-}" ] && [ -f "$VSC_DATA/.venvs/nasim/bin/activate" ]; then
    source "$VSC_DATA/.venvs/nasim/bin/activate"
    cd "$VSC_DATA/NASimLLM" 2>/dev/null || true
elif ! python -c "import nasim" >/dev/null 2>&1; then
    echo "ERROR: nasim not importable and no venv found - run 'bash hpc/setup.sh' first."
    exit 1
fi
mkdir -p logs

# Output tree is overridable so a POMDP run can live beside the full-obs runs:
#   RUNS_SUBDIR=runs_pomdp JOBTAG=p_ bash hpc/submit_rq3_extended.sh
RUNS_ROOT="${VSC_SCRATCH:-runs_root}/${RUNS_SUBDIR:-runs}"
WEIGHTS_ROOT="${VSC_SCRATCH:-runs_root}/llm_weights"
resolve_model() {
    case "$1" in
        llama-1B) echo "$WEIGHTS_ROOT/llama-3.2-1B-Instruct" ;;
        llama-3B) echo "$WEIGHTS_ROOT/llama-3.2-3B-Instruct" ;;
        llama-8B) echo "$WEIGHTS_ROOT/llama-8B" ;;
        qwen-4B)  echo "$WEIGHTS_ROOT/Qwen3-4B" ;;
        qwen-8B)  echo "$WEIGHTS_ROOT/qwen-8B" ;;
        *)        echo "$1" ;;
    esac
}

DEP=""
if [ $GATE -eq 1 ]; then
    GATEID=$(sbatch --parsable hpc/jobs/gate.sh)
    echo "Pre-flight gate: job $GATEID (everything below waits on afterok)"
    DEP="--dependency=afterok:$GATEID"
fi

TOTAL=0
submit_cmdfile() {   # $1=jobname $2=command_file
    local name="$1" f="$2" n
    [ -f "$f" ] || { echo "  [skip] $name - no command file"; return; }
    n=$(wc -l < "$f")
    [ "$n" -gt 0 ] || { echo "  [skip] $name - 0 tasks"; return; }
    TOTAL=$((TOTAL + n))
    if [ $DRYRUN -eq 1 ]; then
        printf "  [dry-run] %-22s %3d tasks   (%s)\n" "$name" "$n" "$f"; return
    fi
    sbatch $DEP --export="ALL,SYNC_TO_DATA=${SYNC_TO_DATA:-1}" --job-name="${JOBTAG}$name" --array="1-$n" \
        --output="logs/${JOBTAG}${name}_%A_%a.out" --error="logs/${JOBTAG}${name}_%A_%a.err" \
        hpc/jobs/rq3b.sh "$f"
    printf "  submitted %-22s %3d tasks\n" "$name" "$n"
}

echo "Set A - component ablations (incl. working avoidlist_mask)"
for T in "${A_TEACHERS[@]}"; do
    OUT="$RUNS_ROOT/$T"; MP=$(resolve_model "$T"); mkdir -p "$OUT/rq3"
    if [ "$T" = "qwen-4B" ]; then NAMES=(--names avoidlist_mask); else NAMES=(); fi
    python -m nasim.scripts.gen_rq3_extended --set A --out-dir "$OUT" --model "$MP" \
        --episodes $EPISODES --seeds $SEEDS $OBS "${NAMES[@]}" > "$OUT/rq3/commands_A.txt"
    submit_cmdfile "rq3A_${T}" "$OUT/rq3/commands_A.txt"
done

echo ""
echo "Set B - hyperparameter sensitivity (lambda-schedule + guidance)"
for T in "${B_TEACHERS[@]}"; do
    OUT="$RUNS_ROOT/$T"; MP=$(resolve_model "$T"); mkdir -p "$OUT/sens"
    python -m nasim.scripts.gen_rq3_extended --set B --out-dir "$OUT" --model "$MP" \
        --episodes $EPISODES --seeds $SEEDS $OBS > "$OUT/sens/commands_B.txt"
    submit_cmdfile "sensB_${T}" "$OUT/sens/commands_B.txt"
done

echo ""
echo "Set C - action-history length sweep (hist_4/8/16/32/64)"
for T in "${C_TEACHERS[@]}"; do
    OUT="$RUNS_ROOT/$T"; MP=$(resolve_model "$T"); mkdir -p "$OUT/sens"
    python -m nasim.scripts.gen_rq3_extended --set C --out-dir "$OUT" --model "$MP" \
        --episodes $EPISODES --seeds $SEEDS $OBS > "$OUT/sens/commands_hist.txt"
    submit_cmdfile "sensC_${T}" "$OUT/sens/commands_hist.txt"
done

echo ""
echo "Total array tasks: $TOTAL"
[ $DRYRUN -eq 1 ] && echo "(dry-run - nothing submitted)" || echo "Monitor: squeue -u \$USER"
