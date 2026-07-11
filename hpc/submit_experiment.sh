#!/bin/bash
################################################################################
# hpc/submit_experiment.sh — one-shot submission of the thesis experiment.
#
# Design (multi-LLM teacher comparison):
#   RQ1 (effectiveness + which teacher):  ALL teachers  ×  small only
#        - primary teacher also carries the shared baselines
#          (ppo_options / random / bruteforce) so they are trained once.
#        - every other teacher: llm_full only.
#   RQ2 (generalisation):                 panel teachers  ×  all 4 scenarios
#        - primary: llm_full + ppo_options (the no-LLM baseline lives here).
#        - other panel teachers: llm_full only.
#   RQ3 (component ablation):             primary teacher ×  small only.
#
# Safety: a fast GPU gate (hpc/jobs/gate.sh) is submitted first and EVERY array
# job depends on it with afterok — if the environment is broken the backbone is
# cancelled automatically instead of wasting GPU-hours. Skip with --no-gate.
# Preview counts without submitting with --dry-run.
#
# Usage (on HPC, from $VSC_DATA/NASimLLM):
#   bash hpc/submit_experiment.sh              # gate + submit everything
#   bash hpc/submit_experiment.sh --dry-run    # generate command files, show counts
#   bash hpc/submit_experiment.sh --no-gate    # submit without the pre-flight gate
################################################################################
set -e

# ── Experiment configuration ──────────────────────────────────────────────────
PRIMARY="qwen-4B"
RQ1_TEACHERS=(qwen-4B llama-1B llama-3B llama-8B qwen-8B)   # all 5, small only
RQ2_TEACHERS=(qwen-4B llama-3B llama-8B)                    # panel, all scenarios
RQ3_TEACHER="qwen-4B"                                       # primary, small only
SEEDS="0 1 2"
EPISODES=1000        # lowered from 2000: halves wall-time; footprint now dominated by nothing (step-logs off, only final ckpt)
# Shared hyperparameters (match hpc/launch.sh / EXECUTION.md).
HP="--episode-length 500 --lambda-decay 0.97 --lambda-mode competence --lambda-min 0.05 \
--batch-size 64 --num-epochs 4 --llm-mix-weight 0.5"
DEVICE="cuda"

# ── Flags ─────────────────────────────────────────────────────────────────────
GATE=1
DRYRUN=0
for arg in "$@"; do
    case "$arg" in
        --no-gate) GATE=0 ;;
        --dry-run) DRYRUN=1; GATE=0 ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ── Environment ───────────────────────────────────────────────────────────────
# Needed even for --dry-run: the command generators import nasim/gymnasium, so
# the venv must be active regardless of mode. Guards keep it working on a laptop
# (no `module`, no VSC venv → falls back to whatever python is on PATH).
if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null \
      || module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
fi
if [ -n "${VSC_DATA:-}" ] && [ -f "$VSC_DATA/.venvs/nasim/bin/activate" ]; then
    source "$VSC_DATA/.venvs/nasim/bin/activate"
    cd "$VSC_DATA/NASimLLM" 2>/dev/null || true
elif [ -f ".venvs/nasim/bin/activate" ]; then
    source ".venvs/nasim/bin/activate"
elif ! python -c "import nasim" >/dev/null 2>&1; then
    echo "ERROR: nasim virtualenv not found at \$VSC_DATA/.venvs/nasim and 'nasim'"
    echo "       is not importable with the current python."
    echo "       → Run 'bash hpc/setup.sh' first (it creates the venv + installs deps)."
    exit 1
fi
mkdir -p logs

RUNS_ROOT="${VSC_SCRATCH:-runs_root}/runs"
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

# ── Optional pre-flight gate ──────────────────────────────────────────────────
DEP=""
if [ $GATE -eq 1 ]; then
    GATEID=$(sbatch --parsable hpc/jobs/gate.sh)
    echo "Pre-flight gate submitted: job $GATEID"
    echo "  → every experiment array below depends on afterok:$GATEID"
    DEP="--dependency=afterok:$GATEID"
    echo ""
fi

TOTAL=0
# submit_array <jobname> <command_file> <job_script>
submit_array() {
    local name="$1" cmdfile="$2" script="$3" n
    if [ ! -f "$cmdfile" ]; then
        echo "  [skip] $name — $cmdfile not generated"; return
    fi
    n=$(wc -l < "$cmdfile")
    if [ "$n" -eq 0 ]; then
        echo "  [skip] $name — 0 tasks"; return
    fi
    TOTAL=$((TOTAL + n))
    if [ $DRYRUN -eq 1 ]; then
        printf "  [dry-run] %-26s %3d tasks   (%s)\n" "$name" "$n" "$cmdfile"
        return
    fi
    sbatch $DEP --job-name="$name" --array="1-$n" \
        --output="logs/${name}_%A_%a.out" --error="logs/${name}_%A_%a.err" \
        "$script" "$cmdfile"
    printf "  submitted %-26s %3d tasks\n" "$name" "$n"
}

echo "======================================================================"
echo " RQ1 — effectiveness + teacher comparison  (scenario: small)"
echo "======================================================================"
for T in "${RQ1_TEACHERS[@]}"; do
    OUT="$RUNS_ROOT/$T"; MP=$(resolve_model "$T")
    if [ "$T" = "$PRIMARY" ]; then
        CONDS="llm_full ppo_options random bruteforce"
    else
        CONDS="llm_full"
    fi
    python -m nasim.scripts.run_rq1 --scenario small --conditions $CONDS --seeds $SEEDS \
        --out-dir "$OUT/rq1" --device $DEVICE --llama-model "$MP" $HP \
        --episodes $EPISODES --parallel
    submit_array "rq1_${T}_small" "$OUT/rq1/commands_small.txt" hpc/jobs/rq1.sh
done

echo ""
echo "======================================================================"
echo " RQ2 — generalisation across scenarios  (panel: ${RQ2_TEACHERS[*]})"
echo "======================================================================"
for T in "${RQ2_TEACHERS[@]}"; do
    OUT="$RUNS_ROOT/$T"; MP=$(resolve_model "$T")
    if [ "$T" = "$PRIMARY" ]; then
        CONDS="llm_full ppo_options"
    else
        CONDS="llm_full"
    fi
    python -m nasim.scripts.run_rq2 --conditions $CONDS --seeds $SEEDS \
        --out-dir "$OUT/rq2" --device $DEVICE --llama-model "$MP" $HP \
        --episodes $EPISODES --parallel
    submit_array "rq2_${T}" "$OUT/rq2/commands.txt" hpc/jobs/rq2.sh
done

echo ""
echo "======================================================================"
echo " RQ3 — component ablation  (teacher: $RQ3_TEACHER, scenario: small)"
echo "======================================================================"
OUT="$RUNS_ROOT/$RQ3_TEACHER"; MP=$(resolve_model "$RQ3_TEACHER")
python -m nasim.scripts.run_rq3b --scenario small --seeds $SEEDS \
    --out-dir "$OUT/rq3" --device $DEVICE --llama-model "$MP" $HP \
    --episodes $EPISODES --parallel
submit_array "rq3_${RQ3_TEACHER}_small" "$OUT/rq3/commands_small.txt" hpc/jobs/rq3b.sh

echo ""
echo "======================================================================"
echo " Total array tasks: $TOTAL"
if [ $DRYRUN -eq 1 ]; then
    echo " (dry-run — nothing submitted; command files written under $RUNS_ROOT/<teacher>/)"
else
    echo " Monitor:  squeue -u \$USER"
    echo " Results:  \$VSC_DATA/runs/<teacher>/rq{1,2,3}/{scenario}/{condition}/seed{N}/"
    echo " In notebooks, add one RUNS entry per teacher tag (qwen-4B, llama-3B, ...)."
fi
echo "======================================================================"
