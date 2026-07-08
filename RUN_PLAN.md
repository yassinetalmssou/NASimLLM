# Run Plan — NASimLLM Evaluation Pipeline

## Task 0 — RQ3 ablation wiring (findings + fixes)

### Finding
All five ablation conditions were **correctly wired** in `nasim/llm/llm4teach_advisor.py`
and `nasim/train_llm4teach.py` from the outset:

| Condition | Flag | Effect |
|---|---|---|
| `full` | *(none)* | LLM called every step with full history + avoid list |
| `no_history` | `--no-history` | `_build_history()` returns `""` |
| `no_avoidlist` | `--no-avoidlist` | AVOID block skipped in `_build_history()` |
| `verbose_prompt` | `--verbose-prompt` | Trainer uses `state_summary` (verbose) instead of `compact_state_summary()` |
| `llm_cached` | `--llm-call-freq cached` | LLM responses cached; `use_cache=True` in `collect_rollout` |

### Gap identified
No `condition` field was written to `config.json`, making downstream aggregation
fragile (path-string parsing).

### Fixes applied
1. **`nasim/train_llm4teach.py`** — added `condition: str = ""` parameter; auto-detects
   from `use_llm` when blank; writes `"condition"` into `config.json`.
2. **`nasim/scripts/run_rq3b.py`** — `build_command()` now passes `--condition <ablation_name>`.
3. **`nasim/scripts/run_rq1.py`** — `run_trainer_condition()` now passes `condition=condition`
   to `LLM4TeachTrainer`.

---

## New file locations

| File | Purpose |
|---|---|
| `nasim/scripts/eval_policy.py` | Held-out evaluator (no LLM, CPU-only) |
| `nasim/scripts/build_results_table.py` | Walk tree → long-format `results.csv` |
| `analysis/plot_thesis_figures.py` | Three thesis figures (RQ1/2/3) |
| `hpc/jobs/eval.sh` | SLURM job: eval → aggregate |

---

## Output directory structure

All results live under a **run tag** (experiment group name, e.g. `qwen4B-v1`):

```
runs/
  {tag}/
    rq1/
      {scenario}/        ← tiny, small, small-linear, medium
        {condition}/     ← llm_full, ppo_options, random, bruteforce
          seed{N}/
            config.json          ← includes "condition" field
            train.csv
            ckpts/
              ckpt_final.pt
            eval_summary.json    ← written by eval_policy.py
            eval.csv

    rq2/
      {scenario}/        ← tiny, small, small-linear, medium (all in one run)
        seed{N}/
          ...same layout...

    rq3/
      {scenario}/        ← tiny (primary ablation scenario)
        {condition}/     ← full, no_history, no_avoidlist, verbose_prompt, llm_cached
          seed{N}/
            ...same layout...
```

**Generating commands** (parallel/HPC mode):
```bash
# RQ1 (one scenario at a time → one command file per scenario)
python -m nasim.scripts.run_rq1 --scenario tiny --episodes 300 --seeds 0 1 2 3 4 \
    --out-dir runs/<tag>/rq1 --parallel
# produces: runs/<tag>/rq1/commands.txt → sbatch --array=1-10 hpc/jobs/rq1.sh runs/<tag>/rq1/commands.txt

# RQ2 (all scenarios in one script)
python -m nasim.scripts.run_rq2 --episodes 300 --seeds 0 1 2 3 4 \
    --out-dir runs/<tag>/rq2 --parallel
# produces: runs/<tag>/rq2/commands.txt → sbatch --array=1-20 hpc/jobs/rq2.sh runs/<tag>/rq2/commands.txt

# RQ3 (one scenario, all ablations)
python -m nasim.scripts.run_rq3b --scenario tiny --episodes 300 --seeds 0 1 2 3 4 \
    --out-dir runs/<tag>/rq3 --parallel
# produces: runs/<tag>/rq3/commands.txt → sbatch --array=1-25 hpc/jobs/rq3b.sh runs/<tag>/rq3/commands.txt
```

---

## End-to-end flow

```
Training (HPC)
  └─ hpc/jobs/rq1.sh  →  runs/{tag}/rq1/{scenario}/{condition}/seed{N}/
  └─ hpc/jobs/rq2.sh  →  runs/{tag}/rq2/{scenario}/seed{N}/
  └─ hpc/jobs/rq3b.sh →  runs/{tag}/rq3/{scenario}/{condition}/seed{N}/
        Each run dir contains:
          config.json        (includes "condition" field after fix)
          train.csv
          ckpts/ckpt_final.pt

Held-out evaluation (HPC)
  └─ sbatch hpc/jobs/eval.sh runs/{tag} 200
        Writes into each run dir:
          eval.csv, eval_summary.json
          eval_random.csv, eval_summary_random.json
        Then aggregates:
          results.csv, results_random.csv

Analysis (local or HPC)
  └─ python analysis/plot_thesis_figures.py --results results.csv
        Writes to figures/:
          rq1_effectiveness.{png,pdf}
          rq2_generalization.{png,pdf}
          rq3_ablations.{png,pdf}
```

---

## Smoke test commands

```bash
# 1. Quick training run (5 episodes, no LLM, CPU)
python -m nasim.train_llm4teach \
    --scenario tiny \
    --no-llm \
    --episodes 5 \
    --episode-length 200 \
    --save-dir runs/smoke/rq1/tiny/ppo_options/seed0/ckpts \
    --no-step-logs \
    --condition ppo_options

# 2. Evaluate the checkpoint
python -m nasim.scripts.eval_policy \
    --run-dir runs/smoke/rq1/tiny/ppo_options/seed0 \
    --episodes 20 \
    --device cpu

# 3. Verify eval_summary.json was written
cat runs/smoke/rq1/tiny/ppo_options/seed0/eval_summary.json

# 4. Aggregate
python -m nasim.scripts.build_results_table \
    --root runs/smoke \
    --output results_smoke.csv

# 5. Plot (demo mode, no real data needed)
python analysis/plot_thesis_figures.py --demo

# 6. Plot with smoke results
python analysis/plot_thesis_figures.py --results results_smoke.csv
```

Expected outcomes:
- Step 2: `eval_summary.json` present with `success_rate` ∈ [0, 1], no errors
- Step 4: `results_smoke.csv` has ≥ 1 row, `condition == "ppo_options"`
- Step 5: `figures/rq1_effectiveness.png` etc. created with synthetic data
- Step 6: same figures with real (sparse) data

---

## SLURM evaluation job

```bash
# After all training jobs are complete:
sbatch hpc/jobs/eval.sh runs/ 200
```

This job:
- Uses 4 CPUs, 8 GB RAM, 4 hours wall time
- Does **not** request a GPU (evaluation is CPU-only)
- Evaluates every run dir that contains `config.json`
- Produces `results.csv` and `results_random.csv` in the project root
