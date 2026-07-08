# EXECUTION.md — running the NASimLLM experiments on VUB Hydra

Operational runbook for producing the thesis results under time pressure. Ordered so that
whatever finishes first is already a complete, defensible thesis. Do not skip the gates:
a silent failure found at hour 20 of a 24-hour job is the single most expensive thing that
can happen right now, and every gate below costs minutes while protecting hours.

This assumes the evaluation layer is already merged and the harness smoke test passes:
`nasim/scripts/eval_policy.py`, `nasim/scripts/build_results_table.py`,
`analysis/plot_thesis_figures.py`, `hpc/jobs/eval.sh`, and the RQ3 wiring resolved
(the agent's Task 0). If any of those is missing, stop and finish that first.

---

## 0. Configuration (fill these in once)

```
HPC:            vsc11800@login.hpc.vub.be
PROJECT:        $VSC_DATA/NASimLLM
VENV:           $VSC_DATA/.venvs/nasim
WEIGHTS:        $VSC_SCRATCH/llm_weights/{llama-3.2-1B-Instruct, llama-3.2-3B-Instruct,
                                          llama-8B, Qwen3-4B, qwen-8B}
RUNS ROOT:      $VSC_SCRATCH/runs   (synced back to $VSC_DATA/runs by the job scripts)

PRIMARY_TEACHER (T*):   <decide in Step 1>     # model alias, e.g. qwen-4B
PRIMARY_WEIGHTS:        $VSC_SCRATCH/llm_weights/<...>
SEEDS:                  0 1 2
EPISODES:               <from Step 1 pilot; README uses 100>
EPISODE_LENGTH:         500
```

Training-command template (matches the README; used by the generators and the pilots):

```bash
python -m nasim.train_llm4teach \
  --scenario <SCEN> --seed <S> --episodes <E> --episode-length 500 \
  --hidden-dim 128 --learning-rate 3e-4 --batch-size 64 --num-epochs 4 \
  --lambda-mode competence --lambda-start 1.0 --lambda-decay 0.97 --lambda-min 0.05 \
  --llm-mix-weight 0.5 --llama-model <WEIGHTS> --device cuda \
  --save-dir runs/<...>/ckpts --no-step-logs
# baseline: same line with --no-llm and no --llama-model
```

---

## 1. Pick the primary teacher (do this before committing GPU hours)

The whole backbone uses ONE teacher, so this choice sets the project's cost. Do not
deliberate — measure. Run two short single-seed pilots on `small`, one with llama-3.2-3B and
one with Qwen3-4B:

```bash
python -m nasim.train_llm4teach --scenario small --seed 0 --episodes 40 --episode-length 500 \
  --hidden-dim 128 --learning-rate 3e-4 --batch-size 64 --num-epochs 4 \
  --lambda-mode competence --lambda-start 1.0 --lambda-decay 0.97 --lambda-min 0.05 \
  --llm-mix-weight 0.5 --llama-model $VSC_SCRATCH/llm_weights/llama-3.2-3B-Instruct \
  --device cuda --save-dir runs/pilot_llama3b/ckpts --no-step-logs
# repeat with --llama-model .../Qwen3-4B --save-dir runs/pilot_qwen4b/ckpts
```

In each `train.csv`, compare `teacher_agree` and the rate of parse fallbacks (episodes where
the shaped reward collapses to base reward because the JSON failed). Pick the cleaner one as
**T\***. If even this feels like too much, default to **Qwen3-4B** — it is usually the better
compact-JSON model. Keep the 8B models OUT of the backbone; they are the slowest and add
nothing to the core LLM-vs-no-LLM claim.

---

## 2. The run matrix

Every cell is a training run; each seed is a separate array task. Random and bruteforce do
**not** train — they enter only at evaluation as the RQ1 bounds.

| Study | Scenarios | Conditions | Seeds | Training runs |
|---|---|---|---|---|
| **Backbone** (answers RQ1 + RQ2) | tiny, small, small-linear, medium | `llm_full`(T\*), `ppo_options` | 3 | 24 |
| **RQ3 ablation** (gated) | small only | full, no_history, no_avoidlist, verbose_prompt, llm_cached | 3 | 15 |
| **Teacher scaling** (optional) | small only | `llm_full` × {1B, 3B, 4B, 8B, qwen-8B} | 3 | 15 |

Key efficiency rule: **do not run RQ1 and RQ2 as separate studies.** RQ1's scenarios
(tiny/small/medium) are a subset of RQ2's. Train the backbone once across all four scenarios
with both conditions, then derive RQ1 as a plot-time slice — `plot_thesis_figures.py` already
filters by scenario. Teacher scaling reuses the backbone's `small` `ppo_options` run as its
shared no-LLM anchor, so all five teachers are measured against the same baseline.

> Agent note: verify that `run_rq2.py` emits **both** `llm_full` and `ppo_options` across all
> four scenarios. If it only emits the full system, extend it (or add a minimal
> `run_backbone.py`) so the backbone is exactly `{llm_full, ppo_options} × {tiny, small,
> small-linear, medium} × 3 seeds`. If time forbids editing generators, fall back to running
> `run_rq1` (tiny/small/medium) + `run_rq2` (four scenarios) as-is and accept that `llm_full`
> is trained twice on the shared scenarios — a few wasted runs, not a correctness problem.

---

## 3. Four gates (blocking — pass all before launching arrays)

**Gate 1 — harness works without the LLM.**
```bash
python -m nasim.train_llm4teach --scenario tiny --no-llm --episodes 5 \
  --episode-length 200 --save-dir runs/smoke/ckpts --no-step-logs
python -m nasim.scripts.eval_policy --run-dir runs/smoke --episodes 20 --device cpu
```
PASS = `runs/smoke/eval_summary.json` appears with a plausible `success_rate`, no errors.

**Gate 2 — teacher path works end to end.** Fastest model, smallest scenario:
```bash
python -m nasim.train_llm4teach --scenario tiny --seed 0 --episodes 10 --episode-length 300 \
  --lambda-mode competence --llm-mix-weight 0.5 \
  --llama-model $VSC_SCRATCH/llm_weights/llama-3.2-1B-Instruct \
  --device cuda --save-dir runs/gate_llm/ckpts --no-step-logs
```
PASS = Llama loads on GPU, advisor produces parsable JSON, cache fills (see `cache_hits`),
no crash mid-rollout.

**Gate 3 — RQ3 ablations actually differ.** Read the agent's Task 0 conclusion. If the
ablation flags were NOT wired through, **do not submit RQ3** until fixed. Confirm at runtime:
```bash
# two 5-episode runs that should behave differently
python -m nasim.train_llm4teach --scenario small --episodes 5 --episode-length 200 \
  --llama-model $PRIMARY_WEIGHTS --device cuda --save-dir runs/gate_full/ckpts   # full
# + the equivalent no_history command your generator emits
```
PASS = the two runs' advisor prompts / logs / `train.csv` are visibly different, not identical.

**Gate 4 — learning actually happens.** One `small llm_full` pilot with T\* (this can be the
Step 1 pilot). Eyeball `train.csv`: success rate trends up, `lambda_kl` anneals down,
`teacher_agree` is non-trivial. PASS = an upward learning curve, not flat noise.

---

## 4. Estimate cost once, then commit

From the `small llm_full` pilot, read `t_episode_s` in `train.csv`:

```
per_run_wallclock ≈ mean(t_episode_s) × EPISODES
backbone_wallclock ≈ per_run_wallclock × 24 / (concurrent GPU jobs allowed)
```

If the backbone doesn't fit the deadline, cut EPISODES before cutting seeds — three seeds on
a shorter run beats one seed on a long run for the variance story.

---

## 5. Submit in priority tiers (essential first)

Generate command files, then submit selectively so the important jobs queue earliest:

```bash
cd $VSC_DATA/NASimLLM
bash hpc/launch.sh $PRIMARY_TEACHER      # writes commands.txt per RQ/scenario, prints sbatch lines
```

**Tier 1 — Backbone (must-have; this alone is the thesis).**
Submit the four-scenario × two-condition × 3-seed set for T\*. Use the `sbatch --array=...`
lines `launch.sh` printed for the backbone (RQ2 set + the tiny/small/medium conditions).
Then evaluate as each finishes (Section 6).

**Tier 2 — RQ3 ablations on `small` only (only if Gate 3 passed).**
Submit *just* the `small` rq3b array — not tiny, not medium:
```bash
sbatch --array=1-<N> hpc/jobs/rq3b.sh $VSC_SCRATCH/runs/$PRIMARY_TEACHER/rq3/commands_small.txt
```

**Tier 3 — Teacher scaling on `small` only (optional; first to drop).**
For each of the other four models, generate and submit only the `small llm_full` task:
```bash
for M in llama-1B llama-3B llama-8B qwen-8B; do
  bash hpc/launch.sh $M
  # submit only the small/llm_full line from the printed sbatch commands for $M
done
```
The 8B models run ONLY here.

> `bash hpc/submit_all.sh $PRIMARY_TEACHER` submits every RQ × every scenario for a model in
> one go. Convenient, but it also fires RQ3 on all three scenarios and does not respect the
> tiering — prefer the selective `sbatch` above when time is the constraint.

---

## 6. Evaluate as you go (don't wait for everything)

Evaluation is CPU-only and cheap. The moment a training array for a model finishes:
```bash
sbatch hpc/jobs/eval.sh $VSC_DATA/runs/$PRIMARY_TEACHER 200
```
This scores every finished checkpoint (held-out, native success rate), runs the random
baseline, and writes `results.csv` + `results_random.csv` into that tree. By the time the last
GPU job clears, your results are essentially ready.

---

## 7. Render the figures

Pull `results.csv` locally and render:
```bash
python analysis/plot_thesis_figures.py --results results.csv
# preview styling with no data:  python analysis/plot_thesis_figures.py --demo --outdir preview
```
Outputs `figures/rq1_effectiveness`, `rq2_generalization`, `rq3_ablations` as PNG + PDF.

---

## 8. Deadline triage

- **The backbone with 3 seeds is the whole thesis.** RQ1 and RQ2 both come from it. If you
  have only this, you can answer "does the LLM help" and "does the advantage hold as networks
  scale and topology changes" with defensible intervals.
- **RQ3 and teacher scaling are enhancements**, not requirements. Dropping them is an easy
  scope justification. Teacher scaling goes first, then RQ3.
- **If you finish the backbone with margin left**, spend it lifting `small` — your most-cited
  scenario — from 3 seeds to 5. That tightens the one interval reviewers scrutinise hardest.
- **Never trade seeds for scenarios.** A single-seed number cannot carry a variance claim.

---

## 9. Metrics recap (what each figure reports)

- **Success rate** is primary everywhere (native `env.goal_reached()`, held-out episodes).
- **Steps-to-success** and **native return** are the efficiency diagnostics — on `tiny`,
  where everyone succeeds, the LLM's win shows up as fewer steps, not a higher ceiling.
- **Sample efficiency** (episodes to first success, from `train.csv`) is often the clearest
  LLM advantage — include it in the RQ1 discussion.
- **For RQ3**, pair success rate with `cache_hits` and `t_llm_s`: `llm_cached` and
  `verbose_prompt` are cost-vs-performance trade-offs, so report what was paid for what.
- Report mean ± 95% interval (Wilson for the success proportion). With 3 seeds, lean on effect
  sizes, not p-values, and say so.

---

## Optional: gate driver

A `hpc/run_gates.sh` that runs Gates 1–4 in order and exits non-zero on the first failure
would enforce "no failed submissions" mechanically. Logic: run each gate's command, check its
expected artifact/column, print PASS/FAIL, and `exit 1` on failure so a submit step chained
after it never fires on a broken pipeline. Generate it from the gate definitions in Section 3.
