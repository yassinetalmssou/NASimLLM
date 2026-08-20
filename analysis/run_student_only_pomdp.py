"""Teacher-free deployment of the frozen POMDP students.

Same as run_student_only.py but for the partially observable checkpoints in
hpc/results/pomdp. eval_run_dir reads fully_obs=False and belief_accum=True from
each config.json, so evaluation runs under belief aggregation, matching training.
Each student is evaluated on held-out instances of the SAME scenario it trained on
(transfer across scenarios is blocked by the scenario-specific input dimension).

Layout produced:
  student_only/pomdp/deploy/<teacher>/<scenario>/seed<n>/eval_{greedy,sample}.csv
  student_only/pomdp/deploy_raw.csv       one row per (teacher, scenario, seed, mode)
  student_only/pomdp/deploy_summary.csv   mean +- 95% CI over seeds per (teacher, scenario, mode)

Usage:
  python analysis/run_student_only_pomdp.py [episodes] [scope]
    episodes: eval episodes per run (default 100)
    scope:    "rq1" = small only (all 5 teachers); "all" = + rq2 panel scenarios (default)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = r"c:/Users/yassi/Documents/VUB/Master/2MA/Thesis/NASimLLM"
sys.path.insert(0, BASE)
from nasim.scripts.eval_policy import eval_run_dir  # noqa: E402

RUNS = Path(BASE) / "hpc" / "results" / "pomdp"
OUT = Path(BASE) / "student_only" / "pomdp"
TEACHERS = ["qwen-4B", "qwen-8B", "llama-1B", "llama-3B", "llama-8B"]
PANEL = ["qwen-4B", "llama-3B", "llama-8B"]
SEEDS = [0, 1, 2, 3, 4, 5]
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SCOPE = sys.argv[2] if len(sys.argv) > 2 else "all"
EVAL_SEED = 1000

MODES = [
    (True, "greedy", "eval_summary_greedy.json", "eval_greedy.csv"),
    (False, "sample", "eval_summary_sample.json", "eval_sample.csv"),
]


def targets():
    # each student on the scenario it trained on
    tgts = [(t, "rq1", "small") for t in TEACHERS]
    if SCOPE == "all":
        for t in PANEL:
            for scen in ["tiny", "small-linear", "medium"]:
                tgts.append((t, "rq2", scen))
    return tgts


def run():
    rows = []
    for t, grp, scen in targets():
        for s in SEEDS:
            rd = RUNS / t / grp / scen / "llm_full" / f"seed{s}"
            if not (rd / "ckpts" / "ckpt_final.pt").exists():
                print(f"  skip (no ckpt): {rd}")
                continue
            dest = OUT / "deploy" / t / scen / f"seed{s}"
            for greedy, mode, sname, cname in MODES:
                summ = eval_run_dir(
                    run_dir=rd,
                    num_episodes=EPISODES,
                    eval_seed=EVAL_SEED,
                    greedy=greedy,
                    out_dir=dest,
                    summary_name=sname,
                    csv_name=cname,
                )
                rows.append({
                    "teacher": t, "scenario": scen, "seed": s, "mode": mode,
                    "success": summ["success_rate"],
                    "ci_low": summ["success_ci_low"],
                    "ci_high": summ["success_ci_high"],
                    "median_steps": summ["median_steps_to_success"],
                    "episodes": EPISODES,
                })
                print(f"  {t:9s} {scen:12s} seed{s} {mode:6s}: {summ['success_rate']*100:5.1f}%")
    return pd.DataFrame(rows)


def summarise(df):
    out = []
    for (t, scen, mode), g in df.groupby(["teacher", "scenario", "mode"]):
        v = g["success"].values
        ci = stats.sem(v) * stats.t.ppf(0.975, len(v) - 1) if len(v) > 1 else 0.0
        out.append({
            "teacher": t, "scenario": scen, "mode": mode, "n_seeds": len(v),
            "success_mean": round(float(np.mean(v)), 4),
            "success_ci": round(float(ci), 4),
        })
    return pd.DataFrame(out).sort_values(["teacher", "scenario", "mode"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== teacher-free POMDP deployment eval  (episodes={EPISODES}, scope={SCOPE}) ===")
    raw = run()
    raw.to_csv(OUT / "deploy_raw.csv", index=False)
    summ = summarise(raw)
    summ.to_csv(OUT / "deploy_summary.csv", index=False)
    print("\n=== SUMMARY (mean over seeds) ===")
    for _, r in summ.iterrows():
        print(f"  {r['teacher']:9s} {r['scenario']:12s} {r['mode']:6s} "
              f"= {r['success_mean']*100:5.1f}%  (n={int(r['n_seeds'])})")
    print("\nwrote", OUT / "deploy_summary.csv")
