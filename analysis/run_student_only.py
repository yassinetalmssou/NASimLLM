"""Teacher-free deployment of the frozen students (promotor feedback p23/p47).

Runs each final rq1 llm_full checkpoint with the LLM removed, under both greedy
and stochastic option selection, on held-out instances of the SAME scenario the
student trained on. All outputs go under student_only/ so they stay separate from
the training data in runs/.

Layout produced:
  student_only/deploy/<teacher>/seed<n>/eval_{greedy,sample}.csv
  student_only/deploy/<teacher>/seed<n>/eval_summary_{greedy,sample}.json
  student_only/deploy_raw.csv       one row per (teacher, seed, mode)
  student_only/deploy_summary.csv   mean +- 95% CI over seeds per (teacher, mode)
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

RUNS = Path(BASE) / "runs"
OUT = Path(BASE) / "student_only"
TEACHERS = ["qwen-4B", "qwen-8B", "llama-1B", "llama-3B", "llama-8B"]
SEEDS = [0, 1, 2]
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
EVAL_SEED = 1000

MODES = [
    (True, "greedy", "eval_summary_greedy.json", "eval_greedy.csv"),
    (False, "sample", "eval_summary_sample.json", "eval_sample.csv"),
]


def run():
    rows = []
    for t in TEACHERS:
        for s in SEEDS:
            rd = RUNS / t / "rq1" / "small" / "llm_full" / f"seed{s}"
            if not (rd / "ckpts" / "ckpt_final.pt").exists():
                print(f"  skip (no ckpt): {rd}")
                continue
            dest = OUT / "deploy" / t / f"seed{s}"
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
                    "teacher": t, "seed": s, "mode": mode,
                    "success": summ["success_rate"],
                    "ci_low": summ["success_ci_low"],
                    "ci_high": summ["success_ci_high"],
                    "median_steps": summ["median_steps_to_success"],
                    "episodes": EPISODES,
                })
                print(f"  {t:9s} seed{s} {mode:6s}: {summ['success_rate']*100:5.1f}%")
    return pd.DataFrame(rows)


def summarise(df):
    out = []
    for (t, mode), g in df.groupby(["teacher", "mode"]):
        v = g["success"].values
        ci = stats.sem(v) * stats.t.ppf(0.975, len(v) - 1) if len(v) > 1 else 0.0
        out.append({
            "teacher": t, "mode": mode, "n_seeds": len(v),
            "success_mean": round(float(np.mean(v)), 4),
            "success_ci": round(float(ci), 4),
        })
    return pd.DataFrame(out).sort_values(["teacher", "mode"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== teacher-free deployment eval  (episodes={EPISODES}) ===")
    raw = run()
    raw.to_csv(OUT / "deploy_raw.csv", index=False)
    summ = summarise(raw)
    summ.to_csv(OUT / "deploy_summary.csv", index=False)
    print("\n=== SUMMARY (mean over seeds) ===")
    for t in TEACHERS:
        g = summ[summ.teacher == t]
        if g.empty:
            continue
        gr = g[g["mode"] == "greedy"]["success_mean"]
        sa = g[g["mode"] == "sample"]["success_mean"]
        gr = f"{gr.iloc[0]*100:.0f}%" if len(gr) else "-"
        sa = f"{sa.iloc[0]*100:.0f}%" if len(sa) else "-"
        print(f"  {t:9s}  greedy={gr:5s}  stochastic={sa}")
    print("\nwrote", OUT / "deploy_summary.csv")
