"""Regenerate the figures used in the thesis from the merged runs/ tree.

Clean titles (detail lives in the LaTeX captions), consistent Okabe-Ito colours
per teacher, and one-view readability. Writes each PNG to both the working
figures/full_obs/ directory and the thesis figures directory.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import sys
BASE = r"c:/Users/yassi/Documents/VUB/Master/2MA/Thesis/NASimLLM"
# argv[1] = data root (relative to BASE), argv[2] = filename suffix for the output.
ROOT = os.path.join(BASE, sys.argv[1]) if len(sys.argv) > 1 else os.path.join(BASE, "runs")
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
_base = os.path.basename(ROOT.rstrip("/\\"))
_work = "full_obs" if _base == "runs" else _base  # runs->full_obs, hpc/pomdp_2->pomdp_2
OUTDIRS = [os.path.join(BASE, "figures", _work),
           os.path.join(BASE, "Thesis_Draft", "thesis", "figures", _work)]

SEEDS = [0, 1, 2, 3, 4, 5]  # covers full_obs (3 seeds) and POMDP (6 seeds); missing seeds are skipped
W = 100
TEACHERS = ["qwen-4B", "qwen-8B", "llama-1B", "llama-3B", "llama-8B"]
PANEL = ["qwen-4B", "llama-8B", "llama-3B"]

TCOL = {"qwen-4B": "#0072B2", "qwen-8B": "#56B4E9", "llama-1B": "#009E73",
        "llama-3B": "#D55E00", "llama-8B": "#CC79A7"}
TLAB = {"qwen-4B": "Qwen-4B", "qwen-8B": "Qwen-8B", "llama-1B": "Llama-1B",
        "llama-3B": "Llama-3B", "llama-8B": "Llama-8B"}
PPO_COL = "#4D4D4D"
SCN = ["tiny", "small", "small-linear", "medium"]
SCN_LAB = ["Tiny\n(3 hosts)", "Small\n(8 hosts)", "Small-linear\n(8, chain)", "Medium\n(16 hosts)"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def seed_csvs(t, grp, scen, cond):
    out = []
    for s in SEEDS:
        p = f"{ROOT}/{t}/{grp}/{scen}/{cond}/seed{s}/train.csv"
        if os.path.exists(p):
            try:
                d = pd.read_csv(p)
                if len(d):
                    out.append(d)
            except Exception:
                pass
    return out


def final(t, grp, scen, cond, metric="success"):
    v = [d[metric].tail(W).mean() for d in seed_csvs(t, grp, scen, cond) if metric in d]
    if not v:
        return None
    ci = stats.sem(v) * stats.t.ppf(0.975, len(v) - 1) if len(v) > 1 else 0.0
    return float(np.mean(v)), float(ci), len(v)


def rolled(dfs, metric="success", win=75):
    L = min(len(d) for d in dfs)
    rows = [pd.Series(d[metric].values[:L]).rolling(win, min_periods=10).mean().values
            for d in dfs if metric in d]
    R = np.vstack(rows)
    mean = np.nanmean(R, axis=0)
    if R.shape[0] > 1:
        ci = stats.sem(R, axis=0, nan_policy="omit") * stats.t.ppf(0.975, R.shape[0] - 1)
    else:
        ci = np.zeros_like(mean)
    return np.arange(L), mean, np.asarray(ci)


def save(fig, name):
    stem, ext = os.path.splitext(name)
    fname = f"{stem}{SUFFIX}{ext}"
    for od in OUTDIRS:
        os.makedirs(od, exist_ok=True)
        fig.savefig(os.path.join(od, fname))
    plt.close(fig)
    print("  wrote", fname)


def fig_learning():
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in TEACHERS:
        dfs = seed_csvs(t, "rq1", "small", "llm_full")
        if not dfs:
            continue
        x, m, _ = rolled(dfs)
        ax.plot(x, m * 100, color=TCOL[t], lw=1.8, label=TLAB[t])
    dfs = seed_csvs("qwen-4B", "rq1", "small", "ppo_options")
    if dfs:
        x, m, _ = rolled(dfs)
        ax.plot(x, m * 100, color=PPO_COL, lw=2.6, label="PPO (no LLM)")
    for cond, txt, fb in [("bruteforce", "bruteforce", 100), ("random", "random", 56)]:
        r = final("qwen-4B", "rq1", "small", cond)
        y = r[0] * 100 if r else fb  # baseline computed from data (regime-specific); fallback if absent
        ax.axhline(y, ls=":", lw=1.1, color="#9a9a9a")
        ax.text(2010, y, txt, va="center", ha="left", fontsize=9, color="#6a6a6a")
    ax.set_ylim(-3, 106)
    ax.set_xlim(0, 2000)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Task success rate")
    ax.set_title("Training success rate")
    ax.legend(ncol=3, loc="lower right", fontsize=9.5)
    save(fig, "rq1_learning_curves.png")


def eps80(t, thr=0.8, win=50):
    es = []
    for d in seed_csvs(t, "rq1", "small", "llm_full"):
        roll = d["success"].rolling(win, min_periods=win).mean()
        hit = roll[roll >= thr]
        if len(hit):
            es.append(int(hit.index[0]) + 1)
    return float(np.mean(es)) if es else None


def fig_sample():
    vals = [(t, eps80(t)) for t in TEACHERS]
    vals = [(t, e) for t, e in vals if e]
    vals.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = list(range(len(vals)))
    ax.barh(y, [e for _, e in vals], color=[TCOL[t] for t, _ in vals], height=0.62, zorder=3)
    for yi, (t, e) in zip(y, vals):
        ax.text(e + 7, yi, f"{int(round(e))} ep", va="center", fontweight="bold", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels([TLAB[t] for t, _ in vals])
    ax.invert_yaxis()
    ax.set_xlim(0, max(e for _, e in vals) * 1.2)
    ax.set_xlabel("Episodes to reach 80% success")
    ax.set_title("Sample efficiency")
    ax.grid(axis="y", visible=False)
    save(fig, "rq1_sample_efficiency.png")


def fig_rq2():
    defs = [("qwen-4B", "llm_full", TCOL["qwen-4B"], "Qwen-4B"),
            ("llama-8B", "llm_full", TCOL["llama-8B"], "Llama-8B"),
            ("llama-3B", "llm_full", TCOL["llama-3B"], "Llama-3B"),
            ("qwen-4B", "ppo_options", PPO_COL, "PPO (no LLM)")]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = np.arange(len(SCN))
    w = 0.2
    for j, (t, cond, col, lab) in enumerate(defs):
        ys, lo, hi = [], [], []
        for s in SCN:
            r = final(t, "rq2", s, cond)
            y = r[0] * 100 if r else np.nan
            e = r[1] * 100 if r else 0.0
            ys.append(y)
            lo.append(min(e, y) if not np.isnan(y) else 0.0)
            hi.append(min(e, 100 - y) if not np.isnan(y) else 0.0)
        ax.bar(x + (j - 1.5) * w, ys, w, yerr=[lo, hi], capsize=2.5, color=col, label=lab,
               error_kw=dict(lw=1.0, alpha=0.55), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(SCN_LAB)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, 105)
    ax.set_ylabel("Final success rate")
    ax.set_title("Generalisation across scenarios")
    ax.legend(ncol=4, loc="upper right", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    save(fig, "rq2_generalization.png")


def fig_partial(t):
    metrics = [("success", "Mission success", TCOL["qwen-4B"], 100),
               ("compromise_rate", "Hosts compromised (all)", "#E69F00", 100),
               ("discovery_pct", "Hosts discovered (all)", TCOL["llama-1B"], 1)]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x = np.arange(len(SCN))
    w = 0.26
    for j, (m, lab, col, scale) in enumerate(metrics):
        ys = []
        for s in SCN:
            r = final(t, "rq2", s, "llm_full", metric=m)
            ys.append(r[0] * scale if r else np.nan)
        pos = x + (j - 1) * w
        ax.bar(pos, ys, w, color=col, label=lab, zorder=3)
        for xi, v in zip(pos, ys):
            if not np.isnan(v):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SCN_LAB)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, 108)
    ax.set_ylabel("Rate")
    ax.set_title("Progress breakdown by scenario")
    ax.legend(ncol=3, loc="upper center", fontsize=9.5, bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    save(fig, f"rq2_partial_progress_{t}.png")


def fig_rq3_ablations():
    conds = ["no_history", "no_avoidlist", "verbose_prompt"]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x = np.arange(len(conds))
    n = len(TEACHERS)
    w = 0.15
    for j, t in enumerate(TEACHERS):
        f = final(t, "rq3", "small", "full")
        if not f or f[0] < 0.30:  # skip collapsed baselines: deltas vs ~0 are meaningless
            continue
        d = []
        for c in conds:
            r = final(t, "rq3", "small", c)
            d.append((r[0] - f[0]) * 100 if r else np.nan)
        pos = x + (j - (n - 1) / 2) * w
        ax.bar(pos, d, w, color=TCOL[t], label=TLAB[t], zorder=3)
        for xi, v in zip(pos, d):
            if not np.isnan(v):
                ax.text(xi, v + (0.7 if v >= 0 else -0.7), f"{v:+.0f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel("Change in success vs full (pp)")
    ax.set_title("Component ablations by teacher")
    ax.legend(ncol=2, loc="lower right", fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "rq3_ablations_by_teacher.png")


def fig_schedules():
    scheds = [("competence", "full", "rq3"), ("fixed", "lam_fixed", "sens"),
              ("target-kl", "sched_targetkl", "sens"), ("adaptive", "sched_adaptive", "sens")]
    teachers = [t for t in ["qwen-4B", "qwen-8B", "llama-1B"]
                if final(t, "rq3", "small", "full")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(scheds))
    n = len(teachers)
    w = 0.22
    for j, t in enumerate(teachers):
        ys = []
        for _, cond, grp in scheds:
            r = final(t, grp, "small", cond)
            ys.append(r[0] * 100 if r else 0.0)
        pos = x + (j - (n - 1) / 2) * w
        ax.bar(pos, ys, w, color=TCOL[t], label=TLAB[t], zorder=3)
        for xi, v in zip(pos, ys):
            ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=8.5,
                    color="#B00000" if v < 1 else "black",
                    fontweight="bold" if v < 1 else "normal")
    fi = [i for i, s in enumerate(scheds) if s[0] == "fixed"][0]
    ax.annotate("collapse to 0%", xy=(fi, 1.5), xytext=(fi, 30), ha="center",
                fontsize=10.5, color="#B00000",
                arrowprops=dict(arrowstyle="->", color="#B00000", lw=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in scheds])
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, 100)
    ax.set_ylabel("Final success rate")
    ax.set_title("Lambda-annealing schedules")
    ax.legend(loc="center", bbox_to_anchor=(0.375, 0.60), fontsize=9.5)
    ax.grid(axis="x", visible=False)
    save(fig, "rq3_schedules.png")


def lam_curve(t, grp, cond, win=41):
    dfs = seed_csvs(t, grp, "small", cond)
    if not dfs or "lambda_kl" not in dfs[0]:
        return None
    return rolled(dfs, metric="lambda_kl", win=win)


def fig_lambda():
    fig, ax = plt.subplots(figsize=(9, 5))
    series = [("llm_full", "rq1", TCOL["qwen-4B"], "Competence (default)", "-"),
              ("lam_fixed", "sens", "#D55E00", "Fixed (time-based)", "--")]
    for cond, grp, col, lab, ls in series:
        c = lam_curve("qwen-4B", grp, cond)
        if not c:
            continue
        x, m, ci = c
        ax.plot(x, m, color=col, lw=2.2, ls=ls, label=lab)
        ax.fill_between(x, m - ci, m + ci, color=col, alpha=0.14, lw=0)
    ax.axhline(0, color="#9a9a9a", ls=":", lw=1.1)
    ax.set_ylim(-0.03, 1.02)
    ax.set_xlim(0, 2000)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Teacher weight (lambda)")
    ax.set_title("Teacher influence over training")
    ax.text(1960, 0.43, "floor ~0.40", ha="right", va="bottom", fontsize=10, color=TCOL["qwen-4B"])
    ax.legend(loc="upper right", fontsize=10)
    save(fig, "rq1_lambda_annealing.png")


def fig_history():
    hs = [4, 8, 16, 32, 64]
    ys, lo, hi = [], [], []
    for h in hs:
        r = final("qwen-4B", "sens", "small", f"hist_{h}")
        y = r[0] * 100 if r else np.nan
        e = r[1] * 100 if r else 0.0
        ys.append(y)
        lo.append(min(e, y) if not np.isnan(y) else 0.0)
        hi.append(min(e, 100 - y) if not np.isnan(y) else 0.0)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xx = np.arange(len(hs))
    ax.errorbar(xx, ys, yerr=[lo, hi], fmt="o", ms=10, color=TCOL["qwen-4B"],
                capsize=4, lw=0, elinewidth=1.7, ecolor=TCOL["qwen-4B"], zorder=3)
    m = np.nanmean(ys)
    ax.axhline(m, color="#9a9a9a", ls="--", lw=1.2)
    ax.text(len(hs) - 1, m + 1.5, f"mean {m:.0f}%", ha="right", va="bottom", fontsize=9.5, color="#6a6a6a")
    ax.set_xticks(xx)
    ax.set_xticklabels(hs)
    ax.set_ylim(40, 102)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Recent-action log length")
    ax.set_ylabel("Final success rate")
    ax.set_title("Action-history length")
    save(fig, "rq3_history_length.png")


def fig_cost():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = np.arange(len(SCN))
    n = len(PANEL)
    w = 0.24
    top = 0
    for j, t in enumerate(PANEL):
        ys = []
        for s in SCN:
            calls = [d["cache_misses"].sum() for d in seed_csvs(t, "rq2", s, "llm_full") if "cache_misses" in d]
            ys.append(float(np.mean(calls)) if calls else np.nan)
        top = max(top, np.nanmax(ys))
        pos = x + (j - (n - 1) / 2) * w
        ax.bar(pos, ys, w, color=TCOL[t], label=TLAB[t], zorder=3)
        for xi, v in zip(pos, ys):
            if not np.isnan(v):
                ax.text(xi, v + top * 0.012, f"{int(round(v))}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SCN_LAB)
    ax.set_ylim(0, top * 1.12)
    ax.set_ylabel("Queries per run (cache misses)")
    ax.set_title("Real LLM queries per run")
    ax.legend(ncol=3, loc="upper left", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    save(fig, "rq_llm_call_cost.png")


if __name__ == "__main__":
    fig_learning()
    fig_sample()
    fig_rq2()
    for t in PANEL:
        fig_partial(t)
    fig_rq3_ablations()
    fig_schedules()
    fig_lambda()
    if any(final("qwen-4B", "sens", "small", f"hist_{h}") for h in [4, 8, 16, 32, 64]):
        fig_history()
    else:
        print("  (skip history: no data in this root)")
    fig_cost()
    print("done")
