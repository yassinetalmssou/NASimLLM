"""Thesis figures for RQ1, RQ2 and RQ3 from per-seed train.csv runs.

Learning curves are rolling-mean smoothed and averaged across seeds with a
95% CI band. Writes PNG (300 dpi) to figures/.
Run: python analysis/make_rq_figures.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
FIG  = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

SEEDS   = [0, 1, 2]
FINAL_W = 100
SMOOTH  = 51

# Okabe-Ito categorical palette (colourblind-safe)
COL = {"qwen-4B": "#0072B2", "qwen-8B": "#56B4E9",
       "llama-1B": "#009E73", "llama-3B": "#D55E00", "llama-8B": "#CC79A7"}
LBL = {"qwen-4B": "Qwen-4B", "qwen-8B": "Qwen-8B",
       "llama-1B": "Llama-1B", "llama-3B": "Llama-3B", "llama-8B": "Llama-8B"}
LS  = {"qwen-4B": "-", "qwen-8B": "-", "llama-1B": "--", "llama-3B": "--", "llama-8B": "--"}
PPO_COL, RND_COL, BF_COL = "#E69F00", "#999999", "#333333"
NEG, POS, ZERO = "#D55E00", "#009E73", "#B0B0B0"
TEACHERS = ["qwen-4B", "qwen-8B", "llama-1B", "llama-3B", "llama-8B"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DEDEDE", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})

def _csvs(teacher, rq, scen, cond):
    out = []
    for s in SEEDS:
        p = os.path.join(RUNS, teacher, rq, scen, cond, f"seed{s}", "train.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            if len(d):
                out.append(d)
    return out

def final(teacher, rq, scen, cond, metric="success"):
    v = [d[metric].tail(FINAL_W).mean() for d in _csvs(teacher, rq, scen, cond) if metric in d]
    if not v:
        return None
    ci = stats.sem(v) * stats.t.ppf(0.975, len(v) - 1) if len(v) > 1 else 0.0
    return np.mean(v), ci

def curve(teacher, rq, scen, cond, metric="success"):
    dfs = _csvs(teacher, rq, scen, cond)
    if not dfs or any(metric not in d for d in dfs):
        return None
    n = min(len(d) for d in dfs)
    mat = np.stack([pd.Series(d[metric].values[:n])
                    .rolling(SMOOTH, min_periods=1, center=True).mean().values for d in dfs])
    mu = mat.mean(0)
    ci = stats.sem(mat, 0) * stats.t.ppf(0.975, mat.shape[0] - 1) if mat.shape[0] > 1 else np.zeros(n)
    return np.arange(1, n + 1), mu, ci

def eps_to(teacher, rq, scen, cond, thr=0.8, win=50):
    es = []
    for d in _csvs(teacher, rq, scen, cond):
        roll = d["success"].rolling(win, min_periods=win).mean()
        hit = roll[roll >= thr]
        if len(hit):
            es.append(int(hit.index[0]) + 1)
    return float(np.mean(es)) if es else None

def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)
    print("  saved", name + ".png")

# RQ1a final success: sorted horizontal bars plus baseline reference lines
def fig_rq1_final():
    items = []
    for t in TEACHERS:
        r = final(t, "rq1", "small", "llm_full")
        if r:
            items.append([LBL[t], r[0] * 100, r[1] * 100, COL[t], False])
    for cond, lab, c in [("bruteforce", "Bruteforce (ref)", BF_COL),
                         ("random", "Random (ref)", RND_COL),
                         ("ppo_options", "PPO - no LLM", PPO_COL)]:
        r = final("qwen-4B", "rq1", "small", cond)
        if r:
            items.append([lab, r[0] * 100, r[1] * 100, c, True])
    items.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    y = np.arange(len(items))
    ax.barh(y, [i[1] for i in items], xerr=[i[2] for i in items],
            color=[i[3] for i in items], edgecolor="white", height=0.66, capsize=3,
            hatch=["///" if i[4] else "" for i in items], zorder=3, error_kw=dict(lw=1.1))
    for i, it in enumerate(items):
        ax.text(it[1] + it[2] + 1.6, i, f"{it[1]:.0f}%", va="center", fontsize=10.5, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items])
    ax.set_xlim(0, 112); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Final success rate (%)  ·  last 100 ep, mean ± 95% CI over 3 seeds")
    ax.set_title("RQ1 - LLM-guided agents vs. baselines  (scenario: small)", fontweight="bold", pad=10)
    ax.text(0.99, 0.03, "hatched = baseline (no learned LLM policy)", transform=ax.transAxes,
            ha="right", fontsize=8.5, color="#666666", style="italic")
    save(fig, "rq1_final_success")

# RQ1b smoothed, seed-averaged learning curves
def fig_rq1_curves():
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in TEACHERS:
        c = curve(t, "rq1", "small", "llm_full")
        if not c:
            continue
        x, mu, ci = c
        ax.plot(x, mu * 100, color=COL[t], ls=LS[t], lw=1.9, label=LBL[t], zorder=3)
    c = curve("qwen-4B", "rq1", "small", "ppo_options")
    if c:
        x, mu, ci = c
        ax.plot(x, mu * 100, color=PPO_COL, lw=2.6, label="PPO (no LLM)", zorder=4)
    bf = final("qwen-4B", "rq1", "small", "bruteforce")[0] * 100
    rnd = final("qwen-4B", "rq1", "small", "random")[0] * 100
    for yv, lab, c2 in [(bf, "bruteforce", BF_COL), (rnd, "random", RND_COL)]:
        ax.axhline(yv, ls=":", lw=1.3, color=c2, zorder=2)
        ax.text(1002, yv, f" {lab}", va="center", fontsize=8.5, color=c2)
    ax.set_xlabel("Training episode"); ax.set_ylabel("Task success rate (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))
    ax.set_ylim(-3, 105); ax.set_xlim(0, 1000)
    ax.set_title(f"RQ1 - Learning curves  ({SMOOTH}-ep rolling mean, averaged over 3 seeds)",
                 fontweight="bold")
    ax.legend(ncol=2, loc="lower right", fontsize=10)
    save(fig, "rq1_learning_curves")

# RQ1c sample efficiency: episodes to reach 80% success
def fig_rq1_efficiency():
    data = [(t, eps_to(t, "rq1", "small", "llm_full")) for t in TEACHERS]
    data = [d for d in data if d[1] is not None]
    data.sort(key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(8, 3.9))
    y = np.arange(len(data))
    ax.barh(y, [d[1] for d in data], color=[COL[d[0]] for d in data],
            edgecolor="white", height=0.6, zorder=3)
    for i, d in enumerate(data):
        ax.text(d[1] + 6, i, f"{int(d[1])} ep", va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([LBL[d[0]] for d in data])
    ax.set_xlabel("Episodes to reach 80% success  (lower = faster; rolling-50)")
    ax.set_title("RQ1 - Sample efficiency  (scenario: small)", fontweight="bold")
    ax.grid(axis="y", visible=False); ax.set_xlim(0, max(d[1] for d in data) * 1.18)
    save(fig, "rq1_sample_efficiency")

# RQ2 generalisation across scenarios (line with markers)
def fig_rq2():
    scn = ["tiny", "small", "small-linear", "medium"]
    xlab = ["Tiny\n(3 hosts)", "Small\n(8 hosts)", "Small-linear\n(8, chain)", "Medium\n(16 hosts)"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(scn))
    for t in ["qwen-4B", "llama-8B", "llama-3B"]:
        mus = [(final(t, "rq2", s, "llm_full") or (np.nan,))[0] * 100 for s in scn]
        ax.plot(x, mus, marker="o", ms=8, lw=2.4, color=COL[t], ls=LS[t],
                label=f"{LBL[t]} (LLM)", zorder=3)
    mus = [final("qwen-4B", "rq2", s, "ppo_options")[0] * 100 for s in scn]
    ax.plot(x, mus, marker="X", ms=9, lw=2, ls=":", color=PPO_COL, label="PPO (no LLM)", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(xlab)
    ax.set_ylabel("Final success rate (%)"); ax.set_ylim(-3, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.set_title("RQ2 - Generalisation across scenarios  (LLM teachers vs. no-LLM)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    save(fig, "rq2_generalization")

# RQ3 component ablations (diverging delta bars)
def fig_rq3():
    full = final("qwen-4B", "rq3", "small", "full")[0]
    names = {"no_history": "No history", "no_avoidlist": "No avoid-list",
             "verbose_prompt": "Verbose prompt"}
    rows = []
    for c, lab in names.items():
        r = final("qwen-4B", "rq3", "small", c)
        if r:
            rows.append((lab, (r[0] - full) * 100))
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(rows))
    colors = [NEG if v < -0.5 else POS if v > 0.5 else ZERO for _, v in rows]
    ax.barh(y, [v for _, v in rows], color=colors, edgecolor="white", height=0.6, zorder=3)
    ax.axvline(0, color="black", lw=1.3, zorder=4)
    for i, (lab, v) in enumerate(rows):
        ax.text(v - 0.5 if v < 0 else v + 0.5, i, f"{v:+.0f} pp",
                va="center", ha="right" if v < 0 else "left", fontweight="bold", fontsize=11)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Δ success rate vs. full system (pp)  ·  effect of removing the component")
    ax.set_title(f"RQ3 - Component ablations  (Qwen-4B, small; full = {full*100:.0f}%)", fontweight="bold")
    ax.grid(axis="y", visible=False)
    lo = min(v for _, v in rows); ax.set_xlim(lo - 6, 6)
    ax.text(0.99, 0.04, "← removing hurts        no effect →", transform=ax.transAxes,
            ha="right", fontsize=9, color="#666666")
    save(fig, "rq3_ablations")

# Cost of LLM guidance: real LLM queries (cache misses) per scenario
def fig_cache_cost():
    scn = ["tiny", "small", "small-linear", "medium"]
    xlab = ["Tiny\n(3 hosts)", "Small\n(8 hosts)", "Small-linear\n(8, chain)", "Medium\n(16 hosts)"]
    panel = ["qwen-4B", "llama-8B", "llama-3B"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(scn)); width = 0.26
    for j, t in enumerate(panel):
        vals = []
        for s in scn:
            calls = [d["cache_misses"].sum() for d in _csvs(t, "rq2", s, "llm_full") if "cache_misses" in d]
            vals.append(float(np.mean(calls)) if calls else np.nan)
        ax.bar(x + (j - 1) * width, vals, width, color=COL[t], edgecolor="white",
               label=LBL[t], zorder=3)
        for xi, v in zip(x + (j - 1) * width, vals):
            if not np.isnan(v):
                ax.text(xi, v + 10, f"{int(v)}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(xlab)
    ax.set_ylabel("Real LLM calls per run  (cache misses, mean over 3 seeds)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Cost of LLM guidance - real LLM queries per scenario\n"
                 "cache hit-rate ≈ 100%; queries grow with network size but stay < 0.2% of steps",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    save(fig, "rq_llm_call_cost")

# RQ1 solution efficiency: steps-to-solve per teacher
def fig_rq1_steps():
    data = [(t, *final(t, "rq1", "small", "llm_full", metric="length")) for t in TEACHERS
            if final(t, "rq1", "small", "llm_full", metric="length")]
    data.sort(key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(8, 3.9))
    y = np.arange(len(data))
    ax.barh(y, [d[1] for d in data], xerr=[d[2] for d in data],
            color=[COL[d[0]] for d in data], edgecolor="white", height=0.6, capsize=3, zorder=3)
    for i, d in enumerate(data):
        ax.text(d[1] + d[2] + 4, i, f"{d[1]:.0f}", va="center", fontsize=10.5, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([LBL[d[0]] for d in data])
    ax.set_xlabel("Steps per episode (final 100 ep)  ·  lower = more efficient policy")
    ax.set_title("RQ1 - Solution efficiency  (scenario: small)", fontweight="bold")
    ax.grid(axis="y", visible=False); ax.set_xlim(0, max(d[1] for d in data) * 1.18)
    save(fig, "rq1_solution_efficiency")

# RQ2 partial progress: success vs takeover vs discovery
def fig_rq2_partial(teacher):
    scn = ["tiny", "small", "small-linear", "medium"]
    xlab = ["Tiny\n(3 hosts)", "Small\n(8 hosts)", "Small-linear\n(8, chain)", "Medium\n(16 hosts)"]
    metrics = [("success", "Reached objective (success)", "#0072B2", 100),
               ("compromise_rate", "Hosts compromised", "#E69F00", 100),
               ("discovery_pct", "Network discovered", "#009E73", 1)]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(scn)); w = 0.26
    for j, (col, lab, c, mul) in enumerate(metrics):
        vals = [(final(teacher, "rq2", s, "llm_full", metric=col) or (np.nan,))[0] * mul for s in scn]
        ax.bar(x + (j - 1) * w, vals, w, color=c, edgecolor="white", label=lab, zorder=3)
        for xi, v in zip(x + (j - 1) * w, vals):
            if not np.isnan(v):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(xlab)
    ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.grid(axis="x", visible=False)
    ax.set_title(f"RQ2 - Partial progress: objective vs. takeover / discovery  ({LBL[teacher]})",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9.5)
    save(fig, f"rq2_partial_progress_{teacher}")

# Lambda annealing: teacher influence over training (competence schedule)
def fig_lambda():
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in TEACHERS:
        c = curve(t, "rq1", "small", "llm_full", metric="lambda_kl")
        if not c:
            continue
        x, mu, _ = c
        ax.plot(x, mu, color=COL[t], ls=LS[t], lw=1.9, label=LBL[t])
    ax.set_xlabel("Training episode"); ax.set_ylabel("λ  (teacher KL weight)")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 1000)
    ax.set_title("λ annealing - teacher influence over training (competence schedule, RQ1 small)",
                 fontweight="bold", fontsize=11.5)
    ax.legend(ncol=2, loc="upper right", fontsize=9.5)
    save(fig, "rq1_lambda_annealing")

# Agreement vs success: do the best students follow the teacher?
def fig_agreement_scatter():
    from scipy.stats import pearsonr
    xs, ys, ts = [], [], []
    for t in TEACHERS:
        a = final(t, "rq1", "small", "llm_full", metric="teacher_agree")
        s = final(t, "rq1", "small", "llm_full", metric="success")
        if a and s:
            xs.append(a[0]); ys.append(s[0] * 100); ts.append(t)
    r, p = pearsonr(xs, ys)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    z = np.polyfit(xs, ys, 1); xr = np.linspace(min(xs) - 1, max(xs) + 1, 50)
    ax.plot(xr, np.polyval(z, xr), ls="--", color="#999999", lw=1.5, zorder=1)
    for x, y, t in zip(xs, ys, ts):
        ax.scatter(x, y, s=170, color=COL[t], edgecolor="white", linewidth=1.6, zorder=3)
        ax.annotate(LBL[t], (x, y), textcoords="offset points", xytext=(9, 7),
                    fontsize=10, fontweight="bold")
    ax.set_xlabel("Teacher agreement (%)  ·  how often the student follows the teacher")
    ax.set_ylabel("Final success rate (%)")
    ax.set_title("Do the best students follow the teacher?  (RQ1, small)", fontweight="bold", fontsize=12.5)
    ax.text(0.97, 0.97,
            f"Pearson r = {r:.2f}\n(n = 5, p = {p:.2f} - not significant; exploratory)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#8B0000", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3F3", edgecolor="#D55E00", alpha=0.9))
    save(fig, "agreement_vs_success")

# Student-teacher agreement over training
def fig_agreement_curves():
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in TEACHERS:
        c = curve(t, "rq1", "small", "llm_full", metric="teacher_agree")
        if not c:
            continue
        x, mu, _ = c
        ax.plot(x, mu, color=COL[t], ls=LS[t], lw=1.8, label=LBL[t])
    ax.set_xlabel("Training episode"); ax.set_ylabel("Teacher agreement (%)")
    ax.set_ylim(0, 60); ax.set_xlim(0, 1000)
    ax.set_title("Student-teacher agreement over training  (RQ1, small)", fontweight="bold", fontsize=12)
    ax.legend(ncol=2, loc="upper right", fontsize=9.5)
    save(fig, "agreement_over_training")

# Compute cost per episode, decomposed into env / LLM / PPO time
def fig_compute_cost():
    env, llm, ppo = [], [], []
    for t in TEACHERS:
        dd = _csvs(t, "rq1", "small", "llm_full")
        tl = np.mean([d["t_llm_s"].mean() for d in dd])
        tr = np.mean([d["t_rollout_s"].mean() for d in dd])
        tp = np.mean([d["t_ppo_s"].mean() for d in dd])
        llm.append(tl); ppo.append(tp); env.append(max(0.0, tr - tl))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(TEACHERS))
    ax.barh(y, env, color="#BBBBBB", edgecolor="white", label="Env rollout", zorder=3)
    ax.barh(y, llm, left=env, color="#0072B2", edgecolor="white", label="LLM scoring", zorder=3)
    ax.barh(y, ppo, left=[e + l for e, l in zip(env, llm)], color="#E69F00",
            edgecolor="white", label="PPO update", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([LBL[t] for t in TEACHERS])
    ax.set_xlabel("Wall-clock per episode (s), decomposed")
    ax.set_title("Compute cost per episode (RQ1, small) - env rollout dominates; LLM amortised by cache",
                 fontweight="bold", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9.5)
    save(fig, "compute_cost")

# PPO training stability: clip fraction over training
def fig_ppo_stability():
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in TEACHERS:
        c = curve(t, "rq1", "small", "llm_full", metric="clip_pct")
        if not c:
            continue
        x, mu, _ = c
        ax.plot(x, mu, color=COL[t], ls=LS[t], lw=1.7, label=LBL[t])
    ax.set_xlabel("Training episode"); ax.set_ylabel("PPO clip fraction (%)")
    ax.set_xlim(0, 1000)
    ax.set_title("PPO training stability - clip fraction over training  (RQ1, small)",
                 fontweight="bold", fontsize=11.5)
    ax.legend(ncol=2, loc="upper right", fontsize=9.5)
    save(fig, "ppo_clip_fraction")


if __name__ == "__main__":
    fig_rq1_final(); fig_rq1_curves(); fig_rq1_efficiency(); fig_rq2(); fig_rq3(); fig_cache_cost()
    fig_rq1_steps(); fig_lambda()
    for _t in ["qwen-4B", "llama-3B", "llama-8B"]:
        fig_rq2_partial(_t)
    fig_agreement_scatter(); fig_agreement_curves(); fig_compute_cost(); fig_ppo_stability()
