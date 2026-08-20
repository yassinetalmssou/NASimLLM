"""Figures for the teacher-free deployment study (promotor feedback p23/p47).

  teacher_agreement_over_training.png : does the student imitate the teacher?
  student_deployment.png              : teacher-free success, greedy vs stochastic

Both are written to student_only/figures/ and to the thesis figures directory.
Run fig_agreement any time; run fig_deploy after run_student_only.py has produced
student_only/deploy_summary.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

BASE = r"c:/Users/yassi/Documents/VUB/Master/2MA/Thesis/NASimLLM"
RUNS = os.path.join(BASE, "runs")
OUT = os.path.join(BASE, "student_only")
FIGDIRS = [os.path.join(OUT, "figures"),
           os.path.join(BASE, "Thesis_Draft", "thesis", "figures")]

TEACHERS = ["qwen-4B", "qwen-8B", "llama-1B", "llama-3B", "llama-8B"]
SEEDS = [0, 1, 2]
POM_RUNS = os.path.join(BASE, "hpc", "results", "pomdp")
POM_OUT = os.path.join(BASE, "student_only", "pomdp")
POM_SEEDS = [0, 1, 2, 3, 4, 5]
TCOL = {"qwen-4B": "#0072B2", "qwen-8B": "#56B4E9", "llama-1B": "#009E73",
        "llama-3B": "#D55E00", "llama-8B": "#CC79A7"}
TLAB = {"qwen-4B": "Qwen-4B", "qwen-8B": "Qwen-8B", "llama-1B": "Llama-1B",
        "llama-3B": "Llama-3B", "llama-8B": "Llama-8B"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def save(fig, name):
    sub = "pomdp" if name.endswith("_pomdp.png") else "full_obs"
    dirs = [os.path.join(OUT, "figures"),
            os.path.join(BASE, "Thesis_Draft", "thesis", "figures", sub)]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name))
    plt.close(fig)
    print("  wrote", name)


def _seed_curves(t, metric, win=75):
    rows = []
    for s in SEEDS:
        p = f"{RUNS}/{t}/rq1/small/llm_full/seed{s}/train.csv"
        if os.path.exists(p):
            d = pd.read_csv(p)
            if metric in d:
                rows.append(pd.Series(d[metric].values))
    if not rows:
        return None
    L = min(len(r) for r in rows)
    R = np.vstack([r.values[:L] for r in rows])
    roll = np.vstack([pd.Series(R[i]).rolling(win, min_periods=10).mean().values
                      for i in range(R.shape[0])])
    return np.arange(L), np.nanmean(roll, axis=0)


def fig_agreement():
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in TEACHERS:
        c = _seed_curves(t, "teacher_agree")
        if c is None:
            continue
        x, m = c
        ax.plot(x, m, color=TCOL[t], lw=1.8, label=TLAB[t])
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 2000)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Student-teacher option agreement")
    ax.set_title("Agreement with the teacher over training")
    ax.axhspan(0, 20, color="#000000", alpha=0.03, lw=0)
    ax.text(1980, 8, "chance (5 options) = 20%", ha="right", va="center", fontsize=9, color="#6a6a6a")
    ax.axhline(20, color="#9a9a9a", ls=":", lw=1.1)
    ax.legend(ncol=3, loc="upper right", fontsize=9.5)
    save(fig, "teacher_agreement_over_training.png")


def fig_deploy():
    csv = os.path.join(OUT, "deploy_summary.csv")
    if not os.path.exists(csv):
        print("  (deploy_summary.csv not found; run run_student_only.py first)")
        return
    from matplotlib.lines import Line2D
    df = pd.read_csv(csv)
    rows = []
    for t in TEACHERS:
        g = df[(df.teacher == t) & (df["mode"] == "greedy")]["success_mean"]
        s = df[(df.teacher == t) & (df["mode"] == "sample")]
        if not len(s):
            continue
        gx = float(g.iloc[0]) * 100 if len(g) else 0.0
        sx = float(s["success_mean"].iloc[0]) * 100
        ci = float(s["success_ci"].iloc[0]) * 100
        rows.append((t, gx, sx, ci))
    rows.sort(key=lambda r: r[2])  # highest stochastic ends at the top

    fig, ax = plt.subplots(figsize=(9, 5))
    for y, (t, gx, sx, ci) in enumerate(rows):
        ax.annotate("", xy=(sx, y), xytext=(gx, y),
                    arrowprops=dict(arrowstyle="-|>", color="#cbcbcb", lw=2.6,
                                    shrinkA=0, shrinkB=0), zorder=2)
        ax.plot(gx, y, "o", ms=9, color="#4D4D4D", zorder=4)
        lo, hi = min(ci, sx), min(ci, 100 - sx)
        ax.errorbar(sx, y, xerr=[[lo], [hi]], fmt="o", ms=13, color="#0072B2",
                    ecolor="#0072B2", elinewidth=1.6, capsize=3, zorder=5)
        ax.text(sx + 3, y, f"{sx:.0f}%", va="center", ha="left",
                fontweight="bold", fontsize=11.5, color="#0072B2")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([TLAB[r[0]] for r in rows])
    ax.set_xlim(-4, 112)
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Teacher-free success rate")
    ax.set_title("Deployment without the teacher")
    ax.text(0, len(rows) - 0.45, "greedy: 0%", ha="center", va="bottom",
            fontsize=9.5, color="#4D4D4D")
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#4D4D4D", ms=9,
                  label="Greedy (arg-max option)"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", ms=12,
                  label="Stochastic (sampled option)")]
    ax.legend(handles=leg, loc="lower right", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    save(fig, "student_deployment.png")


def fig_deploy_pomdp():
    csv = os.path.join(POM_OUT, "deploy_summary.csv")
    if not os.path.exists(csv):
        print("  (pomdp deploy_summary.csv not found; run run_student_only_pomdp.py first)")
        return
    from matplotlib.lines import Line2D
    df = pd.read_csv(csv)
    df = df[df["scenario"] == "small"]
    rows = []
    for t in TEACHERS:
        g = df[(df.teacher == t) & (df["mode"] == "greedy")]["success_mean"]
        s = df[(df.teacher == t) & (df["mode"] == "sample")]
        if not len(s):
            continue
        gx = float(g.iloc[0]) * 100 if len(g) else 0.0
        sx = float(s["success_mean"].iloc[0]) * 100
        ci = float(s["success_ci"].iloc[0]) * 100
        rows.append((t, gx, sx, ci))
    rows.sort(key=lambda r: r[2])

    fig, ax = plt.subplots(figsize=(9, 5))
    for y, (t, gx, sx, ci) in enumerate(rows):
        ax.annotate("", xy=(sx, y), xytext=(gx, y),
                    arrowprops=dict(arrowstyle="-|>", color="#cbcbcb", lw=2.6,
                                    shrinkA=0, shrinkB=0), zorder=2)
        ax.plot(gx, y, "o", ms=9, color="#4D4D4D", zorder=4)
        lo, hi = min(ci, sx), min(ci, 100 - sx)
        ax.errorbar(sx, y, xerr=[[lo], [hi]], fmt="o", ms=13, color="#0072B2",
                    ecolor="#0072B2", elinewidth=1.6, capsize=3, zorder=5)
        ax.text(sx + 3, y, f"{sx:.0f}%", va="center", ha="left",
                fontweight="bold", fontsize=11.5, color="#0072B2")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([TLAB[r[0]] for r in rows])
    ax.set_xlim(-4, 112)
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Teacher-free success rate")
    ax.set_title("Deployment without the teacher (POMDP)")
    ax.text(0, len(rows) - 0.45, "greedy: 0%", ha="center", va="bottom",
            fontsize=9.5, color="#4D4D4D")
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#4D4D4D", ms=9,
                  label="Greedy (arg-max option)"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", ms=12,
                  label="Stochastic (sampled option)")]
    ax.legend(handles=leg, loc="lower right", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    save(fig, "student_deployment_pomdp.png")


def fig_agreement_pomdp():
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in TEACHERS:
        rows = []
        for s in POM_SEEDS:
            p = f"{POM_RUNS}/{t}/rq1/small/llm_full/seed{s}/train.csv"
            if os.path.exists(p):
                d = pd.read_csv(p)
                if "teacher_agree" in d:
                    rows.append(pd.Series(d["teacher_agree"].values))
        if not rows:
            continue
        L = min(len(r) for r in rows)
        R = np.vstack([r.values[:L] for r in rows])
        roll = np.vstack([pd.Series(R[i]).rolling(75, min_periods=10).mean().values
                          for i in range(R.shape[0])])
        ax.plot(np.arange(L), np.nanmean(roll, axis=0), color=TCOL[t], lw=1.8, label=TLAB[t])
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 2000)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Student-teacher option agreement")
    ax.set_title("Agreement with the teacher over training (POMDP)")
    ax.axhspan(0, 20, color="#000000", alpha=0.03, lw=0)
    ax.text(1980, 8, "chance (5 options) = 20%", ha="right", va="center", fontsize=9, color="#6a6a6a")
    ax.axhline(20, color="#9a9a9a", ls=":", lw=1.1)
    ax.legend(ncol=3, loc="upper right", fontsize=9.5)
    save(fig, "teacher_agreement_over_training_pomdp.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "agreement"):
        fig_agreement()
    if which in ("all", "deploy"):
        fig_deploy()
    if which in ("all", "pomdp"):
        fig_deploy_pomdp()
        fig_agreement_pomdp()
    print("done")
