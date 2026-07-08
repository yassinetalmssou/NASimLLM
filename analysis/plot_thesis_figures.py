"""Plot thesis figures for NASimLLM.

Generates three publication-quality figures from results.csv:
  rq1_effectiveness  — dot-and-CI plot (success rate per scenario/condition)
  rq2_generalization — slope graph (LLM-full vs PPO across scenarios)
  rq3_ablations      — horizontal tornado chart (vs full baseline)

Figures are saved as PNG (300 dpi) and PDF in --out-dir.

Usage (synthetic demo):
  python analysis/plot_thesis_figures.py --demo

Usage (real data):
  python analysis/plot_thesis_figures.py --results results.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# ── Okabe-Ito palette ─────────────────────────────────────────────────────────
PALETTE = {
    "llm_full":        "#0072B2",
    "ppo_options":     "#D55E00",
    "random":          "#999999",
    "bruteforce":      "#F0E442",
    # RQ3 ablations
    "full":            "#0072B2",
    "no_history":      "#E69F00",
    "no_avoidlist":    "#009E73",
    "verbose_prompt":  "#CC79A7",
    "llm_cached":      "#56B4E9",
}

SCENARIO_ORDER = ["tiny", "small", "small-linear", "medium"]

# ── Synthetic demo data ───────────────────────────────────────────────────────

def _make_demo_df() -> pd.DataFrame:
    """Generate plausible synthetic data for all three RQs."""
    records = []
    rng = np.random.default_rng(42)

    # RQ1 — llm_full vs ppo_options vs random
    rq1_rates = {
        "tiny":         {"llm_full": 0.90, "ppo_options": 0.82, "random": 0.15},
        "small":        {"llm_full": 0.76, "ppo_options": 0.67, "random": 0.08},
        "small-linear": {"llm_full": 0.71, "ppo_options": 0.60, "random": 0.05},
        "medium":       {"llm_full": 0.55, "ppo_options": 0.46, "random": 0.02},
    }
    for scenario, conds in rq1_rates.items():
        for cond, rate in conds.items():
            for seed in range(5):
                n = 200
                hits = rng.binomial(n, rate)
                records.append({
                    "rq": "rq1", "model": "demo", "scenario": scenario,
                    "condition": cond, "train_seed": seed,
                    "success_rate": hits / n,
                    "success_ci_low":  max(0, rate - 0.06),
                    "success_ci_high": min(1, rate + 0.06),
                    "mean_return": hits * 0.5,
                    "std_return": 2.0,
                    "median_steps_to_success": 40 if hits > 0 else None,
                    "eval_episodes": n,
                    "policy": "checkpoint",
                })

    # RQ2 — generalisation (same scenarios, model only trained on tiny)
    for scenario, conds in rq1_rates.items():
        for cond in ["llm_full", "ppo_options"]:
            rate = conds[cond] * (0.9 if scenario != "tiny" else 1.0)
            n = 200
            hits = rng.binomial(n, rate)
            records.append({
                "rq": "rq2", "model": "demo", "scenario": scenario,
                "condition": cond, "train_seed": 0,
                "success_rate": hits / n,
                "success_ci_low":  max(0, rate - 0.07),
                "success_ci_high": min(1, rate + 0.07),
                "mean_return": hits * 0.5,
                "std_return": 2.0,
                "median_steps_to_success": None,
                "eval_episodes": n,
                "policy": "checkpoint",
            })

    # RQ3 ablations (tiny only)
    ablation_rates = {
        "full":           0.88,
        "no_history":     0.73,
        "no_avoidlist":   0.79,
        "verbose_prompt": 0.82,
        "llm_cached":     0.85,
    }
    for cond, rate in ablation_rates.items():
        for seed in range(5):
            n = 200
            hits = rng.binomial(n, rate)
            records.append({
                "rq": "rq3", "model": "demo", "scenario": "tiny",
                "condition": cond, "train_seed": seed,
                "success_rate": hits / n,
                "success_ci_low":  max(0, rate - 0.06),
                "success_ci_high": min(1, rate + 0.06),
                "mean_return": hits * 0.5,
                "std_return": 2.0,
                "median_steps_to_success": None,
                "eval_episodes": n,
                "policy": "checkpoint",
            })

    return pd.DataFrame(records)


# ── Saving helper ─────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ── Figure 1 — RQ1 Effectiveness ─────────────────────────────────────────────

def plot_rq1(df: pd.DataFrame, out_dir: Path):
    rq1 = df[df["rq"] == "rq1"].copy()
    conditions = ["llm_full", "ppo_options", "random"]
    present = [c for c in conditions if c in rq1["condition"].unique()]

    # Aggregate over seeds: mean success_rate + min/max CI
    agg = (
        rq1.groupby(["scenario", "condition"])
           .agg(
               sr_mean=("success_rate", "mean"),
               ci_lo=("success_ci_low", "mean"),
               ci_hi=("success_ci_high", "mean"),
           )
           .reset_index()
    )

    scenarios = [s for s in SCENARIO_ORDER if s in agg["scenario"].unique()]
    n_scen = len(scenarios)

    fig, ax = plt.subplots(figsize=(7, 4))
    offsets = np.linspace(-0.15, 0.15, len(present))
    x = np.arange(n_scen)

    for idx, cond in enumerate(present):
        sub = agg[agg["condition"] == cond].set_index("scenario")
        ys    = [sub.loc[s, "sr_mean"] if s in sub.index else np.nan for s in scenarios]
        lo    = [sub.loc[s, "ci_lo"]   if s in sub.index else np.nan for s in scenarios]
        hi    = [sub.loc[s, "ci_hi"]   if s in sub.index else np.nan for s in scenarios]
        yerr_lo = [max(0.0, y - l) if not math.isnan(y) else 0 for y, l in zip(ys, lo)]
        yerr_hi = [max(0.0, h - y) if not math.isnan(y) else 0 for y, h in zip(ys, hi)]

        ax.errorbar(
            x + offsets[idx], ys,
            yerr=[yerr_lo, yerr_hi],
            fmt="o", markersize=7, capsize=4, linewidth=1.5,
            color=PALETTE.get(cond, "#333333"),
            label=cond.replace("_", " "),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Success rate")
    ax.set_title("RQ1 — Effectiveness of LLM-guided training")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "rq1_effectiveness", out_dir)


# ── Figure 2 — RQ2 Generalisation ────────────────────────────────────────────

def plot_rq2(df: pd.DataFrame, out_dir: Path):
    rq2 = df[df["rq"] == "rq2"].copy()
    conditions = ["llm_full", "ppo_options"]
    present = [c for c in conditions if c in rq2["condition"].unique()]

    agg = (
        rq2.groupby(["scenario", "condition"])["success_rate"]
           .mean()
           .reset_index()
    )

    scenarios = [s for s in SCENARIO_ORDER if s in agg["scenario"].unique()]

    fig, ax = plt.subplots(figsize=(6, 4))
    for cond in present:
        sub = agg[agg["condition"] == cond].set_index("scenario")
        ys = [sub.loc[s, "success_rate"] if s in sub.index else np.nan for s in scenarios]
        ax.plot(
            scenarios, ys,
            marker="o", linewidth=2, markersize=7,
            color=PALETTE.get(cond, "#333333"),
            label=cond.replace("_", " "),
        )

    # Shade the two middle topologies
    if "small" in scenarios and "small-linear" in scenarios:
        idx_s  = scenarios.index("small")
        idx_sl = scenarios.index("small-linear")
        ax.axvspan(idx_s - 0.5, idx_sl + 0.5, color="#DDDDDD", alpha=0.4, zorder=0,
                   label="different topology")

    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Scenario")
    ax.set_title("RQ2 — Generalisation across scenarios")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "rq2_generalization", out_dir)


# ── Figure 3 — RQ3 Ablations ─────────────────────────────────────────────────

def plot_rq3(df: pd.DataFrame, out_dir: Path):
    rq3 = df[df["rq"] == "rq3"].copy()
    ablation_order = ["no_history", "no_avoidlist", "verbose_prompt", "llm_cached"]
    ablation_labels = {
        "no_history":     "No history",
        "no_avoidlist":   "No avoid list",
        "verbose_prompt": "Verbose prompt",
        "llm_cached":     "Cached LLM",
    }

    agg = (
        rq3.groupby(["scenario", "condition"])["success_rate"]
           .mean()
           .reset_index()
    )

    scenarios = [s for s in SCENARIO_ORDER if s in agg["scenario"].unique()]
    if not scenarios:
        print("  [RQ3] No data — skipping figure.")
        return

    fig, axes = plt.subplots(
        1, len(scenarios), figsize=(4 * len(scenarios), 4), sharey=True,
    )
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        sub = agg[agg["scenario"] == scenario].set_index("condition")
        if "full" not in sub.index:
            ax.set_title(scenario)
            ax.set_xlabel("Δ success rate")
            continue

        baseline = sub.loc["full", "success_rate"]
        present_abl = [a for a in ablation_order if a in sub.index]
        deltas = [sub.loc[a, "success_rate"] - baseline for a in present_abl]
        labels = [ablation_labels.get(a, a) for a in present_abl]
        colors = [PALETTE.get(a, "#AAAAAA") for a in present_abl]
        y = np.arange(len(present_abl))

        ax.barh(y, deltas, color=colors, height=0.55, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Δ success rate")
        ax.set_title(scenario)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Ablation condition")
    fig.suptitle("RQ3 — Ablation analysis (vs. full LLM)", y=1.02)
    fig.tight_layout()
    _save(fig, "rq3_ablations", out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures for NASimLLM")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--results", type=str, default="results.csv",
                     help="Path to aggregated results.csv (default: results.csv)")
    src.add_argument("--demo", action="store_true",
                     help="Use synthetic demo data (no results.csv needed)")
    parser.add_argument("--out-dir", type=str, default="figures",
                        help="Output directory for figures (default: figures/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.demo:
        print("Using synthetic demo data …")
        df = _make_demo_df()
    else:
        p = Path(args.results)
        if not p.exists():
            sys.exit(f"ERROR: {p} not found. Run build_results_table.py first, or pass --demo.")
        df = pd.read_csv(p)

    print("Plotting RQ1 …")
    plot_rq1(df, out_dir)
    print("Plotting RQ2 …")
    plot_rq2(df, out_dir)
    print("Plotting RQ3 …")
    plot_rq3(df, out_dir)
    print("All figures written to", out_dir)


if __name__ == "__main__":
    main()
