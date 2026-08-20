"""Reward-shaping on/off ablation analysis (promotor: "did you run before/after?").

Compares the full system (enhanced reward shaping) against the raw-NASim-reward
condition (--no-shaping), same scenario / hyperparameters / seeds, Llama-3.2-1B teacher.

Reads the run layout produced by run_rq3b.py:
    <root>/<scenario>/<condition>/seed<N>/train.csv

Reports, per condition (mean +/- 95% CI across seeds, over the last WINDOW episodes):
  - success rate            (native mission completion; comparable across conditions)
  - host-compromise rate    (native)
  - episode length
  - sample efficiency       (first episode with >=80% success over a 50-ep window)

Note: the per-episode `reward` column is the SHAPED return for the full condition and the
RAW return for no_shaping, so it is NOT comparable between conditions and is not used as the
headline metric. Success/compromise are computed on the native reward and are comparable.
"""
import os
import sys
import glob
import csv
import io
import math

ROOT = sys.argv[1] if len(sys.argv) > 1 else "student_only/reward_shaping_ablation"
SCENARIO = sys.argv[2] if len(sys.argv) > 2 else "small"
WINDOW = 100          # final-metric averaging window (last N episodes), matches thesis
SE_WINDOW = 50        # sample-efficiency rolling window
SE_TARGET = 0.80      # sample-efficiency success target
MIN_EPISODES = 350    # skip runs still in progress (target is 400); avoids polluting the aggregate


def _read_csv(path):
    with io.open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row, key, default=float("nan")):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def sample_efficiency(succ):
    """First episode index (1-based) where the rolling SE_WINDOW success mean >= SE_TARGET."""
    for i in range(len(succ)):
        lo = max(0, i - SE_WINDOW + 1)
        w = succ[lo:i + 1]
        if len(w) >= SE_WINDOW and (sum(w) / len(w)) >= SE_TARGET:
            return i + 1
    return None


def seed_metrics(path):
    rows = _read_csv(path)
    if not rows:
        return None
    succ = [1.0 if _f(r, "success", 0) >= 0.5 else 0.0 for r in rows]
    tail = rows[-WINDOW:]
    return {
        "episodes": len(rows),
        "success":  _mean([_f(r, "success") for r in tail]) * 100,
        "compromise": _mean([_f(r, "compromise_rate") for r in tail]) * 100,
        "length":   _mean([_f(r, "length") for r in tail]),
        "reward":   _mean([_f(r, "reward") for r in tail]),
        "sample_eff": sample_efficiency(succ),
        "t_ep":     _mean([_f(r, "t_episode_s") for r in rows]),
    }


def ci95(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return 1.96 * sd / math.sqrt(len(xs))


def summarize(condition):
    seeds = sorted(glob.glob(os.path.join(ROOT, SCENARIO, condition, "seed*", "train.csv")))
    per = []
    for p in seeds:
        m = seed_metrics(p)
        if not m:
            continue
        sname = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if m["episodes"] < MIN_EPISODES:
            print(f"  [skip] {condition}/{sname}: only {m['episodes']} episodes (< {MIN_EPISODES}, still running)")
            continue
        per.append((sname, m))
    return per


def agg(per, key):
    vals = [m[key] for _, m in per if m[key] is not None and not (isinstance(m[key], float) and math.isnan(m[key]))]
    if not vals:
        return float("nan"), 0.0, 0
    return sum(vals) / len(vals), ci95(vals), len(vals)


if __name__ == "__main__":
    print(f"Reward-shaping ablation  |  root={ROOT}  scenario={SCENARIO}")
    print(f"(final metrics = mean over last {WINDOW} episodes; CI across seeds)\n")
    conds = ["full", "no_shaping"]
    results = {}
    for cond in conds:
        per = summarize(cond)
        results[cond] = per
        if not per:
            print(f"[{cond}] no runs found under {os.path.join(ROOT, SCENARIO, cond)}")
            continue
        print(f"=== {cond}  ({len(per)} seed(s)) ===")
        for sname, m in per:
            se = m["sample_eff"]
            print(f"  {sname}: eps={m['episodes']:>4}  success={m['success']:5.1f}%  "
                  f"compromise={m['compromise']:5.1f}%  len={m['length']:5.1f}  "
                  f"sample_eff={se if se is not None else 'n/a':>5}  t_ep={m['t_ep']:.2f}s")
        for label, key, unit in [("success", "success", "%"), ("compromise", "compromise", "%"),
                                 ("episode length", "length", ""), ("sample_eff (ep to 80%)", "sample_eff", "")]:
            mean, ci, n = agg(per, key)
            print(f"  MEAN {label:24}: {mean:6.1f}{unit}  +/- {ci:4.1f}  (n={n})")
        print()

    if all(results.get(c) for c in conds):
        sf, _, _ = agg(results["full"], "success")
        sn, _, _ = agg(results["no_shaping"], "success")
        print(f"HEADLINE: success  full(shaped)={sf:.1f}%  vs  no_shaping(raw)={sn:.1f}%  "
              f"=>  shaping delta = {sf - sn:+.1f} pp")
