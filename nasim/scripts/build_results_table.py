"""Aggregate eval_summary.json files into a long-format results.csv.

Walk a runs tree, find every eval_summary.json, emit one row per run.

Usage:
  python -m nasim.scripts.build_results_table --root runs/ --output results.csv

For the random baseline pass:
  python -m nasim.scripts.build_results_table --root runs/ \\
      --output results_random.csv --summary-name eval_summary_random.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)

# ── Known labels ──────────────────────────────────────────────────────────────
_ABLATIONS   = {"full", "no_history", "no_avoidlist", "verbose_prompt", "llm_cached"}
_CONDITIONS  = {"llm_full", "ppo_options", "random", "bruteforce"} | _ABLATIONS

_OUTPUT_FIELDS = [
    "model", "rq", "scenario", "condition", "train_seed",
    "success_rate", "success_ci_low", "success_ci_high",
    "mean_return", "std_return", "median_steps_to_success",
    "policy", "eval_episodes", "run_dir",
]


# ── Inference helpers ─────────────────────────────────────────────────────────

def _infer_rq(run_dir: Path, root: Path) -> str:
    try:
        rel_parts = run_dir.relative_to(root).parts
    except ValueError:
        rel_parts = run_dir.parts
    for p in rel_parts:
        pl = p.lower()
        if "rq1" in pl:
            return "rq1"
        if "rq2" in pl:
            return "rq2"
        if "rq3b" in pl or "rq3" in pl:
            return "rq3"
    return "unknown"


def _infer_model(run_dir: Path, root: Path) -> str:
    """Model name = first path component under root.

    Robust to both invocations:
      * --root runs           → parts = (model, rqN, scenario, condition, seedN)
      * --root runs/<model>   → parts = (rqN, scenario, condition, seedN)
    In the second case parts[0] is an RQ label, so the model is the root dir itself.
    """
    try:
        parts = run_dir.relative_to(root).parts
    except ValueError:
        return run_dir.name
    if not parts:
        return root.name
    if re.match(r"rq\d", parts[0].lower()):
        return root.name
    return parts[0]


def _infer_condition(run_dir: Path, merged: dict) -> str:
    """Prefer config 'condition' → known token in path → use_llm fallback."""
    # 1. Explicit field written by the trainer (Task 0 fix)
    if merged.get("condition") not in (None, "unknown", ""):
        return merged["condition"]

    # 2. Scan path components for a known condition token
    for part in run_dir.parts:
        if part in _CONDITIONS:
            return part

    # 3. Fall back to use_llm
    return "llm_full" if merged.get("use_llm", False) else "ppo_options"


def _infer_seed(run_dir: Path, config: dict) -> str:
    if "seed" in config:
        return str(config["seed"])
    if "train_seed" in config:
        return str(config["train_seed"])
    for part in run_dir.parts:
        m = re.match(r"seed(\d+)$", part)
        if m:
            return m.group(1)
    return ""


# ── Tree walker ───────────────────────────────────────────────────────────────

def process_tree(root: Path, summary_name: str = "eval_summary.json") -> list[dict]:
    rows: list[dict] = []
    conditions_seen: set[str] = set()
    rq3_bad: list[str] = []

    for summary_path in sorted(root.rglob(summary_name)):
        run_dir    = summary_path.parent
        config_path = run_dir / "config.json"

        try:
            with open(summary_path, encoding="utf-8") as fh:
                summary = json.load(fh)
        except Exception as exc:
            logger.warning(f"Cannot read {summary_path}: {exc}")
            continue

        config: dict = {}
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as fh:
                    config = json.load(fh)
            except Exception:
                pass

        # Merge: summary fields override config for condition/use_llm
        merged = {**config, **{k: v for k, v in summary.items() if k in ("condition", "use_llm")}}
        condition = _infer_condition(run_dir, merged)
        conditions_seen.add(condition)

        rq = _infer_rq(run_dir, root)
        if rq == "rq3" and condition == "llm_full":
            rq3_bad.append(str(run_dir))

        rows.append({
            "model":                   _infer_model(run_dir, root),
            "rq":                      rq,
            "scenario":                summary.get("scenario", config.get("scenario", "")),
            "condition":               condition,
            "train_seed":              summary.get("train_seed", _infer_seed(run_dir, config)),
            "success_rate":            summary.get("success_rate", ""),
            "success_ci_low":          summary.get("success_ci_low", ""),
            "success_ci_high":         summary.get("success_ci_high", ""),
            "mean_return":             summary.get("mean_return", ""),
            "std_return":              summary.get("std_return", ""),
            "median_steps_to_success": summary.get("median_steps_to_success", ""),
            "policy":                  summary.get("policy", "checkpoint"),
            "eval_episodes":           summary.get("eval_episodes", ""),
            "run_dir":                 str(run_dir),
        })

    logger.info(f"Conditions seen: {sorted(conditions_seen)}")
    for bad in rq3_bad:
        logger.warning(
            "RQ3 run resolved to 'llm_full' — ablation label missing from config.json "
            f"and not found in path. Add --condition flag to training command. Run: {bad}"
        )

    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate eval_summary.json files into long-format results.csv"
    )
    parser.add_argument("--root", required=True,
                        help="Root directory of the run tree")
    parser.add_argument("--output", default="results.csv",
                        help="Output CSV path (default: results.csv)")
    parser.add_argument("--summary-name", default="eval_summary.json",
                        help="Summary filename to search for (default: eval_summary.json)")
    args = parser.parse_args()

    root   = Path(args.root)
    output = Path(args.output)

    rows = process_tree(root, summary_name=args.summary_name)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} rows → {output}")


if __name__ == "__main__":
    main()
