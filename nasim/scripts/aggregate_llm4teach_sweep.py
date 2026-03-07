"""Aggregate a LLM4Teach sweep directory into a single summary CSV.

Reads:
  <sweep_dir>/manifest.jsonl          — written by run_llm4teach_sweep
  <run_dir>/train.csv (per run)       — written by the trainer directly

Writes:
  A single CSV with one row per run: config fields + summary statistics
  (mean/max/final/last-k reward, success rates, loss, etc.)

Usage
-----
  python -m nasim.scripts.aggregate_llm4teach_sweep \\
    --sweep-dir runs/my_sweep \\
    --output   runs/my_sweep/summary.csv \\
    --last-k   20

Notes
-----
- Missing metrics are left blank in the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


def _float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _safe_mean(xs: List[float]) -> Optional[float]:
    return mean(xs) if xs else None


def _safe_std(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return 0.0 if len(xs) == 1 else None
    return pstdev(xs)


def _last_k(xs: List[float], k: int) -> List[float]:
    if k <= 0:
        return xs
    return xs[-k:] if len(xs) >= k else xs


def _group_key(config: Dict[str, Any]) -> str:
    cfg = dict(config)
    cfg.pop("seed", None)
    payload = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


@dataclass
class RunSummary:
    run_id: str
    group_key: str
    exit_code: Optional[int]
    parser_exit_code: Optional[int]

    scenario: Optional[str]
    seed: Optional[int]
    episodes: Optional[int]
    episode_length: Optional[int]

    learning_rate: Optional[float]
    lambda_start: Optional[float]
    lambda_decay: Optional[float]
    lambda_mode: Optional[str]
    batch_size: Optional[int]
    num_epochs: Optional[int]
    hidden_dim: Optional[int]

    fully_obs: Optional[bool]
    use_llm: Optional[bool]
    device: Optional[str]

    n_episodes_parsed: int

    final_reward: Optional[float]
    max_reward: Optional[float]
    mean_reward: Optional[float]
    mean_reward_last_k: Optional[float]
    std_reward_last_k: Optional[float]

    success_rate_overall: Optional[float]
    success_rate_last_k: Optional[float]

    mean_compromise_overall: Optional[float]
    mean_discovery_overall: Optional[float]

    mean_loss_actor_last_k: Optional[float]
    mean_loss_kl_last_k: Optional[float]
    mean_loss_critic_last_k: Optional[float]
    mean_clip_last_k: Optional[float]

    final_lambda: Optional[float]

    run_dir: str
    log_path: str
    csv_path: str
    command: str


def _read_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            # Skip malformed lines
            continue
    return items


def _read_train_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _summarize_rows(rows: List[Dict[str, Any]], last_k: int) -> Dict[str, Any]:
    rewards: List[float] = []
    compromise: List[float] = []
    disc: List[float] = []
    loss_actor: List[float] = []
    loss_kl: List[float] = []
    loss_critic: List[float] = []
    clip: List[float] = []
    lambdas: List[float] = []
    success_flags: List[int] = []

    for row in rows:
        r = _float_or_none(row.get("reward"))
        if r is not None:
            rewards.append(r)

        hc = _int_or_none(row.get("hosts_compromised"))
        ht = _int_or_none(row.get("hosts_total"))
        if hc is not None:
            compromise.append(float(hc))
        if hc is not None and ht is not None and ht > 0:
            success_flags.append(1 if hc == ht else 0)

        d = _float_or_none(row.get("discovery_pct"))
        if d is not None:
            disc.append(d)

        la = _float_or_none(row.get("loss_actor"))
        if la is not None:
            loss_actor.append(la)

        lkl = _float_or_none(row.get("loss_kl_norm"))
        if lkl is not None:
            loss_kl.append(lkl)

        lc = _float_or_none(row.get("loss_critic"))
        if lc is not None:
            loss_critic.append(lc)

        c = _float_or_none(row.get("clip_pct"))
        if c is not None:
            clip.append(c)

        lam = _float_or_none(row.get("lambda_kl"))
        if lam is not None:
            lambdas.append(lam)

    rewards_last = _last_k(rewards, last_k)
    success_last = _last_k([float(x) for x in success_flags], last_k)

    return {
        "n_episodes_parsed":       len(rows),
        "final_reward":            rewards[-1] if rewards else None,
        "max_reward":              max(rewards) if rewards else None,
        "mean_reward":             _safe_mean(rewards),
        "mean_reward_last_k":      _safe_mean(rewards_last),
        "std_reward_last_k":       _safe_std(rewards_last),
        "success_rate_overall":    _safe_mean([float(x) for x in success_flags]) if success_flags else None,
        "success_rate_last_k":     _safe_mean(success_last) if success_last else None,
        "mean_compromise_overall": _safe_mean(compromise),
        "mean_discovery_overall":  _safe_mean(disc),
        "mean_loss_actor_last_k":  _safe_mean(_last_k(loss_actor, last_k)),
        "mean_loss_kl_last_k":     _safe_mean(_last_k(loss_kl, last_k)),
        "mean_loss_critic_last_k": _safe_mean(_last_k(loss_critic, last_k)),
        "mean_clip_last_k":        _safe_mean(_last_k(clip, last_k)),
        "final_lambda":            lambdas[-1] if lambdas else None,
    }


def aggregate(sweep_dir: Path, output_csv: Path, last_k: int) -> None:
    manifest_path = sweep_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.jsonl not found in {sweep_dir}")

    manifest = _read_manifest(manifest_path)
    if not manifest:
        raise SystemExit(f"No entries found in {manifest_path}")

    summaries: List[RunSummary] = []

    for item in manifest:
        run_id = str(item.get("run_id", ""))
        run_dir = Path(str(item.get("run_dir", sweep_dir / run_id)))
        csv_path = Path(str(item.get("csv_path", run_dir / "train.csv")))

        cfg = item.get("config") or {}
        gk = _group_key(cfg) if isinstance(cfg, dict) else ""

        rows = _read_train_csv(csv_path)
        stats = _summarize_rows(rows, last_k=last_k)

        summaries.append(RunSummary(
            run_id=run_id,
            group_key=gk,
            exit_code=_int_or_none(item.get("exit_code")),
            parser_exit_code=_int_or_none(item.get("parser_exit_code")),
            scenario=cfg.get("scenario"),
            seed=_int_or_none(cfg.get("seed")),
            episodes=_int_or_none(cfg.get("episodes")),
            episode_length=_int_or_none(cfg.get("episode_length")),
            learning_rate=_float_or_none(cfg.get("learning_rate")),
            lambda_start=_float_or_none(cfg.get("lambda_start")),
            lambda_decay=_float_or_none(cfg.get("lambda_decay")),
            lambda_mode=cfg.get("lambda_mode"),
            batch_size=_int_or_none(cfg.get("batch_size")),
            num_epochs=_int_or_none(cfg.get("num_epochs")),
            hidden_dim=_int_or_none(cfg.get("hidden_dim")),
            fully_obs=cfg.get("fully_obs"),
            use_llm=cfg.get("use_llm"),
            device=cfg.get("device"),
            n_episodes_parsed=int(stats["n_episodes_parsed"]),
            final_reward=stats["final_reward"],
            max_reward=stats["max_reward"],
            mean_reward=stats["mean_reward"],
            mean_reward_last_k=stats["mean_reward_last_k"],
            std_reward_last_k=stats["std_reward_last_k"],
            success_rate_overall=stats["success_rate_overall"],
            success_rate_last_k=stats["success_rate_last_k"],
            mean_compromise_overall=stats["mean_compromise_overall"],
            mean_discovery_overall=stats["mean_discovery_overall"],
            mean_loss_actor_last_k=stats["mean_loss_actor_last_k"],
            mean_loss_kl_last_k=stats["mean_loss_kl_last_k"],
            mean_loss_critic_last_k=stats["mean_loss_critic_last_k"],
            mean_clip_last_k=stats["mean_clip_last_k"],
            final_lambda=stats["final_lambda"],
            run_dir=str(run_dir),
            log_path=str(item.get("log_path", run_dir / "train.log.txt")),
            csv_path=str(csv_path),
            command=str(item.get("command", "")),
        ))

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = list(RunSummary.__annotations__.keys())
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            w.writerow(s.__dict__)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate a LLM4Teach sweep into a summary CSV")
    p.add_argument("--sweep-dir", type=str, required=True, help="Directory containing manifest.jsonl")
    p.add_argument("--output", type=str, required=True, help="Output summary CSV path")
    p.add_argument("--last-k", type=int, default=20, help="Window for last-k stats (default: 20)")
    args = p.parse_args()

    aggregate(Path(args.sweep_dir), Path(args.output), last_k=int(args.last_k))
    print(f"Wrote summary -> {args.output}")


if __name__ == "__main__":
    main()
