"""Grid sweep runner for nasim.train_llm4teach.

Each run gets its own folder under <out-dir>/<run_id>/ containing:
  train.log.txt, train.csv, command.txt, ckpts/

Usage:
  python -m nasim.scripts.run_llm4teach_sweep \\
    --scenario small --episodes 200 --seeds 0 1 2 \\
    --learning-rates 1e-4 3e-4 --out-dir runs/my_sweep

Dry run (print commands only):
  python -m nasim.scripts.run_llm4teach_sweep --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SweepConfig:
    scenario: str
    episodes: int
    episode_length: int
    seed: int
    learning_rate: float
    lambda_start: float
    lambda_decay: float
    batch_size: int
    num_epochs: int
    hidden_dim: int
    fully_obs: bool
    use_llm: bool
    device: str
    llama_4bit: bool = False
    lambda_mode: str = "fixed"
    kl_target: float = 0.01
    lambda_min: float = 0.005
    competence_window: int = 10
    llm_mix_weight: float = 0.3
    deterministic_torch: bool = False


def _as_run_id(cfg: SweepConfig) -> str:
    payload = json.dumps(cfg.__dict__, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:10]
    mode = "llm" if cfg.use_llm else "no_llm"
    obs = "full" if cfg.fully_obs else "partial"
    lmode = f"_{cfg.lambda_mode}" if cfg.lambda_mode != "fixed" else ""
    return (
        f"{cfg.scenario}_{obs}_{mode}_seed{cfg.seed}_"
        f"lr{cfg.learning_rate:g}_ls{cfg.lambda_start:g}_ld{cfg.lambda_decay:g}_"
        f"bs{cfg.batch_size}_ep{cfg.num_epochs}_hd{cfg.hidden_dim}{lmode}_"
        f"{h}"
    )


def _cmd_for(cfg: SweepConfig, run_dir: Path) -> List[str]:
    ckpt_dir = run_dir / "ckpts"
    step_log_dir = run_dir / "step_logs"

    cmd: List[str] = [
        sys.executable,
        "-m",
        "nasim.train_llm4teach",
        "--scenario",
        cfg.scenario,
        "--episodes",
        str(cfg.episodes),
        "--episode-length",
        str(cfg.episode_length),
        "--seed",
        str(cfg.seed),
        "--learning-rate",
        str(cfg.learning_rate),
        "--lambda-start",
        str(cfg.lambda_start),
        "--lambda-decay",
        str(cfg.lambda_decay),
        "--batch-size",
        str(cfg.batch_size),
        "--num-epochs",
        str(cfg.num_epochs),
        "--hidden-dim",
        str(cfg.hidden_dim),
        "--device",
        cfg.device,
        "--save-dir",
        str(ckpt_dir),
        "--no-step-logs",
    ]

    if cfg.fully_obs:
        cmd.append("--fully-obs")
    else:
        cmd.append("--partial-obs")

    if not cfg.use_llm:
        cmd.append("--no-llm")

    if not cfg.llama_4bit:
        cmd.append("--no-llama-4bit")

    if cfg.lambda_mode != "fixed":
        cmd += ["--lambda-mode", cfg.lambda_mode,
                "--kl-target", str(cfg.kl_target),
                "--lambda-min", str(cfg.lambda_min),
                "--competence-window", str(cfg.competence_window)]

    if cfg.use_llm and cfg.llm_mix_weight != 0.3:
        cmd += ["--llm-mix-weight", str(cfg.llm_mix_weight)]

    if cfg.deterministic_torch:
        cmd.append("--deterministic-torch")

    return cmd


def _run_one(cfg: SweepConfig, out_dir: Path, dry_run: bool) -> Dict:
    run_id = _as_run_id(cfg)
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train.log.txt"
    csv_path = run_dir / "train.csv"
    cmd_path = run_dir / "command.txt"

    cmd = _cmd_for(cfg, run_dir)
    cmd_str = " ".join(cmd)
    cmd_path.write_text(cmd_str + "\n", encoding="utf-8")

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    if dry_run:
        return {
            "run_id": run_id,
            "config": cfg.__dict__,
            "command": cmd_str,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "csv_path": str(csv_path),
            "started_at": started_at,
            "ended_at": None,
            "exit_code": None,
            "parser_exit_code": None,
        }

    # Run training
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )

    ended_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "run_id": run_id,
        "config": cfg.__dict__,
        "command": cmd_str,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "csv_path": str(csv_path),
        "started_at": started_at,
        "ended_at": ended_at,
        "exit_code": int(proc.returncode),
        "parser_exit_code": 0,
    }
    


def _iter_grid(
    scenario: str,
    episodes: int,
    episode_length: int,
    seeds: Sequence[int],
    learning_rates: Sequence[float],
    lambda_starts: Sequence[float],
    lambda_decays: Sequence[float],
    batch_sizes: Sequence[int],
    num_epochs_list: Sequence[int],
    hidden_dims: Sequence[int],
    fully_obs: bool,
    include_no_llm: bool,
    device: str,
    llama_4bit: bool = False,
    lambda_modes: Sequence[str] = ("fixed",),
    kl_target: float = 0.01,
    lambda_min: float = 0.005,
    competence_window: int = 10,
    llm_mix_weight: float = 0.3,
    deterministic_torch: bool = False,
) -> Iterable[SweepConfig]:
    use_llm_modes = [True]
    if include_no_llm:
        use_llm_modes.append(False)

    for (
        seed,
        lr,
        ls,
        ld,
        bs,
        ne,
        hd,
        use_llm,
        lm,
    ) in product(
        seeds,
        learning_rates,
        lambda_starts,
        lambda_decays,
        batch_sizes,
        num_epochs_list,
        hidden_dims,
        use_llm_modes,
        lambda_modes,
    ):
        yield SweepConfig(
            scenario=scenario,
            episodes=episodes,
            episode_length=episode_length,
            seed=int(seed),
            learning_rate=float(lr),
            lambda_start=float(ls),
            lambda_decay=float(ld),
            batch_size=int(bs),
            num_epochs=int(ne),
            hidden_dim=int(hd),
            fully_obs=bool(fully_obs),
            use_llm=bool(use_llm),
            device=str(device),
            llama_4bit=bool(llama_4bit),
            lambda_mode=lm,
            kl_target=kl_target,
            lambda_min=lambda_min,
            competence_window=competence_window,
            llm_mix_weight=llm_mix_weight,
            deterministic_torch=deterministic_torch,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Run a grid sweep for LLM4Teach")

    p.add_argument("--scenario", type=str, default="tiny")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--episode-length", type=int, default=500)

    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--learning-rates", type=float, nargs="*", default=[1e-4])
    p.add_argument("--lambda-starts", type=float, nargs="*", default=[1.0])
    p.add_argument("--lambda-decays", type=float, nargs="*", default=[0.99])
    p.add_argument("--batch-sizes", type=int, nargs="*", default=[32])
    p.add_argument("--num-epochs", type=int, nargs="*", dest="num_epochs_list", default=[3])
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[64])

    p.add_argument("--fully-obs", action="store_true", default=True)
    p.add_argument("--partial-obs", action="store_false", dest="fully_obs")

    p.add_argument("--include-no-llm", action="store_true", help="Also run a no-LLM ablation")
    p.add_argument("--no-llama-4bit", action="store_false", dest="llama_4bit",
                   help="Disable 4-bit quantisation")
    p.set_defaults(llama_4bit=False)

    p.add_argument("--device", type=str, default="cpu")

    # Adaptive λ options
    p.add_argument("--lambda-modes", type=str, nargs="*",
                   default=["fixed"],
                   choices=["fixed", "target-kl", "competence", "adaptive"],
                   help="One or more lambda scheduling modes to sweep over.")
    p.add_argument("--kl-target", type=float, default=0.01)
    p.add_argument("--lambda-min", type=float, default=0.005)
    p.add_argument("--competence-window", type=int, default=10)
    p.add_argument("--llm-mix-weight", type=float, default=0.3,
                   help="Max LLM fraction in option sampling, decays with lambda (default: 0.3)")
    p.add_argument("--deterministic-torch", action="store_true", default=False,
                   help="Enable deterministic CUDA ops for reproducibility")

    p.add_argument("--out-dir", type=str, default="runs/llm4teach_sweep")
    p.add_argument("--dry-run", action="store_true", help="Only write command.txt + manifest entries")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"

    grid = list(
        _iter_grid(
            scenario=args.scenario,
            episodes=args.episodes,
            episode_length=args.episode_length,
            seeds=args.seeds,
            learning_rates=args.learning_rates,
            lambda_starts=args.lambda_starts,
            lambda_decays=args.lambda_decays,
            batch_sizes=args.batch_sizes,
            num_epochs_list=args.num_epochs_list,
            hidden_dims=args.hidden_dims,
            fully_obs=args.fully_obs,
            include_no_llm=args.include_no_llm,
            device=args.device,
            llama_4bit=args.llama_4bit,
            lambda_modes=args.lambda_modes,
            kl_target=args.kl_target,
            lambda_min=args.lambda_min,
            competence_window=args.competence_window,
            llm_mix_weight=args.llm_mix_weight,
            deterministic_torch=args.deterministic_torch,
        )
    )

    (out_dir / "sweep_summary.txt").write_text(
        (
            f"total_runs={len(grid)}\n"
            f"started_utc={datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}\n"
            f"dry_run={args.dry_run}\n"
        ),
        encoding="utf-8",
    )

    with manifest_path.open("a", encoding="utf-8") as mf:
        for idx, cfg in enumerate(grid, start=1):
            result = _run_one(cfg, out_dir=out_dir, dry_run=args.dry_run)
            mf.write(json.dumps(result, ensure_ascii=False) + "\n")
            mf.flush()
            print(f"[{idx}/{len(grid)}] {result['run_id']} exit={result['exit_code']}")

    # ── Auto-aggregate ────────────────────────────────────────────────────────
    if not args.dry_run:
        print("\nAggregating results...")
        from nasim.scripts.aggregate_llm4teach_sweep import aggregate
        summary_path = out_dir / "summary.csv"
        aggregate(out_dir, summary_path, last_k=20)


if __name__ == "__main__":
    main()
