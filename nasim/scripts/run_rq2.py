"""
RQ2: Scenario generalization across network sizes (tiny, small, small-linear, medium).

Runs BOTH conditions (llm_full, ppo_options) across all four scenarios.

Output layout: <out-dir>/{scenario}/{condition}/seed{N}/

Usage: python -m nasim.scripts.run_rq2 --episodes 300 --seeds 0 1 2
       --out-dir runs/rq2 --device cuda --llama-model llm_weights/Qwen3-4B
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


SCENARIOS = ["tiny", "small", "small-linear", "medium"]
CONDITIONS = ["llm_full", "ppo_options"]


def build_command(
    condition: str,
    scenario: str,
    episodes: int,
    seed: int,
    out_dir: Path,
    device: str,
    llama_model: str,
    episode_length: int,
    teacher_temp: float,
    lambda_start: float,
    lambda_decay: float,
    lambda_mode: str,
    lambda_min: float,
    llm_mix_weight: float,
    hidden_dim: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
) -> List[str]:
    """Build train_llm4teach command for specific condition x scenario."""

    save_dir = out_dir / scenario / condition / f"seed{seed}" / "ckpts"

    cmd = [
        sys.executable,
        "-m",
        "nasim.train_llm4teach",
        "--scenario", scenario,
        "--episodes", str(episodes),
        "--episode-length", str(episode_length),
        "--seed", str(seed),
        "--device", device,
        "--teacher-temp", str(teacher_temp),
        "--lambda-start", str(lambda_start),
        "--lambda-decay", str(lambda_decay),
        "--lambda-mode", lambda_mode,
        "--lambda-min", str(lambda_min),
        "--llm-mix-weight", str(llm_mix_weight),
        "--hidden-dim", str(hidden_dim),
        "--learning-rate", str(learning_rate),
        "--batch-size", str(batch_size),
        "--num-epochs", str(num_epochs),
        "--save-dir", str(save_dir),
        "--condition", condition,
        "--no-step-logs",
    ]

    if condition == "llm_full":
        cmd.extend(["--llama-model", llama_model])
    else:
        cmd.append("--no-llm")

    return cmd


def run_experiment(
    condition: str,
    scenario: str,
    episodes: int,
    seed: int,
    out_dir: Path,
    device: str,
    llama_model: str,
    episode_length: int,
    teacher_temp: float,
    lambda_start: float,
    lambda_decay: float,
    lambda_mode: str,
    lambda_min: float,
    llm_mix_weight: float,
    hidden_dim: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    dry_run: bool = False,
) -> int:
    """Run single experiment (one condition + one scenario + one seed)."""

    cmd = build_command(
        condition, scenario, episodes, seed, out_dir, device, llama_model,
        episode_length, teacher_temp, lambda_start, lambda_decay,
        lambda_mode, lambda_min, llm_mix_weight, hidden_dim, learning_rate,
        batch_size, num_epochs,
    )

    cmd_str = " ".join(cmd)

    if dry_run:
        print(cmd_str)
        return 0

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error(f"Failed: {condition}/{scenario}/seed{seed} (exit code {result.returncode})")
    else:
        logger.info(f"Completed: {condition}/{scenario}/seed{seed}")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="RQ2: Scenario Generalization Experiments (llm_full + ppo_options)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--episodes", type=int, default=300,
        help="Number of training episodes per run (default: 300)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Random seeds to use (default: 0 1 2)"
    )
    parser.add_argument(
        "--conditions", nargs="+", default=CONDITIONS,
        help=f"Conditions to run (default: {CONDITIONS})"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("runs/rq2"),
        help="Root output directory (default: runs/rq2)"
    )

    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use: cuda or cpu (default: cuda)"
    )
    parser.add_argument(
        "--llama-model", type=str, default="llm_weights/Qwen3-4B",
        help="Path to LLM weights (default: llm_weights/Qwen3-4B)"
    )

    parser.add_argument(
        "--episode-length", type=int, default=500,
        help="Max steps per episode (default: 500)"
    )
    parser.add_argument(
        "--teacher-temp", type=float, default=2.0,
        help="Teacher temperature for softening (default: 2.0)"
    )
    parser.add_argument(
        "--lambda-start", type=float, default=1.0,
        help="Initial distillation coefficient (default: 1.0)"
    )
    parser.add_argument(
        "--lambda-decay", type=float, default=0.97,
        help="Lambda decay factor per episode (default: 0.97)"
    )
    parser.add_argument(
        "--lambda-mode", type=str, default="competence",
        choices=["fixed", "target-kl", "competence", "adaptive"],
        help="Lambda scheduling mode (default: competence)"
    )
    parser.add_argument(
        "--lambda-min", type=float, default=0.05,
        help="Minimum lambda for competence mode (default: 0.05)"
    )
    parser.add_argument(
        "--llm-mix-weight", type=float, default=0.5,
        help="LLM mixing weight (default: 0.5)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Hidden dimension for student network (default: 128)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="PPO batch size (default: 64)"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=4,
        help="PPO update epochs (default: 4)"
    )

    parser.add_argument(
        "--parallel", action="store_true",
        help="Generate command file for SLURM array jobs instead of running"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )

    args = parser.parse_args()

    experiments = []
    for condition in args.conditions:
        for scenario in SCENARIOS:
            for seed in args.seeds:
                experiments.append((condition, scenario, seed))

    total = len(experiments)
    logger.info(
        f"Total experiments: {total} "
        f"({len(args.conditions)} conditions x {len(SCENARIOS)} scenarios x {len(args.seeds)} seeds)"
    )

    # Helper to forward all shared hyperparams
    def _kwargs():
        return dict(
            episodes=args.episodes,
            out_dir=args.out_dir,
            device=args.device,
            llama_model=args.llama_model,
            episode_length=args.episode_length,
            teacher_temp=args.teacher_temp,
            lambda_start=args.lambda_start,
            lambda_decay=args.lambda_decay,
            lambda_mode=args.lambda_mode,
            lambda_min=args.lambda_min,
            llm_mix_weight=args.llm_mix_weight,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
        )

    if args.parallel:
        cmd_file = args.out_dir / "commands.txt"
        args.out_dir.mkdir(parents=True, exist_ok=True)

        with open(cmd_file, "w") as f:
            for condition, scenario, seed in experiments:
                cmd = build_command(condition=condition, scenario=scenario, seed=seed, **_kwargs())
                f.write(" ".join(cmd) + "\n")

        logger.info(f"Command file generated: {cmd_file} ({total} jobs)")
        return 0

    failed = 0
    for i, (condition, scenario, seed) in enumerate(experiments, 1):
        logger.info(f"[{i}/{total}] {condition}/{scenario}/seed{seed}")
        ret = run_experiment(
            condition=condition, scenario=scenario, seed=seed,
            dry_run=args.dry_run, **_kwargs(),
        )
        if ret != 0:
            failed += 1

    logger.info(f"Completed: {total - failed}/{total} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
