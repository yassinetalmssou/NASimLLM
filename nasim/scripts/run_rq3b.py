"""
RQ3b: Component ablation study (full, no_history, no_avoidlist, verbose_prompt, llm_cached).

Usage: python -m nasim.scripts.run_rq3b --scenario tiny --episodes 300 --seeds 0 1 2 \
       --out-dir runs/<tag>/rq3 --device cuda --llama-model llm_weights/Qwen3-4B

Output layout: <out-dir>/{scenario}/{condition}/seed{N}/
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


ABLATIONS = {
    "full": {
        "desc": "Full system (reference)",
        "flags": []
    },
    "no_history": {
        "desc": "No action history mechanism",
        "flags": ["--no-history"]
    },
    "no_avoidlist": {
        "desc": "No failed action tracking",
        "flags": ["--no-avoidlist"]
    },
    "verbose_prompt": {
        "desc": "Verbose prompts (not nomad-style)",
        "flags": ["--verbose-prompt"]
    },
    "llm_cached": {
        "desc": "Cached LLM calls",
        "flags": ["--llm-call-freq", "cached"]
    },
}


def build_command(
    ablation_name: str,
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
    lambda_mode: str = "competence",
    lambda_min: float = 0.05,
    llm_mix_weight: float = 0.5,
    hidden_dim: int = 128,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    num_epochs: int = 4,
) -> List[str]:
    """Build train_llm4teach command for specific ablation."""

    ablation = ABLATIONS[ablation_name]
    save_dir = out_dir / scenario / ablation_name / f"seed{seed}" / "ckpts"

    cmd = [
        sys.executable,
        "-m",
        "nasim.train_llm4teach",
        "--scenario", scenario,
        "--episodes", str(episodes),
        "--episode-length", str(episode_length),
        "--seed", str(seed),
        "--device", device,
        "--save-dir", str(save_dir),
        "--llama-model", llama_model,
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
        "--no-step-logs",
    ]

    cmd.extend(ablation["flags"])
    cmd.extend(["--condition", ablation_name])

    return cmd


def run_experiment(
    ablation_name: str,
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
    lambda_mode: str = "competence",
    lambda_min: float = 0.05,
    llm_mix_weight: float = 0.5,
    hidden_dim: int = 128,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    num_epochs: int = 4,
    dry_run: bool = False,
) -> bool:
    """Run single ablation experiment."""

    cmd = build_command(
        ablation_name, scenario, episodes, seed, out_dir,
        device, llama_model, episode_length, teacher_temp,
        lambda_start, lambda_decay,
        lambda_mode=lambda_mode, lambda_min=lambda_min,
        llm_mix_weight=llm_mix_weight, hidden_dim=hidden_dim,
        learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs,
    )

    if dry_run:
        return True
    
    run_dir = out_dir / scenario / ablation_name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    cmd_file = run_dir / "command.txt"
    with open(cmd_file, "w") as f:
        f.write(" ".join(cmd) + "\n")
    
    log_file = run_dir / "train.log"
    try:
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
        logger.info(f"Completed: {ablation_name}/seed{seed}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {ablation_name}/seed{seed} (exit code {e.returncode})")
        return False
    except Exception as e:
        logger.error(f"Error: {ablation_name}/seed{seed}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="RQ3b: Component ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--scenario", type=str, default="small",
                        help="NASim scenario (default: small)")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Training episodes per run (default: 300)")
    parser.add_argument("--episode-length", type=int, default=500,
                        help="Max steps per episode (default: 500)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Random seeds (default: 0 1 2)")
    
    parser.add_argument("--ablations", type=str, nargs="+",
                        choices=list(ABLATIONS.keys()) + ["all"],
                        default=["all"],
                        help="Which ablations to run (default: all)")
    
    parser.add_argument("--out-dir", type=str, default="runs/rq3",
                        help="Output directory (default: runs/rq3)")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cpu or cuda (default: cuda)")
    parser.add_argument("--llama-model", type=str, 
                        default="llm_weights/Qwen3-4B",
                        help="LLM model path (default: llm_weights/Qwen3-4B)")
    
    parser.add_argument("--teacher-temp", type=float, default=2.0,
                        help="Teacher temperature (default: 2.0)")
    parser.add_argument("--lambda-start", type=float, default=1.0,
                        help="Initial lambda (default: 1.0)")
    parser.add_argument("--lambda-decay", type=float, default=0.97,
                        help="Lambda decay rate (default: 0.97)")
    parser.add_argument("--lambda-mode", type=str, default="competence",
                        choices=["fixed", "target-kl", "competence", "adaptive"],
                        help="Lambda scheduling mode (default: competence)")
    parser.add_argument("--lambda-min", type=float, default=0.05,
                        help="Minimum lambda for competence mode (default: 0.05)")
    parser.add_argument("--llm-mix-weight", type=float, default=0.5,
                        help="LLM mixing weight (default: 0.5)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension (default: 128)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="PPO batch size (default: 64)")
    parser.add_argument("--num-epochs", type=int, default=4,
                        help="PPO update epochs (default: 4)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--parallel", action="store_true",
                        help="Generate commands_{scenario}.txt for parallel execution (HPC mode)")
    
    args = parser.parse_args()
    
    if "all" in args.ablations:
        ablations = list(ABLATIONS.keys())
    else:
        ablations = args.ablations
    
    out_dir = Path(args.out_dir)
    
    logger.info(f"RQ3b: Scenario={args.scenario} | Ablations={', '.join(ablations)} | Seeds={args.seeds}")
    
    if args.parallel:
        cmd_file = out_dir / f"commands_{args.scenario}.txt"
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(cmd_file, "w") as f:
            for ablation in ablations:
                for seed in args.seeds:
                    cmd = build_command(
                        ablation, args.scenario, args.episodes, seed,
                        out_dir, args.device, args.llama_model,
                        args.episode_length, args.teacher_temp,
                        args.lambda_start, args.lambda_decay,
                        lambda_mode=args.lambda_mode,
                        lambda_min=args.lambda_min,
                        llm_mix_weight=args.llm_mix_weight,
                        hidden_dim=args.hidden_dim,
                        learning_rate=args.learning_rate,
                        batch_size=args.batch_size,
                        num_epochs=args.num_epochs,
                    )
                    f.write(" ".join(cmd) + "\n")

        logger.info(f"Commands saved to: {cmd_file} ({len(ablations) * len(args.seeds)} jobs)")
        return
    
    total = len(ablations) * len(args.seeds)
    completed = 0
    failed = 0
    
    for i, (ablation, seed) in enumerate(
        [(a, s) for a in ablations for s in args.seeds], start=1
    ):
        logger.info(f"[{i}/{total}] {ablation}/seed{seed}")
        
        success = run_experiment(
            ablation, args.scenario, args.episodes, seed,
            out_dir, args.device, args.llama_model,
            args.episode_length, args.teacher_temp,
            args.lambda_start, args.lambda_decay,
            lambda_mode=args.lambda_mode,
            lambda_min=args.lambda_min,
            llm_mix_weight=args.llm_mix_weight,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            dry_run=args.dry_run,
        )
        
        if success:
            completed += 1
        else:
            failed += 1
    
    logger.info(f"SUMMARY: {completed}/{total} completed, {failed} failed")
    
    if not args.dry_run:
        logger.info(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
