"""
RQ1: LLM guidance vs baselines.

Conditions: llm_full, ppo_options, random, bruteforce.
Usage: python -m nasim.scripts.run_rq1 --scenario tiny --episodes 300 --seeds 0 1 2 \
       --out-dir runs/<tag>/rq1 --device cuda --llama-model llm_weights/Qwen3-4B

Output layout: <out-dir>/{scenario}/{condition}/seed{N}/
"""

import argparse
import csv
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import nasim

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

_CSV_FIELDS = [
    'episode', 'reward', 'avg_reward_10', 'length',
    'lambda_kl', 'next_lambda',
    'hosts_compromised', 'hosts_total', 'compromise_rate', 'discovery_pct',
    'loss_actor', 'loss_critic', 'loss_kl_norm', 'clip_pct', 'teacher_agree', 'success',
    't_rollout_s', 't_ppo_s', 't_llm_s', 't_episode_s', 'cache_hits', 'cache_misses',
]


def _make_summarizer(env):
    """Lightweight proxy that provides summarize_state() without loading LLM."""
    from nasim.train_llm4teach import LLM4TeachTrainer
    s = LLM4TeachTrainer.__new__(LLM4TeachTrainer)
    s.env = env
    s.fully_obs = True
    return s


def _host_metrics(summarizer, obs):
    """Return (hosts_compromised, hosts_discovered, hosts_total) from obs."""
    try:
        _, host_obs, _ = summarizer.summarize_state(obs)
        if not host_obs:
            return 0, 0, 1
        total      = len(host_obs)
        compromised = sum(1 for h in host_obs if h.get('Compromised', False))
        discovered  = sum(1 for h in host_obs if h.get('Discovered',  False))
        return compromised, discovered, total
    except Exception:
        return 0, 0, 1



def run_random_episodes(scenario: str, seed: int, num_episodes: int,
                        episode_length: int, out_csv: Path):
    env = nasim.make_benchmark(
        scenario, seed=seed, fully_obs=True, flat_actions=True, flat_obs=True
    )
    summarizer = _make_summarizer(env)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    recent_rewards: deque = deque(maxlen=10)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()

        for ep in range(num_episodes):
            t0 = time.perf_counter()
            obs, _ = env.reset()
            total_r, done, steps = 0.0, False, 0
            while not done and steps < episode_length:
                action = env.action_space.sample()
                obs, r, done, truncated, _ = env.step(action)
                total_r += r
                steps += 1
                if truncated:
                    break
            t_ep = time.perf_counter() - t0

            recent_rewards.append(total_r)
            comp, disc, total_h = _host_metrics(summarizer, obs)
            success = 1 if env.goal_reached() else 0

            writer.writerow({
                'episode':           ep + 1,
                'reward':            round(total_r, 3),
                'avg_reward_10':     round(float(np.mean(recent_rewards)), 3),
                'length':            steps,
                'lambda_kl':         '',
                'next_lambda':       '',
                'hosts_compromised': comp,
                'hosts_total':       total_h,
                'compromise_rate':   round(comp / max(total_h, 1), 4),
                'discovery_pct':     round(disc / max(total_h, 1) * 100, 1),
                'loss_actor':        '',
                'loss_critic':       '',
                'loss_kl_norm':      '',
                'clip_pct':          '',
                'teacher_agree':     '',
                'success':           success,
                't_rollout_s':       round(t_ep, 2),
                't_ppo_s':           '',
                't_llm_s':           '',
                't_episode_s':       round(t_ep, 2),
                'cache_hits':        '',
                'cache_misses':      '',
            })
            if (ep + 1) % 50 == 0:
                logger.info(f"  [random seed={seed}] ep {ep+1}/{num_episodes} "
                            f"reward={total_r:.2f} comp={comp}/{total_h}")

    logger.info(f"Random results: {out_csv}")


def run_bruteforce_episodes(scenario: str, seed: int, num_episodes: int,
                            episode_length: int, out_csv: Path):
    env = nasim.make_benchmark(
        scenario, seed=seed, fully_obs=True, flat_actions=True, flat_obs=True
    )
    summarizer  = _make_summarizer(env)
    num_actions = len(env.action_space.actions)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    recent_rewards: deque = deque(maxlen=10)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()

        for ep in range(num_episodes):
            t0 = time.perf_counter()
            obs, _ = env.reset()
            total_r, done, steps, action_idx = 0.0, False, 0, 0
            while not done and steps < episode_length:
                obs, r, done, truncated, _ = env.step(action_idx % num_actions)
                total_r += r
                steps += 1
                action_idx += 1
                if truncated:
                    break
            t_ep = time.perf_counter() - t0

            recent_rewards.append(total_r)
            comp, disc, total_h = _host_metrics(summarizer, obs)
            success = 1 if env.goal_reached() else 0

            writer.writerow({
                'episode':           ep + 1,
                'reward':            round(total_r, 3),
                'avg_reward_10':     round(float(np.mean(recent_rewards)), 3),
                'length':            steps,
                'lambda_kl':         '',
                'next_lambda':       '',
                'hosts_compromised': comp,
                'hosts_total':       total_h,
                'compromise_rate':   round(comp / max(total_h, 1), 4),
                'discovery_pct':     round(disc / max(total_h, 1) * 100, 1),
                'loss_actor':        '',
                'loss_critic':       '',
                'loss_kl_norm':      '',
                'clip_pct':          '',
                'teacher_agree':     '',
                'success':           success,
                't_rollout_s':       round(t_ep, 2),
                't_ppo_s':           '',
                't_llm_s':           '',
                't_episode_s':       round(t_ep, 2),
                'cache_hits':        '',
                'cache_misses':      '',
            })
            if (ep + 1) % 50 == 0:
                logger.info(f"  [bruteforce seed={seed}] ep {ep+1}/{num_episodes} "
                            f"reward={total_r:.2f} comp={comp}/{total_h}")

    logger.info(f"Bruteforce results: {out_csv}")


def run_trainer_condition(condition: str, scenario: str, seed: int,
                          num_episodes: int, episode_length: int,
                          out_dir: Path, device: str, llama_model: str,
                          extra_kwargs: dict):
    """Train with LLM4TeachTrainer and save CSV + checkpoint."""
    from nasim.train_llm4teach import LLM4TeachTrainer

    save_dir = out_dir / scenario / condition / f"seed{seed}" / "ckpts"
    csv_path = out_dir / scenario / condition / f"seed{seed}" / "train.csv"
    step_log_dir = out_dir / scenario / condition / f"seed{seed}" / "step_logs"
    save_dir.mkdir(parents=True, exist_ok=True)

    use_llm = (condition == "llm_full")

    trainer = LLM4TeachTrainer(
        scenario=scenario,
        seed=seed,
        num_episodes=num_episodes,
        episode_length=episode_length,
        device=device,
        save_dir=str(save_dir),
        step_log_dir=None,   # disabled: per-step logs were a multi-GB disk hog on SCRATCH
        use_llm=use_llm,
        llama_model=llama_model,
        use_local_llama=use_llm,
        llama_use_4bit=True,
        fully_obs=True,
        csv_log=str(csv_path),
        condition=condition,
        **extra_kwargs,
    )
    trainer.train()
    logger.info(f"[{condition} seed={seed}] Done. CSV at {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RQ1: Compare NASimLLM vs PPO-options vs Random vs Bruteforce"
    )
    parser.add_argument("--scenario", default="small",
                        help="NASim benchmark scenario (default: small)")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Training episodes per condition/seed (default: 300)")
    parser.add_argument("--episode-length", type=int, default=500,
                        help="Max steps per episode (default: 500)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Random seeds to run (default: 0 1 2)")
    parser.add_argument("--out-dir", default="runs/rq1",
                        help="Root output directory (default: runs/rq1)")
    parser.add_argument("--device", default="cuda",
                        help="Device for RL agent (default: cuda)")
    parser.add_argument("--llama-model", default="llm_weights/Qwen3-4B",
                        help="Path to LLM model (default: llm_weights/Qwen3-4B)")
    parser.add_argument("--conditions", nargs="+",
                        default=["llm_full", "ppo_options", "random", "bruteforce"],
                        help="Which conditions to run (default: all four)")
    parser.add_argument("--lambda-decay", type=float, default=0.97)
    parser.add_argument("--lambda-mode", default="competence",
                        choices=["fixed", "target-kl", "competence", "adaptive"])
    parser.add_argument("--lambda-min", type=float, default=0.05,
                        help="Minimum lambda for competence mode (default: 0.05)")
    parser.add_argument("--llm-mix-weight", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--parallel", action="store_true",
                        help="Generate commands_{scenario}.txt for SLURM array jobs instead of running")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    extra_kwargs = dict(
        lambda_start=1.0,
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
        out_dir.mkdir(parents=True, exist_ok=True)
        # Use scenario-specific filename so parallel calls from launch.sh don't overwrite each other
        cmd_file = out_dir / f"commands_{args.scenario}.txt"
        with open(cmd_file, "w") as f:
            for seed in args.seeds:
                for condition in args.conditions:
                    cmd = [
                        sys.executable, "-m", "nasim.scripts.run_rq1",
                        "--conditions", condition,
                        "--seeds", str(seed),
                        "--scenario", args.scenario,
                        "--episodes", str(args.episodes),
                        "--episode-length", str(args.episode_length),
                        "--out-dir", str(out_dir),
                        "--device", args.device,
                        "--llama-model", args.llama_model,
                        "--lambda-decay", str(args.lambda_decay),
                        "--lambda-mode", args.lambda_mode,
                        "--lambda-min", str(args.lambda_min),
                        "--llm-mix-weight", str(args.llm_mix_weight),
                        "--hidden-dim", str(args.hidden_dim),
                        "--learning-rate", str(args.learning_rate),
                        "--batch-size", str(args.batch_size),
                        "--num-epochs", str(args.num_epochs),
                    ]
                    f.write(" ".join(cmd) + "\n")
        total = len(args.seeds) * len(args.conditions)
        logger.info(f"Command file generated: {cmd_file} ({total} jobs)")
        return

    for seed in args.seeds:
        for condition in args.conditions:
            logger.info(f"CONDITION={condition}  SEED={seed}  SCENARIO={args.scenario}")

            if condition == "random":
                run_random_episodes(
                    scenario=args.scenario, seed=seed,
                    num_episodes=args.episodes,
                    episode_length=args.episode_length,
                    out_csv=out_dir / args.scenario / "random" / f"seed{seed}" / "train.csv",
                )
            elif condition == "bruteforce":
                run_bruteforce_episodes(
                    scenario=args.scenario, seed=seed,
                    num_episodes=args.episodes,
                    episode_length=args.episode_length,
                    out_csv=out_dir / args.scenario / "bruteforce" / f"seed{seed}" / "train.csv",
                )
            else:
                run_trainer_condition(
                    condition=condition,
                    scenario=args.scenario,
                    seed=seed,
                    num_episodes=args.episodes,
                    episode_length=args.episode_length,
                    out_dir=out_dir,
                    device=args.device,
                    llama_model=args.llama_model,
                    extra_kwargs=extra_kwargs,
                )

    logger.info("\nAll RQ1 conditions finished.")
    logger.info(f"Results in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
