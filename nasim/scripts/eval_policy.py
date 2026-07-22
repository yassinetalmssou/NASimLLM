"""Held-out evaluator for LLM4Teach student policies.

Evaluates a FROZEN student network (no LLM) on the native NASim reward metric.
CPU-only: does not import or load the LLM model.

Usage (single run):
  python -m nasim.scripts.eval_policy --run-dir runs/smoke --episodes 20

Usage (full tree):
  python -m nasim.scripts.eval_policy --root runs/ --episodes 200

Output written INTO each run directory:
  eval.csv                   - per-episode rows
  eval_summary.json          - scenario/condition/success_rate/CI/etc.

For random baseline:
  python -m nasim.scripts.eval_policy --root runs/ --policy random
  -> writes eval_random.csv + eval_summary_random.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import nasim
from nasim.agents.llm4teach_agent import LLM4TeachAgent

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)


def _wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion (handles n=0 gracefully)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def eval_run_dir(
    run_dir: Path,
    num_episodes: int = 200,
    eval_seed: int = 1000,
    device: str = "cpu",
    greedy: bool = True,
    episode_length_override: Optional[int] = None,
    policy: str = "checkpoint",
    summary_name: str = "eval_summary.json",
) -> dict:
    """Evaluate one run directory.  Returns the summary dict."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")

    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)

    scenario      = config["scenario"]
    fully_obs     = config.get("fully_obs", True)
    hidden_dim    = config.get("hidden_dim", 128)
    episode_length = episode_length_override or config.get("episode_length", 500)
    condition     = config.get("condition", "unknown")
    use_llm       = config.get("use_llm", False)
    train_seed    = config.get("seed", 0)

    # Decide output file names based on policy
    if policy == "random":
        csv_out  = run_dir / "eval_random.csv"
        json_out = run_dir / summary_name
    else:
        csv_out  = run_dir / "eval.csv"
        json_out = run_dir / summary_name

    # Build environment (eval seeds kept far from training seeds)
    template_env = nasim.make_benchmark(
        scenario,
        seed=eval_seed,
        fully_obs=fully_obs,
        flat_actions=True,
        flat_obs=True,
    )
    obs0, _ = template_env.reset()
    state_dim   = obs0.shape[0]
    action_list = template_env.action_space.actions
    num_actions = len(action_list)
    template_env.close() if hasattr(template_env, "close") else None

    # Build action-selection function
    if policy == "checkpoint":
        ckpt_path = run_dir / "ckpts" / "ckpt_final.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        agent = LLM4TeachAgent(
            state_dim=state_dim,
            action_list=action_list,
            hidden_dim=hidden_dim,
            num_options=5,
            device=device,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
        )
        try:
            agent.load_checkpoint(str(ckpt_path))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint/model dimension mismatch in {run_dir}.\n"
                f"  config: hidden_dim={hidden_dim}, state_dim={state_dim}\n"
                f"  PyTorch error: {exc}"
            ) from exc

        agent.student.eval()

        def choose_action(obs: np.ndarray) -> int:
            opt = agent.choose_option(obs, training=False)
            candidates = list(range(num_actions))
            return agent.select_action_from_option(obs, opt, candidates)

    else:
        _rng = np.random.default_rng(eval_seed)

        def choose_action(obs: np.ndarray) -> int:  # noqa: F811
            return int(_rng.integers(num_actions))

    # Evaluation loop
    records: list[dict] = []
    successes = 0
    steps_to_success: list[int] = []

    for ep in range(num_episodes):
        ep_seed = eval_seed + 10_000 + ep
        env = nasim.make_benchmark(
            scenario,
            seed=ep_seed,
            fully_obs=fully_obs,
            flat_actions=True,
            flat_obs=True,
        )
        obs, _ = env.reset()
        done       = False
        native_return = 0.0
        step       = 0

        while not done and step < episode_length:
            action_idx = choose_action(obs)
            obs, reward, done, truncated, _ = env.step(action_idx)
            native_return += float(reward)
            step += 1
            if truncated:
                break

        success = 1 if env.goal_reached() else 0
        successes += success
        if success:
            steps_to_success.append(step)

        records.append({
            "episode":          ep + 1,
            "success":          success,
            "native_return":    round(native_return, 4),
            "length":           step,
            "steps_to_success": step if success else "",
        })

    # Write per-episode CSV
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["episode", "success", "native_return", "length", "steps_to_success"],
        )
        writer.writeheader()
        writer.writerows(records)

    # Compute summary
    returns = [r["native_return"] for r in records]
    ci_lo, ci_hi = _wilson_ci(successes, num_episodes)
    median_s2s = float(np.median(steps_to_success)) if steps_to_success else None

    summary = {
        "scenario":               scenario,
        "condition":              condition,
        "use_llm":                use_llm,
        "train_seed":             train_seed,
        "eval_episodes":          num_episodes,
        "success_rate":           round(successes / num_episodes, 4),
        "success_ci_low":         round(ci_lo, 4),
        "success_ci_high":        round(ci_hi, 4),
        "mean_return":            round(float(np.mean(returns)), 4),
        "std_return":             round(float(np.std(returns)), 4),
        "median_steps_to_success": median_s2s,
        "eval_seed":              eval_seed,
        "greedy":                 greedy,
        "policy":                 policy,
    }

    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        f"[{run_dir.name}] success={summary['success_rate']:.2%} "
        f"({successes}/{num_episodes}) | mean_return={summary['mean_return']:.2f} "
        f"| {json_out.name}"
    )
    return summary


def find_run_dirs(root: Path):
    """Yield every directory under *root* that contains config.json."""
    for p in sorted(root.rglob("config.json")):
        yield p.parent


def main():
    parser = argparse.ArgumentParser(
        description="Held-out evaluator for LLM4Teach policies (no LLM required)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir", type=str,
        help="Single run directory to evaluate (must contain config.json)",
    )
    group.add_argument(
        "--root", type=str,
        help="Root tree - evaluate every directory that contains config.json",
    )
    parser.add_argument("--episodes", type=int, default=200,
                        help="Eval episodes per run (default: 200)")
    parser.add_argument("--eval-seed", type=int, default=1000,
                        help="Base RNG seed (default: 1000, offset far from training seeds)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for network inference (default: cpu)")
    parser.add_argument("--sample", action="store_true",
                        help="Stochastic option sampling; default is greedy")
    parser.add_argument(
        "--policy", choices=["checkpoint", "random"], default="checkpoint",
        help="checkpoint=trained network; random=lower-bound baseline (no ckpt needed)",
    )
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Override episode cap from config (default: use config value)")
    parser.add_argument("--summary-name", type=str, default="eval_summary.json",
                        help="Output summary filename (default: eval_summary.json)")

    args = parser.parse_args()
    greedy = not args.sample

    dirs = [Path(args.run_dir)] if args.run_dir else list(find_run_dirs(Path(args.root)))
    if not dirs:
        logger.error("No run directories found.")
        return
    logger.info(f"Evaluating {len(dirs)} run director{'y' if len(dirs)==1 else 'ies'}")

    ok = failed = 0
    for d in dirs:
        try:
            eval_run_dir(
                run_dir=d,
                num_episodes=args.episodes,
                eval_seed=args.eval_seed,
                device=args.device,
                greedy=greedy,
                episode_length_override=args.episode_length,
                policy=args.policy,
                summary_name=args.summary_name,
            )
            ok += 1
        except Exception as exc:
            logger.error(f"Failed [{d}]: {exc}")
            failed += 1

    logger.info(f"Done: {ok} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
