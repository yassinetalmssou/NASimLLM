"""LLM-guided NASim demo.

Usage: python -m nasim.demo_llm tiny [--epsilon 0.05] [--nl]
"""
import argparse
import os.path as osp
import sys

import nasim
 

from nasim.llm.LLM_API import LLMAPIClient
from nasim.llm.llm_advisor import LLMAdvisor
from nasim.agents.llm_guided_agent import LLMGuidedEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run an LLM-guided demo")
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Epsilon for epsilon-greedy exploration (0-1)")
    parser.add_argument("--nl", action="store_true",
                        help="Print a natural-language state summary each step (requires LLM or env description)")
    args = parser.parse_args()

    env = nasim.make_benchmark(
        args.env_name,
        fully_obs=True,
        flat_actions=True,
        flat_obs=True,
        render_mode="human",
    )

    client = None
    try:
        client = LLMAPIClient(model_id="meta-llama/Llama-3.1-8B-Instruct")
    except Exception as e:
        print("Warning: could not initialize LLMAPIClient:", e, file=sys.stderr)
        print("Falling back to dummy LLM client (uniform scores)")
        client = None

    advisor = LLMAdvisor(client=client)

    evaluator = LLMGuidedEvaluator(advisor=advisor, tau=1.0, alpha=0.3, top_k=5)

    agent = None
    DQN_POLICY_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "agents", "policies")
    DQN_POLICIES = {"tiny": osp.join(DQN_POLICY_DIR, "dqn_tiny.pt"), "small": osp.join(DQN_POLICY_DIR, "dqn_small.pt")}

    from nasim.agents.dqn_agent import DQNAgent
    print("Using DQNAgent.")
    dqn_agent = DQNAgent(env, verbose=False, **vars(args))
    ckpt = DQN_POLICIES.get(args.env_name)
    if ckpt and osp.exists(ckpt):
        try:
            dqn_agent.load(ckpt)
            print(f"Loaded DQN checkpoint: {ckpt}")
        except Exception as e:
            print("Warning: failed to load DQN checkpoint:", e, file=sys.stderr)
    else:
        print("No pre-trained DQN checkpoint found for this env; continuing with untrained DQN network.")

    class _DQNQFunc:
        def __init__(self, agent):
            self.agent = agent

        def forward(self, s):
            import torch
            o = torch.from_numpy(s).float().to(self.agent.device)
            if len(o.shape) == 1:
                o = o.view(1, -1)
            with torch.no_grad():
                q = self.agent.dqn(o)
            return q.cpu().numpy().squeeze()

    dqn_agent.qfunc = _DQNQFunc(dqn_agent)
    agent = dqn_agent


    # Header
    header_break = f"\n{'-'*60}"
    print(header_break)
    print(f"Running Demo on {args.env_name} environment")
    print("Using AI policy")
    print(header_break)

    # Delegate interactive stepping to evaluator
    evaluator.run_eval_episode(agent, env, eval_epsilon=args.epsilon, render=True, show_nl=args.nl)


if __name__ == "__main__":
    main()
