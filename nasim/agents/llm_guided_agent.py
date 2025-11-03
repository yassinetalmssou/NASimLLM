from typing import List, Tuple, Dict
from collections import deque
import numpy as np
import math
import random

from nasim.llm.llm_advisor import LLMAdvisor


class LLMGuidedEvaluator:
    def __init__(self, advisor: LLMAdvisor = None, tau: float = 1.0, alpha: float = 0.3, top_k: int = 5,
                 history_len: int = 8, history_penalty: float = 0.2):
        self.advisor = advisor or LLMAdvisor()
        self.tau = tau if tau != 0 else 1.0
        self.alpha = alpha
        self.top_k = top_k
        self.history_len = max(0, int(history_len))
        self.history_penalty = float(history_penalty)

    def run_eval_episode(self, agent, env, eval_epsilon: float = 0.05, render: bool = False, show_nl: bool = False):
        s, _ = env.reset()
        done = False
        env_step_limit_reached = False

        steps = 0
        episode_return = 0
        history = deque(maxlen=self.history_len)

        line_break = "="*60
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render()
            input("Initial state. Press enter to continue..")

        while not done and not env_step_limit_reached:
            q_values = self._get_q_values(agent, s)

            action_indices = np.argsort(q_values)[::-1]
            k = min(self.top_k, len(action_indices))
            topk_idx = action_indices[:k]

            candidates = []
            for idx in topk_idx:
                try:
                    desc = env.action_space.get_action(int(idx))
                except Exception:
                    desc = f"action_{idx}"
                candidates.append((str(int(idx)), str(desc)))

            # Summary with recent actions
            base_summary = self._summarize_state(env, s)
            if history:
                hist_items = []
                for h in list(history)[::-1]:
                    h_desc = h.get("desc", f"action_{h['a']}")
                    hist_items.append(f"{h['a']}: {h_desc} (r={h['r']})")
                hist_text = ", ".join(hist_items)
                summary = base_summary + "\nRecent actions: " + hist_text
            else:
                summary = base_summary

            llm_scores = self.advisor.score_actions(summary, candidates)

            final_scores = np.array(q_values, dtype=float) / float(self.tau)

            for cid, _ in candidates:
                idx = int(cid)
                p = llm_scores.get(cid, 1e-6)
                final_scores[idx] = final_scores[idx] + self.alpha * math.log(p + 1e-12)

            # Penalize repeating same action in same state using recent history
            try:
                state_key = base_summary
                if self.history_penalty > 0.0 and history:
                    counts = {}
                    for h in history:
                        if h.get("state") == state_key:
                            a = int(h.get("a", -1))
                            counts[a] = counts.get(a, 0) + 1
                    for a, c in counts.items():
                        if 0 <= a < len(final_scores):
                            final_scores[a] -= self.history_penalty * float(c)
            except Exception:
                pass

            # epsilon-greedy over top-k
            epsilon_triggered = False
            if random.random() < float(eval_epsilon):
                epsilon_triggered = True
                try:
                    chosen = int(random.choice(topk_idx))
                except Exception:
                    chosen = int(final_scores.argmax())
            else:
                chosen = int(final_scores.argmax())

            next_s, r, done, env_step_limit_reached, _ = env.step(chosen)
            # record to history
            try:
                desc_chosen = None
                try:
                    desc_chosen = env.action_space.get_action(int(chosen))
                except Exception:
                    desc_chosen = f"action_{int(chosen)}"
                history.append({"state": base_summary, "a": int(chosen), "r": float(r), "desc": str(desc_chosen)})
            except Exception:
                pass
            s = next_s
            episode_return += r
            steps += 1

            if render:
                print("\n" + line_break)
                print(f"Step {steps}")
                print(line_break)
                print(f"Action Performed = {env.action_space.get_action(chosen)}")
                if show_nl:
                    try:
                        nl = self.advisor.summarize_state_nl(env, s)
                        print("\n" + nl)
                    except Exception:
                        pass
                # Reasoning line: top-k and LLM scores or epsilon exploration
                try:
                    if epsilon_triggered:
                        tk = [int(i) for i in topk_idx]
                        print(f"Reasoning: epsilon={float(eval_epsilon):.2f} triggered; picked random from top-k={tk} -> chosen={int(chosen)}")
                    else:
                        tk = [int(i) for i in topk_idx]
                        ls = [float(llm_scores.get(str(int(i)), llm_scores.get(int(i), 0.0))) for i in topk_idx]
                        ls_disp = [round(x, 3) for x in ls]
                        print(f"Reasoning: top-k={tk} LLM={ls_disp} -> greedy={int(chosen)}")
                except Exception:
                    pass
                env.render()
                print(f"Reward = {r}")
                print(f"Done = {done}")
                print(f"Step limit reached = {env_step_limit_reached}")
                input("Press enter to continue..")

                if done or env_step_limit_reached:
                    print("\n" + line_break)
                    print("EPISODE FINISHED")
                    print(line_break)
                    print(f"Goal reached = {env.goal_reached()}")
                    print(f"Total steps = {steps}")
                    print(f"Total reward = {episode_return}")

        return episode_return, steps, env.goal_reached()

    def _get_q_values(self, agent, state):
        try:
            q_vals = agent.qfunc.forward(state)
            return np.asarray(q_vals, dtype=float)
        except Exception:
            try:
                q_vals = agent.qfunc(state)
                return np.asarray(q_vals, dtype=float)
            except Exception:
                try:
                    n = agent.env.action_space.n
                except Exception:
                    n = 1
                return np.ones(n, dtype=float)

    def _summarize_state(self, env, state) -> str:
        try:
            desc = env.get_state_description() if hasattr(env, "get_state_description") else None
        except Exception:
            desc = None
        if desc:
            return str(desc)
        return f"obs={str(state)[:200]}"


__all__ = ["LLMGuidedEvaluator"]

if __name__ == "__main__":
    try:
        from nasim.agents.ql_agent import TabularQLearningAgent
        import nasim

        env = nasim.make_benchmark("tiny", seed=0, fully_obs=True, flat_actions=True, flat_obs=True)
        agent = TabularQLearningAgent(env, verbose=False)
        evaluator = LLMGuidedEvaluator()
        ret, steps, goal = evaluator.run_eval_episode(agent, env)
        print(f"ret={ret} steps={steps} goal={goal}")
    except Exception:
        print("Demo unavailable.")
