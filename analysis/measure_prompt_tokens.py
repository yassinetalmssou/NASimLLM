"""Measure the prompt token count of the compact vs verbose state summary
(promotor feedback p37), compared across the teacher tokenizers and scenarios.

Builds the trainer with use_llm=False (no model loaded), rolls out a random policy
to collect a representative spread of states, then tokenizes both the compact and
the verbose form with each teacher tokenizer. No LLM weights are loaded.
"""
import os
import sys

import numpy as np

BASE = r"c:/Users/yassi/Documents/VUB/Master/2MA/Thesis/NASimLLM"
os.chdir(BASE)
sys.path.insert(0, BASE)

from transformers import AutoTokenizer  # noqa: E402
from nasim.train_llm4teach import LLM4TeachTrainer  # noqa: E402

N_EP = 25
MAX_STEPS = 500
SCENARIOS = ["tiny", "small", "small-linear", "medium"]
# (label, candidate paths tried in order); Qwen3-4B and 8B share a tokenizer,
# as do all Llama-3.x models, so three tokenizers cover the five teachers.
TOKENIZERS = [
    ("Qwen3 (4B, 8B)", ["llm_weights/Qwen3-4B", "Qwen/Qwen3-4B"]),
    ("Llama-3.x (1B, 3B, 8B)", ["llm_weights/llama-3.2-3B-Instruct",
                                "llm_weights/llama-3.2-1B-Instruct"]),
]


def load_tokenizer(paths):
    for p in paths:
        try:
            return AutoTokenizer.from_pretrained(p, local_files_only=True)
        except Exception:
            try:
                return AutoTokenizer.from_pretrained(p)
            except Exception:
                continue
    return None


def collect_strings(scenario):
    tr = LLM4TeachTrainer(scenario=scenario, use_llm=False, verbose=False,
                          step_log_dir=None, save_dir=None, csv_log=None,
                          num_episodes=1)
    env = tr.env
    n_actions = len(tr.action_list)
    rng = np.random.default_rng(0)
    comp, verb = [], []
    for _ in range(N_EP):
        obs, _ = env.reset()
        for _ in range(MAX_STEPS):
            try:
                state_summary, host, aux = tr.summarize_state(obs)
                compact = tr.compact_state_summary(host, aux)
            except Exception:
                break
            comp.append(compact)
            verb.append(state_summary)
            step = env.step(int(rng.integers(n_actions)))
            obs, done, trunc = step[0], step[2], step[3]
            if done or trunc:
                break
    return comp, verb


def n_tokens(tok, strings):
    return np.array([len(tok(s, add_special_tokens=False)["input_ids"]) for s in strings])


if __name__ == "__main__":
    toks = [(lbl, load_tokenizer(paths)) for lbl, paths in TOKENIZERS]
    toks = [(lbl, t) for lbl, t in toks if t is not None]
    print("loaded tokenizers:", [lbl for lbl, _ in toks])

    table = []
    for scen in SCENARIOS:
        comp, verb = collect_strings(scen)
        print(f"\n===================  {scen}  (n={len(comp)} states)  ===================")
        print(f"  {'tokenizer':22s} {'compact':>9s} {'verbose':>9s} {'saved':>8s} {'ratio':>7s}")
        for lbl, tok in toks:
            nc, nv = n_tokens(tok, comp).mean(), n_tokens(tok, verb).mean()
            saved = (1 - nc / nv) * 100
            print(f"  {lbl:22s} {nc:9.0f} {nv:9.0f} {saved:7.0f}% {nv / nc:6.2f}x")
            table.append((scen, lbl, round(nc), round(nv), round(saved), round(nv / nc, 2)))

    print("\n=== CSV ===")
    print("scenario,tokenizer,compact,verbose,saved_pct,ratio")
    for r in table:
        print(",".join(str(x) for x in r))
