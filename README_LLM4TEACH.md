# LLM4Teach — LLM-guided PPO for Network Penetration Testing

An extension of [NASim](https://networkattacksimulator.readthedocs.io/) that trains a PPO agent using a local LLM (Llama 3.2) as a teacher signal. The student agent learns to compromise networks via option-based action selection; the LLM guides exploration by scoring high-level options (scan, exploit, privilege escalation, lateral movement, pivot). Teacher influence is annealed over time so the student eventually acts independently.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU inference with 4-bit quantization (recommended):
```bash
pip install bitsandbytes>=0.41 accelerate
```

### 2. Download LLM weights

Weights are **not** included in the repository. Download them from Hugging Face:

```bash
# 1B model (fast, for testing)
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
    --local-dir llm_weights/llama-3.2-1B-Instruct

# 3B model (better quality, for thesis runs)
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
    --local-dir llm_weights/llama-3.2-3B-Instruct
```

You need a Hugging Face account and access to the Llama 3.2 gated models.

---

## Training

### LLM-guided run (competence-based λ annealing)

```bash
python nasim/train_llm4teach.py \
  --scenario small \
  --seed 42 \
  --episodes 100 \
  --episode-length 500 \
  --hidden-dim 128 \
  --learning-rate 3e-4 \
  --batch-size 64 \
  --num-epochs 4 \
  --lambda-mode competence \
  --lambda-start 1.0 \
  --lambda-decay 0.97 \
  --lambda-min 0.05 \
  --llm-mix-weight 0.5 \
  --llama-model llm_weights/llama-3.2-3B-Instruct \
  --device cuda \
  --save-dir runs/my_run/ckpts \
  --no-step-logs
```

### Baseline (no LLM)

```bash
python nasim/train_llm4teach.py \
  --scenario small \
  --seed 42 \
  --episodes 100 \
  --episode-length 500 \
  --hidden-dim 128 \
  --learning-rate 3e-4 \
  --batch-size 64 \
  --num-epochs 4 \
  --no-llm \
  --device cuda \
  --save-dir runs/my_run_nollm/ckpts \
  --no-step-logs
```

### Output files

After training, the **parent** of `--save-dir` will contain:

| File | Contents |
|---|---|
| `train.csv` | Per-episode metrics (reward, loss, λ, timing, …) |
| `config.json` | All hyperparameters used for this run |
| `ckpts/ckpt_final.pt` | Final model checkpoint |
| `ckpts/ckpt_episode_N.pt` | Periodic checkpoints (every 20 episodes) |

---

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--scenario` | `tiny` | NASim benchmark scenario (`tiny`, `small`, `medium`, …) |
| `--episodes` | `100` | Number of training episodes |
| `--episode-length` | `500` | Max steps per episode |
| `--hidden-dim` | `128` | Hidden layer size of the student network |
| `--learning-rate` | `3e-4` | PPO optimizer learning rate |
| `--batch-size` | `32` | Minibatch size for PPO updates |
| `--num-epochs` | `3` | PPO update epochs per episode |
| `--lambda-mode` | `fixed` | λ schedule: `fixed`, `competence`, `target-kl`, `adaptive` |
| `--lambda-start` | `1.0` | Initial teacher weight λ |
| `--lambda-decay` | `0.99` | Per-episode decay factor (used by `fixed` mode) |
| `--lambda-min` | `0.0` | Minimum λ floor |
| `--llm-mix-weight` | `0.3` | Max fraction of LLM signal in option sampling |
| `--llama-model` | `llm_weights/llama-3.2-1B-Instruct` | Path to local Llama model |
| `--no-llm` | — | Disable LLM teacher (pure PPO baseline) |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--no-step-logs` | — | Skip per-step `.txt` log files (faster) |
| `--seed` | `0` | Random seed |

### λ modes explained

| Mode | Behaviour |
|---|---|
| `fixed` | Exponential decay: `λ = λ_start × λ_decay^t`. No feedback. |
| `competence` | Lowers λ as the agent's average compromise rate improves. |
| `target-kl` | Feedback loop: scales λ up/down to keep KL divergence near `--kl-target`. |
| `adaptive` | Combines `target-kl` feedback within a `competence` ceiling. |

---

## Grid sweep

The sweep script takes **lists** for most arguments and runs every combination automatically. For example, sweeping over 3 seeds × 3 λ modes × 2 learning rates (+ no-LLM baselines for each seed) in one command:

```bash
python -m nasim.scripts.run_llm4teach_sweep \
  --scenario small \
  --seeds 42 43 44 \
  --episodes 100 \
  --episode-length 500 \
  --hidden-dims 128 \
  --learning-rates 1e-4 3e-4 \
  --batch-sizes 64 \
  --num-epochs 4 \
  --lambda-modes fixed competence adaptive \
  --lambda-decays 0.97 \
  --lambda-min 0.05 \
  --llm-mix-weight 0.5 \
  --include-no-llm \
  --device cuda \
  --out-dir runs/my_sweep
```

This launches **3 seeds × 3 λ modes × 2 learning rates = 18 LLM runs** plus **3 no-LLM baselines** (one per seed), for 21 runs total. Each run gets its own subdirectory; a `manifest.jsonl` tracks all runs.

After all runs complete, `runs/my_sweep/summary.csv` is written automatically with one row per run.

To aggregate an existing sweep manually:

```bash
python -m nasim.scripts.aggregate_llm4teach_sweep \
  --sweep-dir runs/my_sweep \
  --output runs/my_sweep/summary.csv
```

---

## Analysis notebook

Open `analysis.ipynb` in Jupyter or VS Code. Edit the **Configure** cell (cell 4) to point at your run directories:

```python
RUNS_TO_COMPARE = [
    ("LLM run",    "runs/my_run"),
    ("no-LLM run", "runs/my_run_nollm"),
]
```

Then **Run All**. The notebook produces:
- Summary statistics table
- Reward learning curves
- Discovery & compromise rate plots
- Training loss dynamics (actor / critic / KL / clip)
- λ schedule & KL divergence
- Teacher agreement & LLM cache efficiency (LLM runs only)
- Timing breakdown (wall-clock per component)
- Final performance comparison bar charts
- Reward distribution box plots
- Pairwise Welch t-tests + Cohen's d effect sizes

---

## Project structure

```
nasim/
  train_llm4teach.py       # Main trainer
  agents/
    llm4teach_agent.py     # Option-based PPO student
  llm/
    llama_local.py         # Local Llama inference wrapper
    llm4teach_advisor.py   # LLM teacher: state → option scores + cache
    prompts.py             # Prompt templates
  scripts/
    run_llm4teach_sweep.py      # Grid sweep runner
    aggregate_llm4teach_sweep.py # Aggregates sweep → summary.csv
llm_weights/               # Local model weights (not in git)
runs/                      # Training outputs (not in git)
analysis.ipynb             # Interactive analysis notebook
```

---

## Architecture

```
Observation (flat vector)
       │
       ▼
  LLM Teacher                  PPO Student
  (Llama 3.2)  ──── λ ────►  Option policy
  score_options()              + action head
       │                          │
       └──── mixed π_T ───────────┘
                  │
                  ▼
             Environment step
```

At each step, a mixed policy `π = (1−w)·π_student + w·π_LLM` selects a high-level option, where `w = λ × llm_mix_weight`. The student is updated via PPO with an additional KL loss towards the teacher's distribution, weighted by λ. As training progresses, λ decreases and the student acts more independently.

---

## Citation / acknowledgements

Built on top of [NASim](https://github.com/Jjschwartz/NetworkAttackSimulator) by Jonathon Schwartz.  
LLM teacher uses [Meta Llama 3.2](https://huggingface.co/meta-llama) via 🤗 Transformers.
