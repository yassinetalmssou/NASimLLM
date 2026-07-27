"""Generate train_llm4teach commands for the extended RQ3 study.

Two sets, one teacher at a time (call once per teacher; redirect stdout to a
command file that hpc/jobs/rq3b.sh runs as a SLURM array):

  Set A - component ablations (extends runs/<teacher>/rq3/small/<condition>/):
      full, no_history, no_avoidlist, verbose_prompt, avoidlist_mask
      (avoidlist_mask = the NEW working avoid-list, --avoidlist-mask)

  Set B - hyperparameter sensitivity (runs/<teacher>/sens/small/<config>/):
      one knob varied from the backbone baseline, rest held fixed.

  Set C - recent-action history-length sweep (runs/<teacher>/sens/small/hist_<N>/):
      the teacher's recent-action log length varied over {4, 8, 16, 32, 64}.

Baseline matches hpc/submit_experiment.sh so results are comparable.

Usage:
  python -m nasim.scripts.gen_rq3_extended --set A --out-dir $VSC_SCRATCH/runs/llama-1B \
      --model $VSC_SCRATCH/llm_weights/llama-3.2-1B-Instruct --episodes 1000 --seeds 0 1 2 > cmds.txt
"""
import argparse
import sys
from pathlib import Path

# Backbone baseline (identical to submit_experiment.sh).
BASE = dict(episode_length=500, lambda_mode="competence", lambda_start=1.0,
            lambda_decay=0.97, lambda_min=0.05, kl_target=0.01,
            llm_mix_weight=0.5, teacher_temp=2.0, batch_size=64, num_epochs=4,
            history_len=8)

# Set A: component ablations -> extra trainer flags (all are llm_full).
SET_A = {
    "full":           [],
    "no_history":     ["--no-history"],
    "no_avoidlist":   ["--no-avoidlist"],
    "verbose_prompt": ["--verbose-prompt"],
    "avoidlist_mask": ["--avoidlist-mask"],
}

# Set B: one knob overridden from BASE (name -> (knob, value)).
# kl_target only affects target-kl/adaptive modes, so the schedule variants set
# lambda_mode directly; a bare kl_target sweep under the competence baseline is a no-op.
SET_B = {
    "lam_fixed":      ("lambda_mode", "fixed"),
    "sched_targetkl": ("lambda_mode", "target-kl"),
    "sched_adaptive": ("lambda_mode", "adaptive"),
    "lstart_0.5":     ("lambda_start", 0.5),
    "lstart_2.0":     ("lambda_start", 2.0),
    "mix_0.3":        ("llm_mix_weight", 0.3),
    "mix_0.8":        ("llm_mix_weight", 0.8),
    "temp_1.0":       ("teacher_temp", 1.0),
    "temp_4.0":       ("teacher_temp", 4.0),
}

# Set C: recent-action history-length sweep (RQ3 component sensitivity).
SET_C = {
    "hist_4":  ("history_len", 4),
    "hist_8":  ("history_len", 8),
    "hist_16": ("history_len", 16),
    "hist_32": ("history_len", 32),
    "hist_64": ("history_len", 64),
}


def _base_flags(knobs):
    return [
        "--episode-length", str(knobs["episode_length"]),
        "--lambda-mode", str(knobs["lambda_mode"]),
        "--lambda-start", str(knobs["lambda_start"]),
        "--lambda-decay", str(knobs["lambda_decay"]),
        "--lambda-min", str(knobs["lambda_min"]),
        "--kl-target", str(knobs["kl_target"]),
        "--llm-mix-weight", str(knobs["llm_mix_weight"]),
        "--teacher-temp", str(knobs["teacher_temp"]),
        "--batch-size", str(knobs["batch_size"]),
        "--num-epochs", str(knobs["num_epochs"]),
        "--history-len", str(knobs["history_len"]),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, choices=["A", "B", "C"])
    p.add_argument("--out-dir", required=True, help="$VSC_SCRATCH/runs/<teacher>")
    p.add_argument("--model", required=True, help="absolute path to the teacher weights")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda")
    p.add_argument("--scenario", default="small")
    p.add_argument("--partial-obs", action="store_false", dest="fully_obs",
                   help="Partial observability (default: fully observable)")
    p.add_argument("--belief-accum", action="store_true",
                   help="Aggregate partial observations into a running belief state")
    p.add_argument("--names", nargs="*", default=None,
                   help="subset of condition/config names (default: all in the set)")
    a = p.parse_args()

    table = {"A": SET_A, "B": SET_B, "C": SET_C}[a.set]
    sub = "rq3" if a.set == "A" else "sens"
    names = a.names if a.names else list(table.keys())

    out = Path(a.out_dir)
    for name in names:
        if name not in table:
            raise SystemExit(f"unknown {a.set} name: {name}")
        for seed in a.seeds:
            knobs = dict(BASE)
            extra = []
            if a.set == "A":
                extra = table[name]
            else:
                knob, val = table[name]
                knobs[knob] = val
            save_dir = out / sub / a.scenario / name / f"seed{seed}" / "ckpts"
            cmd = [
                sys.executable, "-m", "nasim.train_llm4teach",
                "--scenario", a.scenario, "--episodes", str(a.episodes),
                "--seed", str(seed), "--device", a.device,
                "--llama-model", a.model,
                *_base_flags(knobs),
                "--save-dir", str(save_dir), "--condition", name, "--no-step-logs",
                *extra,
                *(["--partial-obs"] if not a.fully_obs else []),
                *(["--belief-accum"] if a.belief_accum else []),
            ]
            print(" ".join(cmd))


if __name__ == "__main__":
    main()
