"""Parse LLM4Teach training stdout into a CSV for statistical analysis.

Usage:
  python -m nasim.scripts.parse_llm4teach_output --input output.md --output runs/llm4teach_tiny_seed0.csv

It extracts per-episode metrics from lines like:
  Episode 80/100 | Reward: 101.90 ... | Length: 100 ... | λ: 0.4520
    Hosts compromised: 2/2 | Discovery: 100.0% | Loss (RL/KL): 587.067 / 0.183 | Clip: 0.00%

Notes
-----
- This is a best-effort parser for the current log format.
- If metrics are missing for an episode, the field is left blank.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List


_EP_RE = re.compile(
    r"Episode\s+(?P<ep>\d+)\/\d+\s+\|\s+Reward:\s+(?P<reward>-?\d+(?:\.\d+)?)"
    r".*?\|\s+Length:\s+(?P<length>\d+)"
    r".*?\|\s+λ:\s+(?P<lambda>\d+(?:\.\d+)?)"
)

_MET_RE = re.compile(
    r"Hosts compromised:\s+(?P<hc>\d+)\/(?P<hc_total>\d+)\s+\|\s+Discovery:\s+(?P<disc>\d+(?:\.\d+)?)%"
    r"\s+\|\s+Loss\s+\(RL\/KL\):\s+(?P<loss_rl>-?\d+(?:\.\d+)?)\s+\/\s+(?P<loss_kl>-?\d+(?:\.\d+)?)"
    r"\s+\|\s+Clip:\s+(?P<clip>\d+(?:\.\d+)?)%"
)


@dataclass
class EpisodeRow:
    episode: int
    reward: float
    length: int
    lambda_kl: float
    hosts_compromised: Optional[int] = None
    hosts_total: Optional[int] = None
    discovery_pct: Optional[float] = None
    loss_rl: Optional[float] = None
    loss_kl: Optional[float] = None
    clip_pct: Optional[float] = None


def parse_file(path: Path) -> List[EpisodeRow]:
    rows: List[EpisodeRow] = []
    current: Optional[EpisodeRow] = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m1 = _EP_RE.search(line)
        if m1:
            if current is not None:
                rows.append(current)
            current = EpisodeRow(
                episode=int(m1.group("ep")),
                reward=float(m1.group("reward")),
                length=int(m1.group("length")),
                lambda_kl=float(m1.group("lambda")),
            )
            continue

        if current is None:
            continue

        m2 = _MET_RE.search(line)
        if m2:
            current.hosts_compromised = int(m2.group("hc"))
            current.hosts_total = int(m2.group("hc_total"))
            current.discovery_pct = float(m2.group("disc"))
            current.loss_rl = float(m2.group("loss_rl"))
            current.loss_kl = float(m2.group("loss_kl"))
            current.clip_pct = float(m2.group("clip"))

    if current is not None:
        rows.append(current)

    return rows


def write_csv(rows: List[EpisodeRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(EpisodeRow(0, 0.0, 0, 0.0)).keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main() -> None:
    p = argparse.ArgumentParser(description="Parse LLM4Teach stdout log into CSV")
    p.add_argument("--input", type=str, required=True, help="Path to stdout log (e.g., output.md)")
    p.add_argument("--output", type=str, required=True, help="Path to output CSV")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows = parse_file(in_path)
    if not rows:
        raise SystemExit(f"No episodes parsed from {in_path}")

    write_csv(rows, out_path)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
