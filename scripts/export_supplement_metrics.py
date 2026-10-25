"""Write the paper's per-cell measurements to CSV: the five atoms, the three
cosines and the novelty slope.

Lets the numbers be checked without the utterance text. Reads the output of
compute_pid.py from outputs/metrics/pid__*.parquet.

    uv run python scripts/export_supplement_metrics.py [--out supplement/metrics]

Writes three CSVs:
    per_series.csv   per (group, dialogue_id, role): turn means of the atoms and
                     cosines, and the OLS novelty slope. Covers all 27,162
                     human--human series, 150 dialogues for each of the five
                     models, and the additional Opus cells.
    per_turn_llm.csv the same quantities per turn, for the LLM cells only
                     (human--human would exceed 400,000 rows)
    groups.csv       group names and their generation conditions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from semantic_closure import config

ATOMS = ["redundancy", "unique_self", "unique_other", "synergy", "novelty"]
COS = ["s_xz", "s_yz", "s_xy", "q_x", "q_y", "q_xy"]

#: The paper's cells: namespace -> (group name, description).
CELLS = {
    20: ("claude-haiku-4-5", "150 dialogues, 20 utterances, 10 persona items, main comparison"),
    24: ("claude-sonnet-5", "150 dialogues, 20 utterances, 10 persona items, main comparison"),
    25: ("claude-opus-5", "150 dialogues, 20 utterances, 10 persona items, main comparison"),
    22: ("gpt-5-nano", "150 dialogues, 20 utterances, 10 persona items, main comparison"),
    21: ("gpt-4o-mini", "150 dialogues, 20 utterances, 10 persona items, main comparison"),
    30: ("claude-opus-5__persona-3", "150 dialogues, 3 persona items"),
    31: ("claude-opus-5__persona-0", "150 dialogues, no persona"),
    32: ("claude-opus-5__persona-10__to60", "50 dialogues continued to 60 utterances, 10 persona items"),
    33: ("claude-opus-5__persona-0__to60", "50 dialogues continued to 60 utterances, no persona"),
    34: ("claude-opus-5__persona-3__to60", "50 dialogues continued to 60 utterances, 3 persona items"),
}


def slope(g: pd.DataFrame) -> float:
    """Novelty slope per 10 utterances.

    The target utterance is 2t+1 for role X and 2t+2 for role Y, so u = 2t + c and
    dN/du = (dN/dt)/2. Times 10 utterances gives the factor 5.
    """
    g = g.sort_values("t")
    y = g.novelty.to_numpy()
    if len(y) < 3:
        return float("nan")
    return float(5.0 * np.polyfit(g["t"].to_numpy(float), y, 1)[0])


def per_series(df: pd.DataFrame, group: str) -> pd.DataFrame:
    keys = ["dialogue_id", "role"]
    means = df.groupby(keys)[ATOMS + COS].mean()
    sl = df.groupby(keys).apply(slope, include_groups=False)
    sl.name = "novelty_slope"
    n = df.groupby(keys).size().rename("n_turns")
    out = pd.concat([means, sl, n], axis=1).reset_index()
    out.insert(0, "group", group)
    return out


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(config.OUTPUT_DIR / "metrics" / f"pid__{config.EMBEDDING_MODEL}{name}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="supplement/metrics", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    series, turns, groups = [], [], []
    series.append(per_series(load(""), "human-human"))
    groups.append(("human-human", "RealPersonaChat, 13,581 dialogues, all utterances"))
    real = load("__llm_llm_oneshot")
    for ns, (group, desc) in CELLS.items():
        q = real[real.dialogue_id // 100000 == ns]
        if q.empty:
            raise SystemExit(f"namespace {ns} ({group}) is not in the parquet")
        series.append(per_series(q, group))
        t = q.copy()
        t.insert(0, "group", group)
        turns.append(t[["group", "dialogue_id", "role", "t"] + ATOMS + COS])
        groups.append((group, desc))

    pd.concat(series).to_csv(args.out / "per_series.csv", index=False, float_format="%.6f")
    pd.concat(turns).to_csv(args.out / "per_turn_llm.csv", index=False, float_format="%.6f")
    pd.DataFrame(groups, columns=["group", "description"]).to_csv(args.out / "groups.csv", index=False)
    print(f"wrote {args.out}/per_series.csv ({sum(len(s) for s in series)} rows), per_turn_llm.csv ({sum(len(t) for t in turns)} rows), groups.csv")


if __name__ == "__main__":
    main()
