"""Compare showing both personas against showing only the speaker's own.

Sequential generation originally put both speakers' personas in every prompt, so
each utterance was written with the partner's persona in view. These cells add the
own-persona-only condition against matched controls. The paper's cells use
own-persona-only.

Also defines the per-cell loader that the other scripts import.

Requires novelty_dynamics.py and compute_pid.py to have been run.

    uv run python scripts/profile_visibility.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from semantic_closure import config, pid
from semantic_closure.data import (
    iter_llm_llm_dialogues,
    iter_llm_llm_oneshot_dialogues,
)

from generation_factors import ATOM_COLS, PID_COLS, PID_LABEL, series_stats

LLM_LLM = ("llm_llm", iter_llm_llm_dialogues)
ONESHOT = ("llm_llm_oneshot", iter_llm_llm_oneshot_dialogues)

#: namespace -> (label, corpus, loader)
CELLS = {
    2: ("both, no examples, no length", *LLM_LLM),
    9: ("both, examples, length", *ONESHOT),
    13: ("own only, examples, length", *ONESHOT),
    14: ("own only, no examples, no length", *ONESHOT),
}

#: (both, own only) pairs that flip persona visibility with the rest matched.
PAIRS = [("examples, length", 9, 13), ("no examples, no length", 2, 14)]

PAIR_N = 10  # dialogues per cell when reading a one-factor difference
FULL_N = 50  # dialogues per cell when reading a condition on its own
HUMAN_LEN = 22.9  # mean utterance length over the whole human--human corpus


def _load(kind: str, corpus: str | None, namespace: int | None, limit: int) -> pd.DataFrame:
    suffix = f"__{corpus}" if corpus else ""
    path = config.OUTPUT_DIR / "metrics" / f"{kind}__{config.EMBEDDING_MODEL}{suffix}.parquet"
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing. Run novelty_dynamics.py and compute_pid.py first.")
    df = pd.read_parquet(path)
    if "variant" in df.columns:
        df = df[df["variant"] == "real"]
    if namespace is not None:
        df = df[(df["dialogue_id"] // 100000) == namespace]
        df = df[(df["dialogue_id"] % 100000) <= limit]
    return df


class Cell:
    """Novelty, atoms and utterance length of one cell; limit caps the dialogues."""

    def __init__(self, ns: int | None, limit: int) -> None:
        if ns is None:  # human×human
            self.label, corpus, loader = "human×human", None, None
        else:
            self.label, corpus, loader = CELLS[ns]
        self.ns, self.limit = ns, limit
        self.nov = series_stats(_load("novelty_adjacent", corpus, ns, limit))
        self.pid = pid.dialogue_means(_load("pid", corpus, ns, limit))
        if loader is None:
            self.utt_len = HUMAN_LEN
        else:
            ids = sorted(self.pid["dialogue_id"].unique())
            self.utt_len = float(np.mean(
                [len(u.text) for d in loader(dialogue_ids=ids) for u in d.utterances]))

    @property
    def n_series(self) -> int:
        return len(self.nov)

    @property
    def slope(self) -> float:
        return float(self.nov["slope"].mean())

    @property
    def ci(self) -> float:
        s = self.nov["slope"]
        return float(1.96 * s.std(ddof=1) / np.sqrt(len(s)))

    @property
    def mean_n(self) -> float:
        return float(self.nov["mean"].mean())

    def atoms(self) -> dict[str, float]:
        assert abs(self.pid[ATOM_COLS].mean().sum() - 1.0) < 1e-9, \
            f"atoms do not sum to one: ns{self.ns}"
        return {c: float(self.pid[c].mean()) for c in PID_COLS}


def novelty_row(name: str, c: Cell) -> str:
    return (f"{name:34s} {c.n_series:>4d} {c.mean_n:6.3f} {c.slope:+8.3f} "
            f"[{c.slope - c.ci:+7.3f},{c.slope + c.ci:+7.3f}] {c.utt_len:7.1f}")


def novelty_header(head: str = "condition") -> str:
    return (f'{head:34s} {"n":>4s} {"mean N":>6s} {"slope":>8s} '
            f'{"95% CI":>18s} {"length":>7s}')


def atom_header(head: str = "condition") -> str:
    return f"{head:34s} " + " ".join(f"{PID_LABEL[c]:>8s}" for c in PID_COLS)


def atom_row(name: str, atoms: dict[str, float], human: dict[str, float] | None) -> str:
    line = f"{name:34s} " + " ".join(f"{atoms[c]:8.3f}" for c in PID_COLS)
    if human is not None:
        line += f" {sum(abs(atoms[c] - human[c]) for c in PID_COLS):9.3f}"
    return line


def main() -> None:
    human = Cell(None, FULL_N)
    h_atoms = human.atoms()
    pair_cells = {ns: Cell(ns, PAIR_N) for _, lo, hi in PAIRS for ns in (lo, hi)}

    print(f"== Flipped persona visibility ({PAIR_N} dialogues per cell)")
    print(novelty_header())
    for factors, lo, hi in PAIRS:
        for ns in (lo, hi):
            print(novelty_row(f"ns{ns} {pair_cells[ns].label}", pair_cells[ns]))
    print(novelty_row("human--human", human))

    print("\n== Effect of persona visibility (both -> own only)")
    print(f'{"other factors":24s} {"d slope":>9s} {"d mean N":>10s} {"d length":>11s}')
    for factors, lo, hi in PAIRS:
        a, b = pair_cells[lo], pair_cells[hi]
        print(f"{factors:24s} {b.slope - a.slope:+9.3f} {b.mean_n - a.mean_n:+10.3f} "
              f"{b.utt_len - a.utt_len:+11.1f}")

    print(f"\n== Atoms, and their distance from human--human ({PAIR_N} dialogues)")
    print(atom_header() + f' {"total":>9s}')
    for factors, lo, hi in PAIRS:
        for ns in (lo, hi):
            print(atom_row(f"ns{ns} {pair_cells[ns].label}",
                           pair_cells[ns].atoms(), h_atoms))
    print(atom_row("human--human", h_atoms, None))

    print(f"\n== Each condition on its own ({FULL_N} dialogues, unlike the tables above)")
    print(novelty_header())
    full = {}
    for ns in (13, 14, 2):
        full[ns] = Cell(ns, FULL_N)
        print(novelty_row(f"ns{ns} {full[ns].label}", full[ns]))
    print(novelty_row("human--human", human))
    print()
    print(atom_header() + f' {"total":>9s}')
    for ns in (13, 14, 2):
        print(atom_row(f"ns{ns} {full[ns].label}", full[ns].atoms(), h_atoms))
    print(atom_row("human--human", h_atoms, None))


if __name__ == "__main__":
    main()
