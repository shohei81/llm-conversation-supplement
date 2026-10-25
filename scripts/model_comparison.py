"""Table 1 of the paper, plus the atom composition of each cell.

Prints, for every model cell and for human--human dialogue, the mean novelty and
its within-conversation slope with a 95% CI, then the five atoms and their
distance from the human profile.

Only cells generated under identical conditions are put side by side: the
comparison is between models and against human dialogue, so any other factor left
unmatched would make the difference unattributable.

Requires compute_pid.py to have been run for human and llm_llm_oneshot.

    uv run python scripts/model_comparison.py
"""

from __future__ import annotations

import numpy as np

from semantic_closure import pid
from semantic_closure.data import iter_llm_llm_oneshot_dialogues

from generation_factors import ATOM_COLS, PID_COLS, PID_LABEL, series_stats
from profile_visibility import HUMAN_LEN, _load

ONESHOT = ("llm_llm_oneshot", iter_llm_llm_oneshot_dialogues)

#: Cell set -> (n dialogues, [(namespace, model, release, corpus, loader)]).
#: The order here is the order of the table rows. v2 is the comparison used in the
#: paper: 150 dialogues per model, prompt revision 2, seeds drawn at random.
#: The generation conditions of each cell are in its manifest__*.json.
CELL_SETS = {
    "v1": (50, [
        (17, "claude-haiku-4-5", "2025-10", *ONESHOT),
        (19, "gpt-5-nano", "2025-08", *ONESHOT),
        (18, "gpt-4o-mini", "2024-07", *ONESHOT),
    ]),
    "v2": (150, [
        (20, "claude-haiku-4-5", "2025-10", *ONESHOT),
        (24, "claude-sonnet-5", "2026-06", *ONESHOT),
        (25, "claude-opus-5", "2026-07", *ONESHOT),
        (22, "gpt-5-nano", "2025-08", *ONESHOT),
        (21, "gpt-4o-mini", "2024-07", *ONESHOT),
    ]),
    # Two cells for the in-context-example comparison: ns23 repeats ns20 on the
    # same seeds and prompt, with two real dialogues placed in the prompt.
    "examples": (150, [
        (20, "claude-haiku-4-5", "2025-10", *ONESHOT),
        (23, "claude-haiku-4-5 + examples", "2025-10", *ONESHOT),
    ]),
}
#: Filename prefix of the figures, so one cell set does not overwrite another.
FIG_PREFIX = {"v1": "model_comparison", "v2": "model_comparison_150",
              "examples": "model_comparison_examples"}


def select_cells() -> tuple[str, int, list]:
    """Cell set named by --cells, defaulting to v2.

    The figure script imports this too, so passing the same --cells keeps the
    tables and the figures on the same cells.
    """
    import sys

    cells = "v2"
    if "--cells" in sys.argv:
        cells = sys.argv[sys.argv.index("--cells") + 1]
    if cells not in CELL_SETS:
        raise SystemExit(f"--cells must be one of {sorted(CELL_SETS)}, got {cells}")
    return (cells, *CELL_SETS[cells])


CELLS, N_DIALOGUES, ROWS = select_cells()


def main() -> None:
    # Levels and slopes come from the pid file, not from novelty_adjacent. That
    # file exists for the direction of the novelty vector and drops turns where
    # novelty is exactly zero, leaving the direction undefined. Those are the
    # lowest-novelty turns, so levels and slopes taken from it would be biased up.
    h_nov = series_stats(_load("pid", None, None, N_DIALOGUES))
    h_pid = pid.dialogue_means(_load("pid", None, None, N_DIALOGUES))
    h_atoms = {c: float(h_pid[c].mean()) for c in PID_COLS}

    print(f"== Novelty ({N_DIALOGUES} dialogues per cell, same prompt and seeds)")
    print(f'{"model":22s} {"release":>9s} {"mean N":>6s} {"slope":>8s} '
          f'{"95% CI":>18s} {"length":>7s}')
    cells = {}
    for ns, model, release, corpus, loader in ROWS:
        nov = series_stats(_load("pid", corpus, ns, N_DIALOGUES))
        cells[ns] = pid.dialogue_means(_load("pid", corpus, ns, N_DIALOGUES))
        ids = sorted(cells[ns]["dialogue_id"].unique())
        utt_len = float(np.mean(
            [len(u.text) for d in loader(dialogue_ids=ids) for u in d.utterances]))
        s = nov["slope"]
        ci = 1.96 * s.std(ddof=1) / np.sqrt(len(s))
        print(f"{model:22s} {release:>9s} {nov['mean'].mean():6.3f} {s.mean():+8.3f} "
              f"[{s.mean() - ci:+7.3f},{s.mean() + ci:+7.3f}] {utt_len:7.1f}")
    print(f'{"human--human":22s} {"-":>9s} {h_nov["mean"].mean():6.3f} '
          f'{h_nov["slope"].mean():+8.3f} {"":>18s} {HUMAN_LEN:7.1f}')

    print("\n== Atoms, and their total distance from human--human")
    print(f'{"model":22s} ' + " ".join(f"{PID_LABEL[c]:>8s}" for c in PID_COLS)
          + f' {"total":>9s}')
    for ns, model, *_ in ROWS:
        vals = {c: float(cells[ns][c].mean()) for c in PID_COLS}
        dist = sum(abs(vals[c] - h_atoms[c]) for c in PID_COLS)
        print(f"{model:22s} " + " ".join(f"{vals[c]:8.3f}" for c in PID_COLS)
              + f" {dist:9.3f}")
    print(f'{"human--human":22s} ' + " ".join(f"{h_atoms[c]:8.3f}" for c in PID_COLS))

    # The paper's sum of |departures| covers the five atoms only. The "total"
    # column above sums eight quantities, Q_X, Q_Y and omega included, so the two
    # differ on the same row.
    print("\n== Departure from human--human (model - human); last column sums the five atoms")
    print(f'{"model":22s} ' + " ".join(f"{PID_LABEL[c]:>8s}" for c in PID_COLS)
          + f' {"sum":>8s}')
    for ns, model, *_ in ROWS:
        diff = {c: float(cells[ns][c].mean()) - h_atoms[c] for c in PID_COLS}
        sigma = sum(abs(diff[c]) for c in ATOM_COLS)
        print(f"{model:22s} " + " ".join(f"{diff[c]:+8.3f}" for c in PID_COLS)
              + f" {sigma:8.3f}")


if __name__ == "__main__":
    main()
