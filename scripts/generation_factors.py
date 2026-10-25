"""Separate the three generation factors in a 2x2x2 design.

Changing three conditions at once removes the decay of novelty; varying them
independently says which factor moves which measurement. The factors are

- unit:   one call per utterance (seq) or one call per dialogue (batch)
- examples: whether two real RealPersonaChat dialogues are shown
- length: whether the prompt asks for 10 to 40 characters per utterance

All eight cells use the same model and the same first 10 dialogues.

Also defines the column sets and the per-series statistics that the other scripts
import.

Requires novelty_dynamics.py and compute_pid.py to have been run.

    uv run python scripts/generation_factors.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from semantic_closure import config, pid
from semantic_closure.data import (
    iter_dialogues,
    iter_llm_llm_dialogues,
    iter_llm_llm_oneshot_dialogues,
)

N_DIALOGUES = 10  # matched across cells; the larger cells are truncated to it

#: namespace -> (unit, examples, length, corpus, loader); the four seq cells first.
CELLS = [
    (2, "seq", False, False, "llm_llm", iter_llm_llm_dialogues),
    (11, "seq", False, True, "llm_llm_oneshot", iter_llm_llm_oneshot_dialogues),
    (12, "seq", True, False, "llm_llm_oneshot", iter_llm_llm_oneshot_dialogues),
    (9, "seq", True, True, "llm_llm_oneshot", iter_llm_llm_oneshot_dialogues),
    (10, "batch", False, False, "llm_llm_oneshot", iter_llm_llm_oneshot_dialogues),
    (8, "batch", False, True, "llm_llm_oneshot", iter_llm_llm_oneshot_dialogues),
    (7, "batch", True, False, "llm_llm_oneshot", iter_llm_llm_oneshot_dialogues),
    (6, "batch", True, True, "llm_llm_oneshot", iter_llm_llm_oneshot_dialogues),
]

ATOM_COLS = ["redundancy", "unique_self", "unique_other", "synergy", "novelty"]
PID_COLS = ["q_x", "q_y", *ATOM_COLS, "omega"]
PID_LABEL = {"q_x": "Q_X", "q_y": "Q_Y", "redundancy": "R", "unique_self": "U_self",
             "unique_other": "U_other", "synergy": "S", "novelty": "N", "omega": "Ω"}

YN = {True: "yes", False: "no"}


def _load(kind: str, corpus: str | None, namespace: int | None) -> pd.DataFrame:
    suffix = f"__{corpus}" if corpus else ""
    path = config.OUTPUT_DIR / "metrics" / f"{kind}__{config.EMBEDDING_MODEL}{suffix}.parquet"
    if not path.is_file():
        raise SystemExit(f"{path} is missing. Run novelty_dynamics.py and compute_pid.py first.")
    df = pd.read_parquet(path)
    if "variant" in df.columns:
        df = df[df["variant"] == "real"]
    if namespace is not None:
        df = df[(df["dialogue_id"] // 100000) == namespace]
        df = df[(df["dialogue_id"] % 100000) <= N_DIALOGUES]
    return df


def series_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Slope, mean and angular SD per (dialogue, role)."""
    rows = []
    for (did, role), g in df.groupby(["dialogue_id", "role"]):
        y = g.sort_values("t")["novelty"].to_numpy()
        if len(y) < 4:
            continue
        s = np.linspace(0.0, 1.0, len(y))
        th = np.degrees(np.arcsin(np.sqrt(np.clip(y, 0.0, 1.0))))
        # The slope is the coefficient against utterance index, per 10 utterances.
        # The target is 2t+1 for role X and 2t+2 for role Y, so u = 2t + c and
        # dN/du = (dN/dt)/2; ten utterances give the factor 5. angle_sd is a
        # residual and is unchanged by that linear rescaling of x.
        rows.append({
            "slope": 5.0 * np.polyfit(g.sort_values("t")["t"].to_numpy(float), y, 1)[0],
            "mean": y.mean(),
            "angle_sd": (th - np.polyval(np.polyfit(s, th, 1), s)).std(ddof=2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    nov, pids, lens = {}, {}, {}
    for ns, gen, ex, ln, corpus, loader in CELLS:
        nov[ns] = series_stats(_load("novelty_adjacent", corpus, ns))
        pids[ns] = pid.dialogue_means(_load("pid", corpus, ns))
        ids = sorted(pids[ns]["dialogue_id"].unique())
        lens[ns] = np.array([len(u.text) for d in loader(dialogue_ids=ids) for u in d.utterances])
    h_nov = series_stats(_load("novelty_adjacent", None, None))
    h_pid = pid.dialogue_means(_load("pid", None, None))
    h_len = 22.9  # mean utterance length over the whole human--human corpus

    print("== Novelty (10 dialogues per cell, same seeds)")
    print(f'{"ns":>4s} {"unit":>5s} {"ex":>4s} {"len":>4s} {"mean N":>6s} {"slope":>8s} '
          f'{"95% CI":>18s} {"angleSD":>7s} {"length":>7s}')
    for ns, gen, ex, ln, *_ in CELLS:
        s = nov[ns]["slope"]
        ci = 1.96 * s.std(ddof=1) / np.sqrt(len(s))
        print(f"{ns:>4d} {gen:>4s} {YN[ex]:>4s} {YN[ln]:>4s} {nov[ns]['mean'].mean():6.3f} "
              f"{s.mean():+8.3f} [{s.mean() - ci:+7.3f},{s.mean() + ci:+7.3f}] "
              f"{nov[ns]['angle_sd'].mean():7.2f} {lens[ns].mean():7.1f}")
    print(f'{"human":>4s} {"-":>5s} {"-":>4s} {"-":>4s} {h_nov["mean"].mean():6.3f} '
          f'{h_nov["slope"].mean():+8.3f} {"":>18s} {h_nov["angle_sd"].mean():7.2f} {h_len:7.1f}')

    # Main effect: average the difference over the four pairs that flip one factor
    # while the other two stay matched.
    def pairs(factor: int):
        """The (low, high) namespace pairs that flip factor 0=unit, 1=examples, 2=length."""
        key = {(gen == "batch", ex, ln): ns for ns, gen, ex, ln, *_ in CELLS}
        out = []
        for combo, ns_lo in key.items():
            if combo[factor]:
                continue
            hi = list(combo)
            hi[factor] = True
            out.append((ns_lo, key[tuple(hi)]))
        return out

    print("\n== Main effect on novelty (mean of the four flipped pairs)")
    print(f'{"factor":16s} {"d slope":>9s} {"d mean N":>10s} {"per pair (slope)":>34s}')
    for fi, label in ((0, "seq -> batch"), (1, "examples no -> yes"), (2, "length no -> yes")):
        ps = pairs(fi)
        d_slope = [nov[hi]["slope"].mean() - nov[lo]["slope"].mean() for lo, hi in ps]
        d_mean = [nov[hi]["mean"].mean() - nov[lo]["mean"].mean() for lo, hi in ps]
        detail = " ".join(f"{d:+.3f}" for d in d_slope)
        print(f"{label:16s} {np.mean(d_slope):+9.3f} {np.mean(d_mean):+10.3f} {detail:>34s}")

    print("\n== Atoms (10 dialogues per cell, same seeds)")
    print(f'{"ns":>4s} {"unit":>5s} {"ex":>4s} {"len":>4s} '
          + " ".join(f"{PID_LABEL[c]:>8s}" for c in PID_COLS))
    for ns, gen, ex, ln, *_ in CELLS:
        d = pids[ns]
        assert abs(d[ATOM_COLS].mean().sum() - 1.0) < 1e-9, f"atoms do not sum to one: ns{ns}"
        print(f"{ns:>4d} {gen:>4s} {YN[ex]:>4s} {YN[ln]:>4s} "
              + " ".join(f"{d[c].mean():8.3f}" for c in PID_COLS))
    print(f'{"human":>4s} {"-":>5s} {"-":>4s} {"-":>4s} '
          + " ".join(f"{h_pid[c].mean():8.3f}" for c in PID_COLS))

    print("\n== Main effect on the atoms")
    print(f'{"factor":16s} ' + " ".join(f"{PID_LABEL[c]:>8s}" for c in PID_COLS))
    for fi, label in ((0, "seq -> batch"), (1, "examples no -> yes"), (2, "length no -> yes")):
        ps = pairs(fi)
        row = [np.mean([pids[hi][c].mean() - pids[lo][c].mean() for lo, hi in ps])
               for c in PID_COLS]
        print(f"{label:16s} " + " ".join(f"{v:+8.3f}" for v in row))

    print("\n== Departure from human--human (cell - human)")
    print(f'{"ns":>4s} ' + " ".join(f"{PID_LABEL[c]:>8s}" for c in PID_COLS)
          + f' {"total":>9s}')
    for ns, *_ in CELLS:
        diff = [pids[ns][c].mean() - h_pid[c].mean() for c in PID_COLS]
        print(f"{ns:>4d} " + " ".join(f"{v:+8.3f}" for v in diff)
              + f" {sum(abs(v) for v in diff):9.3f}")


if __name__ == "__main__":
    main()
