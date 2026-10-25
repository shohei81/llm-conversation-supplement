"""Figures for the persona-visibility comparison.

Mean novelty against conversation position, the slopes with 95% CIs, and the atom
composition, with both personas shown against the speaker's own only.

Also defines the atom colors and the position grid that the other figure scripts
import.

Same prerequisites as profile_visibility.py.

    uv run python scripts/profile_visibility_figures.py

Writes docs/images/profile_visibility_trend.png
       docs/images/profile_visibility_slope.png
      docs/images/profile_visibility_atoms.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from semantic_closure import pid
from profile_visibility import (
    CELLS,
    FULL_N,
    PAIR_N,
    PAIRS,
    Cell,
    _load,
)

plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BOTH_COLOR, OWN_COLOR, HUMAN_COLOR = "#eb6834", "#2a78d6", "#555555"
GRID = np.linspace(0.0, 1.0, 10)

#: (column, display name, color), shared with the other atom figures.
ATOM_STYLE = [
    ("redundancy", "R (redundancy)", "#2a78d6"),
    ("unique_self", "U_self (unique, own side)", "#eb6834"),
    ("unique_other", "U_other (unique, partner)", "#1baf7a"),
    ("synergy", "S (synergy)", "#eda100"),
    ("novelty", "N(novelty)", "#d9d9d9"),
]


def visibility(ns: int) -> str:
    """Whether the cell showed the partner's persona; the label starts with that."""
    return CELLS[ns][0].split(", ")[0]


def color_of(ns: int) -> str:
    return OWN_COLOR if visibility(ns) == "own only" else BOTH_COLOR


def profiles(df: pd.DataFrame) -> np.ndarray:
    """Novelty of each series, linearly interpolated onto the normalized grid."""
    out = []
    for _, g in df.groupby(["dialogue_id", "role"]):
        y = g.sort_values("t")["novelty"].to_numpy()
        if len(y) < 4:
            continue
        out.append(np.interp(GRID, np.linspace(0.0, 1.0, len(y)), y))
    return np.array(out)


def fig_trend(out: Path) -> None:
    """Mean novelty against conversation position, one panel per control pair."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)
    h_prof = profiles(_load("novelty_adjacent", None, None, FULL_N))

    for ax, (factors, lo, hi) in zip(axes, PAIRS):
        # ns9 has only 10 dialogues, so that pair alone is capped at 10.
        limit = PAIR_N if lo == 9 else FULL_N
        for ns in (lo, hi):
            corpus = CELLS[ns][1]
            p = profiles(_load("novelty_adjacent", corpus, ns, limit))
            m, sd = p.mean(axis=0), p.std(axis=0)
            ax.fill_between(GRID, m - sd, m + sd, color=color_of(ns), alpha=0.11,
                            linewidth=0)
            ax.plot(GRID, m, color=color_of(ns), linewidth=2.0, marker="o", markersize=4,
                    label=f"{visibility(ns)} persona (ns{ns}, {limit} dialogues)")
        m = h_prof.mean(axis=0)
        ax.plot(GRID, m, color=HUMAN_COLOR, linewidth=2.6, linestyle="--",
                label="human--human")
        ax.set_title(f"examples and length: {factors}", fontsize=10.5)
        ax.set_xlabel("normalized position in the conversation (0 = start, 1 = end)", fontsize=9.5)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        ax.grid(color="#eeeef1", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("novelty N", fontsize=9.5)
    axes[0].set_ylim(0.2, 1.0)
    fig.suptitle("Novelty decay with the partner's persona shown or withheld "
                 "(claude-haiku-4-5, sequential)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def fig_slope(pair_cells: dict[int, Cell], full_cells: dict[int, Cell],
              h_slope: float, out: Path) -> None:
    """Slopes with 95% CIs: the one-factor pairs above, the larger cells below."""
    rows_top = [(ns, pair_cells[ns]) for _, lo, hi in PAIRS for ns in (lo, hi)]
    rows_bottom = [(ns, full_cells[ns]) for ns in (13, 14, 2)]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [4, 3]})

    for ax, rows, title in (
        (axes[0], rows_top, f"one-factor difference ({PAIR_N} dialogues per cell)"),
        (axes[1], rows_bottom, f"larger cells ({FULL_N} dialogues each)"),
    ):
        ys = range(len(rows))[::-1]
        for y, (ns, c) in zip(ys, rows):
            ax.errorbar(c.slope, y, xerr=c.ci, fmt="o", markersize=7,
                        color=color_of(ns), capsize=4, linewidth=1.8,
                        markeredgecolor="white")
        ax.axvline(h_slope, color=HUMAN_COLOR, linewidth=1.6, linestyle="--")
        ax.axvline(0, color="#cccccc", linewidth=0.9)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([f"ns{ns} {c.label}" for ns, c in rows], fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="x", color="#eeeef1", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    axes[0].axhline(1.5, color="#bbbbbb", linewidth=0.8, linestyle=":")  # between pairs
    axes[1].set_xlabel("within-conversation novelty slope (OLS, per 10 utterances)",
                       fontsize=9.5)
    handles = [
        plt.Line2D([], [], marker="o", color=BOTH_COLOR, linestyle="",
                   label="both personas"),
        plt.Line2D([], [], marker="o", color=OWN_COLOR, linestyle="",
                   label="own persona only"),
        plt.Line2D([], [], color=HUMAN_COLOR, linestyle="--",
                   label=f"human×human({h_slope:+.3f})"),
    ]
    axes[1].legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left")
    fig.suptitle("Persona visibility and the decay of novelty (bars = 95% CI)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def fig_atoms(pair_cells: dict[int, Cell], h_pid: pd.DataFrame, out: Path) -> None:
    """Atom composition of the four one-factor cells and human--human."""
    rows = [(f"ns{ns} {pair_cells[ns].label}", pair_cells[ns].pid)
            for _, lo, hi in PAIRS for ns in (lo, hi)]
    rows.append(("human×human", h_pid))
    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    ys = range(len(rows))[::-1]
    for y, (name, df) in zip(ys, rows):
        means = df[[c for c, _, _ in ATOM_STYLE]].mean()
        left, thin = 0.0, 0
        for col, _, color in ATOM_STYLE:
            v = float(means[col])
            ax.barh(y, v, left=left, height=0.62, color=color,
                    edgecolor="white", linewidth=1.5)
            if v >= 0.05:
                ax.text(left + v / 2, y, f"{v:.2f}", ha="center", va="center",
                        fontsize=8.5, color="#333333")
            else:
                ax.text(left + v / 2, y + 0.34 + 0.22 * (thin % 2), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5, color="#555555")
                thin += 1
            left += v
        assert abs(left - 1.0) < 1e-9  # the atoms sum to one
    ax.axhline(0.5, color="#bbbbbb", linewidth=0.8, linestyle="--")  # cells vs human
    ax.axhline(2.5, color="#bbbbbb", linewidth=0.8, linestyle=":")   # between pairs
    ax.set_yticks(list(ys))
    ax.set_yticklabels([n for n, _ in rows], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share (1 = R + U_self + U_other + S + N)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for _, _, c in ATOM_STYLE]
    ax.legend(handles, [n for _, n, _ in ATOM_STYLE], loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"Persona visibility and the atom composition ({PAIR_N} dialogues)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    pair_cells = {ns: Cell(ns, PAIR_N) for _, lo, hi in PAIRS for ns in (lo, hi)}
    full_cells = {ns: Cell(ns, FULL_N) for ns in (13, 14, 2)}
    human = Cell(None, FULL_N)

    images = Path("docs") / "images"
    images.mkdir(parents=True, exist_ok=True)
    fig_trend(images / "profile_visibility_trend.png")
    fig_slope(pair_cells, full_cells, human.slope,
              images / "profile_visibility_slope.png")
    fig_atoms(pair_cells, pid.dialogue_means(_load("pid", None, None, FULL_N)),
              images / "profile_visibility_atoms.png")


if __name__ == "__main__":
    main()
