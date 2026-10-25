"""Figure 1: atom composition of the five models and human--human dialogue.

Horizontal 100% stacked bars, ordered from the bottom as human--human,
claude-opus-5, claude-sonnet-5, claude-haiku-4-5, gpt-5-nano, gpt-4o-mini. Values
are per (dialogue, role) means averaged within a cell, the same aggregation as the
table from model_comparison.py. Colors match the atom figure of
model_comparison_figures.py.

Requires compute_pid.py to have been run for human and llm_llm_oneshot.

    uv run python scripts/decomposition_profile_figure.py

Writes outputs/figures/decomposition_profile_main.png.
"""

from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semantic_closure import config, pid

ATOMS = ["redundancy", "unique_self", "unique_other", "synergy", "novelty"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d9d9d9"]
LEGEND = ["R (redundancy)", "U_X (unique, own previous)", "U_Y (unique, partner)",
          "S (synergy)", "N (novelty)"]
#: Top to bottom, so claude-opus-5 sits just above human--human at the bottom.
CELLS = [("gpt-4o-mini", 21), ("gpt-5-nano", 22), ("claude-haiku-4-5", 20),
         ("claude-sonnet-5", 24), ("claude-opus-5", 25)]
N_DIALOGUES = 150
#: Bars narrower than this carry no number, only the color.
MIN_LABEL = 0.09


def cell_means(df: pd.DataFrame, ns: int | None = None) -> pd.Series:
    if ns is not None:
        ids = [ns * 100000 + k for k in range(1, N_DIALOGUES + 1)]
        df = df[df.dialogue_id.isin(ids)]
    return pid.dialogue_means(df)[ATOMS].mean()


def main() -> None:
    metrics = config.OUTPUT_DIR / "metrics"
    llm = pd.read_parquet(metrics / f"pid__{config.EMBEDDING_MODEL}__llm_llm_oneshot.parquet")
    human = pd.read_parquet(metrics / f"pid__{config.EMBEDDING_MODEL}.parquet")
    rows = [(label, cell_means(llm, ns)) for label, ns in CELLS]
    rows.append(("human-human", cell_means(human)))

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ys = list(range(len(rows)))[::-1]
    for y, (_, v) in zip(ys, rows):
        left = 0.0
        for col, color in zip(ATOMS, COLORS):
            w = float(v[col])
            ax.barh(y, w, left=left, height=0.64, color=color, linewidth=0)
            if w >= MIN_LABEL:
                ax.text(left + w / 2, y, f"{w:.2f}", ha="center", va="center",
                        fontsize=5.5, color="#333333")
            left += w
    ax.axhline(0.5, color="#bbbbbb", linewidth=0.8, linestyle="--")  # models vs human
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _ in rows], fontsize=6.5)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Share", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=5.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in COLORS]
    ax.legend(handles, LEGEND, loc="upper center", bbox_to_anchor=(0.46, -0.20),
              ncol=2, fontsize=5.0, frameon=False, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    out = config.OUTPUT_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "decomposition_profile_main.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / 'decomposition_profile_main.png'}")


if __name__ == "__main__":
    main()
