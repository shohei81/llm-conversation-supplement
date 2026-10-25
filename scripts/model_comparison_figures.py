"""Figures for the model comparison. Figure 2 of the paper is the trend figure.

Draws four figures: mean novelty and redundancy against utterance index (bands are
the standard error across series), the novelty slopes with 95% CIs, the atom
composition as 100% stacked bars, and the distribution of utterance length.

--cells works as in model_comparison.py, and the filename prefix changes with it so
one cell set does not overwrite another.

Same prerequisites as model_comparison.py.

    uv run python scripts/model_comparison_figures.py --lang en

Writes {prefix}_trend.png, _slope.png, _atoms.png and _length.png into
outputs/figures/ with --lang en, otherwise into docs/images/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from semantic_closure import config, pid
from semantic_closure.data import iter_dialogues

from model_comparison import FIG_PREFIX, N_DIALOGUES, ROWS, CELLS
from profile_visibility import _load
from profile_visibility_figures import ATOM_STYLE
from generation_factors import series_stats

plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

#: Text inside the figures, in Japanese for the docs and English for the paper.
#: The atom subscripts follow the paper: U_self is U_X, the speaker's own previous
#: turn, and U_other is U_Y, the partner's turn.
LABELS = {
    "ja": {
        "human": "human×human",
        "human_all": "human×human(全対話)",
        "trend_title": "同じプロンプトで各モデルを回したときのターン位置に対する平均 novelty",
        "trend_x": "発話番号",
        "trend_y": "novelty N",
        "trend_y2": "redundancy R",
        "slope_title": "novelty の減衰(誤差棒 = 95%CI)",
        "slope_x": "会話内 novelty の傾き(10 発話あたりの OLS 係数)",
        "atoms_title": "PID atom 構成(対話平均のグループ平均)",
        "atoms_x": "構成比",
        "atoms_legend": ["R(redundancy)", "U_self(unique 自分側)",
                         "U_other(unique 相手側)", "S(synergy)", "N(novelty)"],
        "length_title": "同じ長さ指示に対する発話長",
        "length_x": "1発話の文字数(ひげは 1.5 IQR、外れ値は非表示。× は平均)",
        "length_mean": "平均 {:.1f}",
    },
    "en": {
        "human": "human-human",
        "human_all": "human-human (all dialogues)",
        "trend_title": "Mean novelty against turn position",
        "trend_x": "Utterance index",
        "trend_y": "novelty N",
        "trend_y2": "redundancy R",
        "slope_title": "Within-conversation decay of novelty (bars = 95% CI)",
        "slope_x": "OLS slope per 10 utterances",
        "atoms_title": "PID atom composition (group mean of per-dialogue means)",
        "atoms_x": "Share",
        "atoms_legend": ["R (redundancy)", "U_X (unique, own previous)",
                         "U_Y (unique, partner)", "S (synergy)", "N (novelty)"],
        "length_title": "Utterance length under the same length instruction",
        "length_x": "Characters per utterance",
        "length_mean": "mean {:.1f}",
    },
}


def select_lang() -> str:
    """--lang ja|en, default ja. en labels the figures in English and writes them
    into outputs/figures/."""
    import sys

    lang = "en" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "en" \
        else "ja"
    return lang


LANG = select_lang()
L = LABELS[LANG]

#: Figure sizes in inches. en matches the printed width of the paper: one column
#: is about 3.5in and a full spread about 7.16in, so text comes out at its nominal
#: point size and stays legible without shrinking. ja is larger, for the docs, and
#: its horizontal-bar figures grow with the number of models.
SIZES = {
    "ja": {"trend": (8.8, 7.2), "slope": (8.8, 1.1 + 0.6 * len(ROWS)),
           "atoms": (10.4, 1.0 + 0.6 * (len(ROWS) + 1)),
           "length": (8.8, 0.7 + 0.6 * (len(ROWS) + 1))},
    "en": {"trend": (3.5, 4.2), "slope": (3.5, 1.7), "atoms": (7.16, 2.3),
           "length": (3.5, 2.0)},
}


def suptitle(fig, key: str) -> float:
    """Title inside the figure. Omitted in en, where the caption does the same job.

    Returns the top edge to pass to tight_layout.
    """
    if LANG == "en":
        return 1.0
    fig.suptitle(L[key], fontsize=11.5)
    return {"trend_title": 0.94, "slope_title": 0.88,
            "atoms_title": 0.95, "length_title": 0.89}[key]

HUMAN_COLOR = "#555555"
#: model -> (color, line style, marker); the lines differ in style and marker too.
STYLE = {
    "claude-haiku-4-5": ("#eb6834", "-", "o"),
    "claude-sonnet-5": ("#c9a227", (0, (3, 1.5, 1, 1.5)), "D"),
    "claude-opus-5": ("#b5322f", (0, (7, 2, 1, 2)), "v"),
    "gpt-5-nano": ("#7a3fbf", (0, (5, 2)), "s"),
    "gpt-4o-mini": ("#1baf7a", (0, (1, 1.6)), "^"),
}


def name(model: str, release: str) -> str:
    """Model name for the figure. The release date is in the table, not here."""
    return model


#: Upper limit of the x axis. The generated dialogues are 20 utterances long, so
#: every group is compared over that range. The shortest human series has 13 turns,
#: i.e. at least 26 utterances, so every human series contributes at every index.
TREND_MAX_UTT = 20


def by_utterance(df: pd.DataFrame, col: str):
    """Mean across series and its standard error, per utterance index.

    The target of role X's triple (X_t, Y_t, X_{t+1}) is utterance 2t+1 and of role
    Y's (Y_t, X_{t+1}, Y_{t+1}) is 2t+2, so each index draws on one role only.
    """
    d = df.copy()
    d["u"] = np.where(d["role"] == "X", 2 * d["t"] + 1, 2 * d["t"] + 2)
    d = d[d["u"] <= TREND_MAX_UTT]
    piv = d.pivot_table(index=["dialogue_id", "role"], columns="u", values=col)
    se = piv.std(axis=0) / np.sqrt(piv.count(axis=0))
    return (piv.columns.to_numpy(), piv.mean(axis=0).to_numpy(), se.to_numpy())


def fig_trend(out: Path) -> None:
    """Mean novelty (top) and redundancy (bottom) against utterance index.

    Bands are +/-1 SE across series. The two panels share an x axis so that what
    novelty loses can be seen going into redundancy.
    """
    fig, (ax_n, ax_r) = plt.subplots(2, 1, figsize=SIZES[LANG]["trend"], sharex=True)
    cells = [(corpus, ns, model, release) for ns, model, release, corpus, _ in ROWS]
    for corpus, ns, model, release in cells:
        color, ls, marker = STYLE[model]
        df = _load("pid", corpus, ns, N_DIALOGUES)
        for ax, col in ((ax_n, "novelty"), (ax_r, "redundancy")):
            u, m, se = by_utterance(df, col)
            ax.fill_between(u, m - se, m + se, color=color, alpha=0.10, linewidth=0)
            ax.plot(u, m, color=color, linewidth=2.0, linestyle=ls, marker=marker,
                    markersize=3.2, markevery=2, label=name(model, release))
    # Draw the human band in the same style. Without it, a reader cannot tell a
    # small spread from a band that was simply left out.
    hdf = _load("pid", None, None, N_DIALOGUES)
    for ax, col in ((ax_n, "novelty"), (ax_r, "redundancy")):
        u, m, se = by_utterance(hdf, col)
        ax.fill_between(u, m - se, m + se, color=HUMAN_COLOR, alpha=0.10, linewidth=0)
        ax.plot(u, m, color=HUMAN_COLOR, linewidth=2.6, linestyle="--", label=L["human"])

    ax_n.set_ylabel(L["trend_y"], fontsize=9.5)
    ax_r.set_ylabel(L["trend_y2"], fontsize=9.5)
    ax_r.set_xlabel(L["trend_x"], fontsize=9.5)
    # The novelty band bottoms out at 0.492, hence 0.48. Redundancy starts at 0,
    # where zero is meaningful, and the top holds the band's 0.268.
    ax_n.set_ylim(0.48, 0.93)
    ax_r.set_ylim(0.0, 0.28)
    # Legend on the top panel only, in two columns since six entries have to fit a
    # single column width. Cropping the y axis to the bands fills the lower left,
    # leaving the upper right free.
    if LANG == "en":
        ax_n.legend(fontsize=7, frameon=False, loc="upper right", ncol=2, columnspacing=0.8)
    else:
        ax_n.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=2, columnspacing=1.2)
    for ax in (ax_n, ax_r):
        ax.grid(color="#eeeef1", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    top = suptitle(fig, "trend_title")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def fig_slope(out: Path) -> None:
    h_slope = float(
        series_stats(_load("novelty_adjacent", None, None, N_DIALOGUES))["slope"].mean())
    fig, ax = plt.subplots(figsize=SIZES[LANG]["slope"])
    ys = range(len(ROWS))[::-1]
    for y, (ns, model, release, corpus, _) in zip(ys, ROWS):
        s = series_stats(_load("novelty_adjacent", corpus, ns, N_DIALOGUES))["slope"]
        ci = 1.96 * s.std(ddof=1) / np.sqrt(len(s))
        ax.errorbar(s.mean(), y, xerr=ci, fmt="o", markersize=7, color=STYLE[model][0],
                    capsize=4, linewidth=1.8, markeredgecolor="white")
    ax.axvline(h_slope, color=HUMAN_COLOR, linewidth=1.6, linestyle="--",
               label=f'{L["human"]} ({h_slope:+.3f})')
    ax.axvline(0, color="#cccccc", linewidth=0.9)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([name(m, r) for _, m, r, _, _ in ROWS], fontsize=9)
    ax.set_xlabel(L["slope_x"], fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.grid(axis="x", color="#eeeef1", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    top = suptitle(fig, "slope_title")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def fig_atoms(out: Path) -> None:
    rows = [(name(m, r), pid.dialogue_means(_load("pid", corpus, ns, N_DIALOGUES)))
            for ns, m, r, corpus, _ in ROWS]
    rows.append((L["human"], pid.dialogue_means(_load("pid", None, None, N_DIALOGUES))))
    fig, ax = plt.subplots(figsize=SIZES[LANG]["atoms"])
    ys = range(len(rows))[::-1]
    for y, (label, df) in zip(ys, rows):
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
    ax.set_yticks(list(ys))
    ax.set_yticklabels([n for n, _ in rows], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel(L["atoms_x"], fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for _, _, c in ATOM_STYLE]
    ax.legend(handles, L["atoms_legend"], loc="upper center",
              bbox_to_anchor=(0.5, -0.24), ncol=5, fontsize=8, frameon=False)
    top = suptitle(fig, "atoms_title")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def fig_length(out: Path) -> None:
    rows = []
    for ns, model, release, _, loader in ROWS:
        ids = [ns * 100000 + k for k in range(1, N_DIALOGUES + 1)]
        lens = [len(u.text) for d in loader(dialogue_ids=ids) for u in d.utterances]
        rows.append((name(model, release), lens, STYLE[model][0]))
    human = [len(u.text) for d in iter_dialogues() for u in d.utterances]
    rows.append((L["human_all"], human, HUMAN_COLOR))

    fig, ax = plt.subplots(figsize=SIZES[LANG]["length"])
    ys = list(range(len(rows)))[::-1]
    bp = ax.boxplot([r[1] for r in rows], positions=ys, orientation="horizontal",
                    widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 1.6})
    for patch, (_, _, color) in zip(bp["boxes"], rows):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set_color("#888888")
    for y, (_, lens, _) in zip(ys, rows):
        ax.scatter([np.mean(lens)], [y], marker="X", s=70, color="white",
                   edgecolors="#333333", linewidths=1.0, zorder=5)
        ax.text(np.mean(lens), y + 0.34, L["length_mean"].format(np.mean(lens)), ha="center",
                va="bottom", fontsize=8, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 80)
    ax.set_xlabel(L["length_x"], fontsize=9.5)
    ax.grid(axis="x", color="#eeeef1", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    top = suptitle(fig, "length_title")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    images = config.OUTPUT_DIR / "figures" if LANG == "en" else Path("docs") / "images"
    pre = FIG_PREFIX[CELLS]
    images.mkdir(parents=True, exist_ok=True)
    fig_trend(images / f"{pre}_trend.png")
    fig_slope(images / f"{pre}_slope.png")
    fig_atoms(images / f"{pre}_atoms.png")
    fig_length(images / f"{pre}_length.png")


if __name__ == "__main__":
    main()
