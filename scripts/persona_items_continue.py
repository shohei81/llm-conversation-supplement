"""Figure 3 of the paper: the continuation cells, at 10, 3 and 0 persona items.

Each continuation resumes the saved first 20 utterances of the leading dialogues of
its source cell and generates on to 60 utterances, so the first 20 match the source
exactly. ns32 continues ns25 (10 items), ns34 continues ns30 (3 items) and ns33
continues ns31 (no persona).

Requires compute_pid.py to have been run for llm_llm_oneshot.
    uv run python scripts/persona_items_continue.py
    uv run python scripts/persona_items_continue.py --lang en

Prints four tables and writes persona_items_continue.png, into outputs/figures/
with --lang en and otherwise into docs/images/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semantic_closure import config

plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

#: condition -> (namespace of the continuation, namespace of the source cell)
CONDS = {"10 items": (32, 25), "3 items": (34, 30), "no persona": (33, 31)}
STYLE = {"10 items": ("#6a3d9a", (0, (1, 1.6)), "D"), "3 items": ("#3b7dd8", (0, (5, 2)), "s"),
         "no persona": ("#eb6834", "-", "o")}
TURNS = 60
#: Closing formulas. A run of three or more marks the start of a closing sequence.
CLOSING = re.compile(
    "おやすみ|さようなら|さよなら|失礼します|失礼しま|またお話|また話|またね|また明日|"
    "ありがとうございました|良い一日|よい一日|良い夢|バイバイ|お疲れ様でした|それではまた|では、また|"
    "またよろしく|お会いしましょう|またお会い"
)
PAT_FOREIGN = re.compile(r"[ᄀ-ᇿ가-힯Ѐ-ӿ]|[A-Za-z]{4,}|<[^>]+>")
ONESHOT = config.llm_llm_generated_dir() / "oneshot"
#: --lang ja|en, default ja. en labels the figure in English and writes it into
#: outputs/figures/.
LANG = "en" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "en" else "ja"
IMAGES = config.OUTPUT_DIR / "figures" if LANG == "en" else Path("docs/images")
#: Text inside the figure; the legend keys are the condition names of CONDS.
LABELS = {
    "ja": {"10 items": "10 項目の続き", "3 items": "3 項目の続き", "no persona": "0 項目の続き",
           "divider": "← 元の 20 発話 | 続き →",
           "xlabel": "発話番号(実ターン t の発話位置。役割 X・Y をまとめた平均)",
           "ylabel": "novelty N(対話平均、帯は ± SE)",
           "ylabel_r": "redundancy R(対話平均、帯は ± SE)",
           "title": "20 発話の続きを 60 発話まで生成したときの novelty と redundancy"},
    "en": {"10 items": "10 persona items", "3 items": "3 persona items", "no persona": "no persona",
           "divider": "original 20 | continuation",
           "xlabel": "Utterance index", "ylabel": "Novelty $N$",
           "ylabel_r": "Redundancy $R$", "title": None},
}[LANG]


def load(ns: int) -> list[dict]:
    return json.loads((ONESHOT / f"claude-opus-5__oneshot__ns{ns}.json").read_text())


def closing_start(texts: list[str]) -> int | None:
    m = [bool(CLOSING.search(t)) for t in texts]
    for i in range(len(m) - 2):
        if m[i] and m[i + 1] and m[i + 2]:
            return i
    return None


def main() -> None:
    pid = pd.read_parquet(config.OUTPUT_DIR / "metrics" / f"pid__{config.EMBEDDING_MODEL}__llm_llm_oneshot.parquet")
    data = {name: load(ns) for name, (ns, _) in CONDS.items()}
    starts = {}
    print("== First 20 utterances against the source, and closing sequences")
    print("| condition | dialogues | first 20 match source | entered a closing | closing starts at |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, (ns, src_ns) in CONDS.items():
        src = {d["dialogue_id"] % 100000: d for d in load(src_ns)}
        ok = sum([u["text"] for u in d["utterances"][:20]] == [u["text"] for u in src[d["dialogue_id"] % 100000]["utterances"][:20]]
                 for d in data[name])
        starts[name] = {d["dialogue_id"]: closing_start([u["text"] for u in d["utterances"]]) for d in data[name]}
        s = sorted(v for v in starts[name].values() if v is not None)
        print(f"| {name} | {len(data[name])} | {ok} | {len(s)} | {', '.join(map(str, s)) if s else '—'} |")
    print()

    bins = [(1, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29)]
    print("== N / R / s_xz per block of turns (mean over dialogues). "
          "Utterances 2-19 are the source, 20-59 the continuation")
    print("| condition | " + " | ".join(f"utt {2*a}-{2*b+1}" for a, b in bins) + " |")
    print("| --- | " + " | ".join("---:" for _ in bins) + " |")
    for name, (ns, _) in CONDS.items():
        q = pid[pid.dialogue_id // 100000 == ns]
        print(f"| {name} | " + " | ".join(
            f"{q[(q.t >= a) & (q.t <= b)].novelty.mean():.3f} / {q[(q.t >= a) & (q.t <= b)].redundancy.mean():.3f} / "
            f"{q[(q.t >= a) & (q.t <= b)].s_xz.mean():.3f}" for a, b in bins) + " |")
    print()

    print("== N per dialogue over three blocks, and where a closing starts")
    print("| condition | dialogue | utt 2-19 | 20-39 | 40-59 | closing starts at |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for name in CONDS:
        for d in data[name]:
            q = pid[pid.dialogue_id == d["dialogue_id"]]
            s = starts[name][d["dialogue_id"]]
            print(f"| {name} | {d['dialogue_id']} | {q[q.t <= 9].novelty.mean():.3f} | "
                  f"{q[(q.t >= 10) & (q.t <= 19)].novelty.mean():.3f} | {q[q.t >= 20].novelty.mean():.3f} | {s if s is not None else '—'} |")
    print()

    print("== Slope of N over the continuation (utterances 20-59), per 10 utterances, "
          "averaged over series with a 95% CI, and generation quality")
    print("| condition | slope | series | utterances | mean length | tags or foreign script | speaker labels |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, (ns, _) in CONDS.items():
        q = pid[(pid.dialogue_id // 100000 == ns) & (pid.t >= 10)]
        # Per 10 utterances: u = 2t + c, so dN/du = (dN/dt)/2, and ten gives x5.
        sl = np.array([5.0 * np.polyfit(g.sort_values("t")["t"].to_numpy(float), g.sort_values("t").novelty, 1)[0]
                       for _, g in q.groupby(["dialogue_id", "role"])])
        utts = [u["text"] for d in data[name] for u in d["utterances"][20:]]
        print(f"| {name} | {sl.mean():+.3f} ± {1.96 * sl.std(ddof=1) / np.sqrt(len(sl)):.3f} | {len(sl)} | {len(utts)} | "
              f"{np.mean([len(t) for t in utts]):.1f} | {sum(bool(PAT_FOREIGN.search(t)) for t in utts)} | {sum('話者' in t for t in utts)} |")

    IMAGES.mkdir(parents=True, exist_ok=True)
    fig, (ax_n, ax_r) = plt.subplots(
        2, 1, figsize=(7.2, 6.6) if LANG == "ja" else (3.5, 4.2), sharex=True)
    for name, (ns, _) in CONDS.items():
        color, ls, marker = STYLE[name]
        for ax, col in ((ax_n, "novelty"), (ax_r, "redundancy")):
            q = pid[pid.dialogue_id // 100000 == ns].groupby("t")[col].agg(["mean", "std", "count"])
            # Bands are +/-1 SE of the mean. The figure compares how the condition
            # means move, so it shows the precision of the mean, not the spread (SD).
            se = q["std"] / np.sqrt(q["count"])
            ax.fill_between(q.index * 2 + 1, q["mean"] - se, q["mean"] + se, color=color, alpha=0.12, linewidth=0)
            ax.plot(q.index * 2 + 1, q["mean"], color=color, linestyle=ls, marker=marker, markersize=3.5, linewidth=1.6, label=LABELS[name])
    # Boundary between the source and the continuation. In en the caption says it.
    if LANG == "ja":
        for ax in (ax_n, ax_r):
            ax.axvline(19.5, color="#999999", linewidth=0.9, linestyle=":")
        ax_n.text(19.8, 0.97, LABELS["divider"], fontsize=8.5, color="#555555", va="top")
    ax_r.set_xlabel(LABELS["xlabel"], fontsize=9.5 if LANG == "ja" else 8)
    ax_n.set_ylabel(LABELS["ylabel"], fontsize=9.5 if LANG == "ja" else 8)
    ax_r.set_ylabel(LABELS["ylabel_r"], fontsize=9.5 if LANG == "ja" else 8)
    # The band bottoms out at 0.479, for no persona. The full range would be mostly
    # empty and hide the trend.
    ax_n.set_ylim(0.4, 1.0)
    # Redundancy starts at 0, where zero is meaningful; the top holds the band's 0.342.
    ax_r.set_ylim(0.0, 0.36)
    # Legend on the top panel only; both panels show the same series.
    ax_n.legend(fontsize=8.5 if LANG == "ja" else 7, frameon=False, loc="lower left")
    for ax in (ax_n, ax_r):
        if LANG == "en":
            ax.tick_params(labelsize=7)
        ax.grid(color="#eeeef1", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    if LABELS["title"]:
        fig.suptitle(LABELS["title"], fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    fig.savefig(IMAGES / "persona_items_continue.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote the figure into {IMAGES}")


if __name__ == "__main__":
    main()
