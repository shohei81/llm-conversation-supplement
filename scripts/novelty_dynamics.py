"""Cosines between adjacent novelty vectors, over every cached dialogue.

The residuals of the triple decomposition are treated as a series like the
utterance embeddings (see src/semantic_closure/novelty.py):
- cos_next_self:  cosine with the same speaker's next novelty vector
- cos_next_other: cosine with the next novelty vector in time, the partner's
- novelty, novelty_next: the novelty ||r||**2 of each vector

Alongside the real dialogues it computes a surrogate control built from utterances
of other dialogues, drawn either from the same corpus (variant="shuffled_corpus")
or from a pool of all corpora (variant="shuffled_all"). The cosines carry a
mechanical negative bias, so the signal is the difference between real and control,
not the absolute value.

    uv run python scripts/novelty_dynamics.py                          # human--human
    uv run python scripts/novelty_dynamics.py --corpus llm_llm_oneshot # LLM--LLM

Writes outputs/metrics/novelty_adjacent__{model}{__corpus}.parquet.
"""

from __future__ import annotations

import argparse

from semantic_closure import config, embeddings, novelty
from semantic_closure.data import (
    iter_dialogues,
    iter_llm_dialogues,
    iter_llm_llm_dialogues,
    iter_llm_llm_oneshot_dialogues,
)
from semantic_closure.stats import cohens_d

_LOADERS = {
    None: (iter_dialogues, "embed_human.py"),
    "llm": (iter_llm_dialogues, "embed_llm.py"),
    "llm_llm": (iter_llm_llm_dialogues, "embed_llm_llm.py"),
    "llm_llm_oneshot": (iter_llm_llm_oneshot_dialogues, "embed_llm_llm.py --corpus llm_llm_oneshot"),
}

COS_COLS = ["cos_next_self", "cos_next_other", "cos_uptake"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--corpus", default=None,
                        choices=[None, "llm", "llm_llm", "llm_llm_oneshot"])
    parser.add_argument("--variants", default="real,shuffled_corpus,shuffled_all",
                        help="comma-separated variants to compute; use real alone "
                             "where no control is needed")
    parser.add_argument("--seed", type=int, default=0, help="seed of the control draw")
    args = parser.parse_args()

    loader, embed_script = _LOADERS[args.corpus]
    ids = embeddings.cached_dialogue_ids(args.model, args.corpus)
    if not ids:
        raise SystemExit(
            f"No cached embeddings. Run {embed_script} first."
        )
    dialogues = list(loader(dialogue_ids=ids))
    print(f"{len(dialogues)} dialogues (corpus={args.corpus or 'human'}, "
          f"{args.variants})")

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    df = novelty.compute_corpus_novelty_cos(
        dialogues, model=args.model, corpus=args.corpus, variants=variants, seed=args.seed
    )

    suffix = f"__{args.corpus}" if args.corpus else ""
    out = config.OUTPUT_DIR / "metrics" / f"novelty_adjacent__{args.model}{suffix}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows)")

    print("\n=== mean per variant, per turn, NaN dropped ===")
    print(df.groupby("variant")[COS_COLS + ["novelty"]].mean().round(4).to_string())

    print("\n=== mean per variant, per (dialogue, role) ===")
    dm = df.groupby(["variant", "dialogue_id", "role"], as_index=False)[COS_COLS].mean()
    print(dm.groupby("variant")[COS_COLS].mean().round(4).to_string())

    if "shuffled_corpus" in variants:
        real, shuf = df[df["variant"] == "real"], df[df["variant"] == "shuffled_corpus"]
        print("\n=== real vs shuffled_corpus, per turn ===")
        for c in COS_COLS:
            a, b = real[c].dropna().to_numpy(), shuf[c].dropna().to_numpy()
            print(f"  {c:>15}: real {a.mean():+.4f} / shuffled_corpus {b.mean():+.4f} / "
                  f"d={cohens_d(a, b):+.3f}")


if __name__ == "__main__":
    main()
