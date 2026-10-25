"""Compute the geometric PID over every cached dialogue of a corpus.

For each triple (own_t, partner's turn, own_{t+1}) it writes the coverages
Q_X, Q_Y, Q_XY and the atoms redundancy, unique_self, unique_other, synergy and
novelty, plus the balance omega = R - S. Embeddings are used raw, without centering.

    uv run python scripts/compute_pid.py                            # human--human
    uv run python scripts/compute_pid.py --corpus llm_llm_oneshot   # LLM--LLM
    uv run python scripts/compute_pid.py --gaussian                 # add f(Q) in nats
    uv run python scripts/compute_pid.py --variant shuffled_corpus  # surrogate series

Writes outputs/metrics/pid__{model}{__corpus}{__variant}.parquet.
"""

from __future__ import annotations

import argparse

from semantic_closure import config, embeddings, pid
from semantic_closure.data import (
    iter_counterfactual_dialogues,
    iter_dialogues,
    iter_llm_dialogues,
    iter_llm_llm_dialogues,
    iter_llm_llm_oneshot_dialogues,
)

#: corpus -> (dialogue loader, the script that builds its embeddings)
_LOADERS = {
    None: (iter_dialogues, "embed_human.py"),
    "llm": (iter_llm_dialogues, "embed_llm.py"),
    "llm_llm": (iter_llm_llm_dialogues, "embed_llm_llm.py"),
    "llm_llm_oneshot": (iter_llm_llm_oneshot_dialogues, "embed_llm_llm.py --corpus llm_llm_oneshot"),
    "counterfactual": (iter_counterfactual_dialogues, "embed_counterfactual.py"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument(
        "--corpus",
        default=None,
        choices=[None, "llm", "llm_llm", "llm_llm_oneshot", "counterfactual"],
        help="None: human--human / 'llm': human--LLM / 'llm_llm', "
        "'llm_llm_oneshot': LLM--LLM, the latter being the cells used in the paper / "
        "'counterfactual': one human speaker replaced",
    )
    parser.add_argument(
        "--gaussian",
        action="store_true",
        help="add the Gaussian-bridge columns i_x, i_y, i_xy = -1/2 log(1-Q), in nats",
    )
    parser.add_argument(
        "--variant",
        default="real",
        choices=["real", "shuffled_corpus", "shuffled_all"],
        help="compute over a surrogate series drawn from other dialogues instead: "
        "'shuffled_corpus' draws from the same corpus, 'shuffled_all' from a pool "
        "of all corpora",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the surrogate draw")
    args = parser.parse_args()

    loader, embed_script = _LOADERS[args.corpus]
    ids = embeddings.cached_dialogue_ids(args.model, args.corpus)
    if not ids:
        raise SystemExit(
            f"No cached embeddings. Run {embed_script} first."
        )
    dialogues = list(loader(dialogue_ids=ids))
    print(f"{len(dialogues)} dialogues (corpus={args.corpus or 'human'}, variant={args.variant})")

    df = pid.compute_corpus_pid(
        dialogues, model=args.model, corpus=args.corpus,
        variant=args.variant, seed=args.seed,
    )
    if args.gaussian:
        df["i_x"], df["i_y"], df["i_xy"] = pid.gaussian_bridge(
            df["s_xz"].to_numpy(), df["s_yz"].to_numpy(), df["s_xy"].to_numpy()
        )

    suffix = f"__{args.corpus}" if args.corpus else ""
    if args.variant != "real":
        suffix += f"__{args.variant}"
    out = config.OUTPUT_DIR / "metrics" / f"pid__{args.model}{suffix}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows)")

    resid = (df[pid.ATOMS].sum(axis=1) - 1.0).abs().max()
    print(f"largest residual of 1 = R+U_X+U_Y+S+N: {resid:.2e}")

    print("\n=== per turn ===")
    cols = ["q_x", "q_y", "q_xy", *pid.ATOMS, "omega"]
    print(df[cols].describe().loc[["mean", "std", "50%"]].round(4).to_string())

    print("\n=== per (dialogue, role) ===")
    means = pid.dialogue_means(df)
    print(means[cols].describe().loc[["mean", "std", "50%"]].round(4).to_string())


if __name__ == "__main__":
    main()
