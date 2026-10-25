"""Embed and cache the utterances of the LLM--LLM (self-play) dialogues.

Each corpus gets its own subdirectory, at
data/embeddings/{model}/{corpus}/{id:05d}.npz, so it never mixes into the default
corpus when corpus_mean or a full scan runs.

--corpus llm_llm_oneshot holds the cells used in the paper; llm_llm holds the
earlier cells, generated under different conditions.

Run from the repository root:
    uv run python scripts/embed_llm_llm.py --corpus llm_llm_oneshot --dry-run
    uv run python scripts/embed_llm_llm.py --corpus llm_llm_oneshot

Needs OPENAI_API_KEY in .env. Cached dialogues are skipped.
"""

from __future__ import annotations

import argparse
from collections import Counter

from semantic_closure import config, embeddings
from semantic_closure.data import (
    iter_llm_llm_dialogues,
    iter_llm_llm_oneshot_dialogues,
)

LOADER = {
    "llm_llm": iter_llm_llm_dialogues,
    "llm_llm_oneshot": iter_llm_llm_oneshot_dialogues,
}

#: USD per million tokens, for the cost estimate only.
PRICE_PER_MTOK = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--corpus", default="llm_llm", choices=list(LOADER),
                        help="llm_llm_oneshot holds the cells used in the paper")
    parser.add_argument("--dry-run", action="store_true", help="estimate tokens and cost without calling the API")
    args = parser.parse_args()

    CORPUS = args.corpus
    dialogues = list(LOADER[CORPUS]())
    n_todo = sum(
        1 for d in dialogues if not embeddings.is_cached(d.dialogue_id, args.model, CORPUS)
    )
    by_system = Counter(d.system_id for d in dialogues)
    print(f"{len(dialogues)} dialogues, {n_todo} not yet cached; by system: {dict(by_system)}")

    if args.dry_run:
        n_chars = sum(
            len(u.text)
            for d in dialogues
            for u in d.utterances
            if not embeddings.is_cached(d.dialogue_id, args.model, CORPUS)
        )
        price = PRICE_PER_MTOK.get(args.model)
        cost = f"~${n_chars / 1e6 * price:.4f}" if price else "unknown price"
        print(f"estimate: {n_chars:,} characters ~ {n_chars:,} tokens, cost {cost}")
        return

    stats = embeddings.embed_dialogues(dialogues, model=args.model, corpus=CORPUS)
    print(f"done: embedded={stats['embedded']}, skipped={stats['skipped']}")
    print(f"cache: {embeddings.cache_dir(args.model, CORPUS)}")


if __name__ == "__main__":
    main()
