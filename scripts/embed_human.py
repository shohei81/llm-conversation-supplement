"""Embed and cache the utterances of the human--human corpus.

Run from the repository root:
    uv run python scripts/embed_human.py --limit 500            # first 500 dialogues
    uv run python scripts/embed_human.py                        # all dialogues
    uv run python scripts/embed_human.py --limit 500 --dry-run  # cost estimate only

Needs OPENAI_API_KEY in .env. Cached dialogues are skipped, so an interrupted run
continues from where it stopped when the same command is re-run.
"""

from __future__ import annotations

import argparse

from semantic_closure import config, embeddings
from semantic_closure.data import iter_dialogues

#: USD per million tokens, for the cost estimate only.
PRICE_PER_MTOK = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="use the first n dialogues")
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="estimate tokens and cost without calling the API")
    args = parser.parse_args()

    dialogues = list(iter_dialogues(limit=args.limit))
    n_todo = sum(1 for d in dialogues if not embeddings.is_cached(d.dialogue_id, args.model))
    print(f"{len(dialogues)} dialogues, {n_todo} not yet cached")

    if args.dry_run:
        n_chars = sum(len(u.text) for d in dialogues for u in d.utterances
                      if not embeddings.is_cached(d.dialogue_id, args.model))
        # One Japanese character is roughly one token; this is the upper estimate.
        price = PRICE_PER_MTOK.get(args.model)
        cost = f"~${n_chars / 1e6 * price:.3f}" if price else "unknown price"
        print(f"estimate: {n_chars:,} characters ~ {n_chars:,} tokens, cost {cost}")
        return

    stats = embeddings.embed_dialogues(dialogues, model=args.model)
    print(f"done: embedded={stats['embedded']}, skipped={stats['skipped']}")
    print(f"cache: {embeddings.cache_dir(args.model)}")


if __name__ == "__main__":
    main()
