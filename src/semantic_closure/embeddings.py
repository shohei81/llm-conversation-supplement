"""OpenAI embeddings, cached per dialogue.

Cache layout:
- one .npz per dialogue at data/embeddings/{model}/{dialogue_id:05d}.npz
  (arrays: embeddings float32 (n_utt, dim), utterance_ids int64)
- the model name is in the path, so switching models cannot collide
- existence is checked per file, so an interrupted run resumes on re-execution
- requests batch utterances across dialogues (the API takes at most 2048 inputs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from tqdm import tqdm

from . import config
from .data import Dialogue


class EmbeddingClient(Protocol):
    """The part of the OpenAI client we use, so tests can substitute a fake."""

    def embed(self, texts: list[str], model: str) -> np.ndarray: ...


class OpenAIEmbeddingClient:
    def __init__(self) -> None:
        import os

        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Run cp .env.example .env and fill in the API key."
            )
        # The 600s default hangs for a long time on a stalled connection. A short
        # timeout with more retries recovers faster, and small batches make the
        # resend cheap.
        self._client = OpenAI(timeout=60.0, max_retries=5)

    def embed(self, texts: list[str], model: str) -> np.ndarray:
        resp = self._client.embeddings.create(model=model, input=texts)
        # The API preserves input order; reorder by index to be sure.
        data = sorted(resp.data, key=lambda d: d.index)
        return np.asarray([d.embedding for d in data], dtype=np.float32)


def cache_dir(model: str = config.EMBEDDING_MODEL, corpus: str | None = None) -> Path:
    """Directory of the embedding cache.

    A corpus gets its own subdirectory under the model (e.g. "llm" for human--LLM).
    The *.npz globs in cached_dialogue_ids and corpus_mean are non-recursive, so a
    subdirectory keeps other corpora out of a full scan of the default one.
    """
    base = config.EMBEDDING_CACHE_DIR / model
    return base if corpus is None else base / corpus


def cache_path(
    dialogue_id: int, model: str = config.EMBEDDING_MODEL, corpus: str | None = None
) -> Path:
    return cache_dir(model, corpus) / f"{dialogue_id:05d}.npz"


def is_cached(
    dialogue_id: int, model: str = config.EMBEDDING_MODEL, corpus: str | None = None
) -> bool:
    return cache_path(dialogue_id, model, corpus).exists()


def load_embeddings(
    dialogue_id: int, model: str = config.EMBEDDING_MODEL, corpus: str | None = None
) -> np.ndarray:
    """Cached embeddings as (n_utt, dim) float32, in utterance order."""
    with np.load(cache_path(dialogue_id, model, corpus)) as z:
        return z["embeddings"]


def _save(
    dialogue: Dialogue, emb: np.ndarray, model: str, corpus: str | None = None
) -> None:
    path = cache_path(dialogue.dialogue_id, model, corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends .npz unless the name ends in it, hence .tmp.npz.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        embeddings=emb,
        utterance_ids=np.asarray(
            [u.utterance_id for u in dialogue.utterances], dtype=np.int64
        ),
    )
    tmp.rename(path)  # an interrupted write leaves no broken cache entry


def embed_dialogues(
    dialogues: Iterable[Dialogue],
    client: EmbeddingClient | None = None,
    model: str = config.EMBEDDING_MODEL,
    dialogues_per_request: int = 10,
    corpus: str | None = None,
) -> dict[str, int]:
    """Embed and cache a sequence of dialogues, skipping those already cached.

    dialogues_per_request=10 is about 300 utterances per request, against the API
    limit of 2048. The default is small because larger batches produced responses
    of tens of megabytes and stalled connections. Returns {"embedded": n, "skipped": n}.
    """
    all_dialogues = list(dialogues)
    todo = [d for d in all_dialogues if not is_cached(d.dialogue_id, model, corpus)]
    n_skipped = len(all_dialogues) - len(todo)

    batch: list[Dialogue] = []
    n_embedded = 0

    def flush() -> None:
        nonlocal n_embedded, client
        if not batch:
            return
        if client is None:
            client = OpenAIEmbeddingClient()
        texts = [t for d in batch for t in d.texts()]
        emb = client.embed(texts, model)
        if emb.shape[0] != len(texts):
            raise RuntimeError(
                f"embedding rows do not match the input: {emb.shape[0]} != {len(texts)}"
            )
        offset = 0
        for d in batch:
            n = len(d.utterances)
            _save(d, emb[offset : offset + n], model, corpus)
            offset += n
        n_embedded += len(batch)
        batch.clear()

    for d in tqdm(todo, desc=f"embedding ({model})"):
        batch.append(d)
        if len(batch) >= dialogues_per_request:
            flush()
    flush()

    return {"embedded": n_embedded, "skipped": n_skipped}


def cached_dialogue_ids(
    model: str = config.EMBEDDING_MODEL, corpus: str | None = None
) -> list[int]:
    d = cache_dir(model, corpus)
    if not d.is_dir():
        return []
    # Ignore *.tmp.npz left by an interrupted run.
    return sorted(int(p.stem) for p in d.glob("*.npz") if p.stem.isdigit())


def corpus_mean(
    dialogue_ids: list[int] | None = None,
    model: str = config.EMBEDDING_MODEL,
    corpus: str | None = None,
) -> np.ndarray:
    """Mean embedding over all cached utterances, for the anisotropy correction."""
    ids = (
        dialogue_ids
        if dialogue_ids is not None
        else cached_dialogue_ids(model, corpus)
    )
    if not ids:
        raise RuntimeError("No cached embeddings. Run embed_human.py first.")
    total = None
    n = 0
    for i in ids:
        e = load_embeddings(i, model, corpus).astype(np.float64)
        total = e.sum(axis=0) if total is None else total + e.sum(axis=0)
        n += e.shape[0]
    return (total / n).astype(np.float32)
