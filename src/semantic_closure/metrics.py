"""Cosine similarity between adjacent turns of the same speaker.

- sim(X_t, X_{t+1})  t = 1..(Tx-1)
- sim(Y_t, Y_{t+1})  t = 1..(Ty-1)

Each similarity is reported in two variants:
- "raw":      cosine of the embeddings, only row-normalized
- "centered": the corpus mean vector subtracted and renormalized, correcting for
              anisotropy

Sentence embeddings share a common mean direction, which makes raw cosines
uniformly high. Both variants are always reported so the choice of primary
measure can be made against the two distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, embeddings
from .data import Dialogue


def normalize_rows(e: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    return e / np.clip(norms, 1e-12, None)


def center_and_normalize(e: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return normalize_rows(e - mean)


def adjacent_similarities(e: np.ndarray) -> np.ndarray:
    """Cosines between adjacent rows of a row-normalized (n, dim) array; returns (n-1,)."""
    return np.einsum("ij,ij->i", e[:-1], e[1:])


def dialogue_metric_rows(
    dialogue: Dialogue,
    emb: np.ndarray,
    corpus_mean: np.ndarray | None = None,
) -> list[dict]:
    """Adjacent-turn similarities of one dialogue, in long form.

    Columns: dialogue_id, role, speaker_id, t, variant, similarity. t indexes the
    pair, so similarity is the cosine between utterances t and t+1 of that role.
    """
    variants = {"raw": normalize_rows(emb)}
    if corpus_mean is not None:
        variants["centered"] = center_and_normalize(emb, corpus_mean)

    rows = []
    for variant, e in variants.items():
        for role, speaker_id, idx in (
            ("X", dialogue.speaker_x, slice(0, None, 2)),
            ("Y", dialogue.speaker_y, slice(1, None, 2)),
        ):
            sims = adjacent_similarities(e[idx])
            rows.extend(
                {
                    "dialogue_id": dialogue.dialogue_id,
                    "role": role,
                    "speaker_id": speaker_id,
                    "t": t,
                    "variant": variant,
                    "similarity": float(s),
                }
                for t, s in enumerate(sims, start=1)
            )
    return rows


def triple_decomposition(
    u: np.ndarray, v: np.ndarray, w: np.ndarray
) -> dict[str, np.ndarray]:
    """Decomposition of a triple (own_t, partner, own_{t+1}) in the context plane.

    u, v, w are row-normalized (n, dim). w is written as
        w = a*u + b*v_perp + r,  v_perp = (v - (u.v)u) normalized
    where a is self-continuation (the same quantity as the adjacent-turn cosine),
    b is the inflow from the partner that the speaker's own turn does not explain
    (the analogue of the NTIC conditional term), and r_norm is what lies outside
    the context plane. proj_norm = sqrt(a**2 + b**2) and orth_norm = r_norm.
    sc = partner_sim - b mirrors the form of NTIC = I(E;X') - I(E;X'|X).
    """
    uv = np.einsum("ij,ij->i", u, v)
    a = np.einsum("ij,ij->i", u, w)
    partner_sim = np.einsum("ij,ij->i", v, w)

    v_perp = v - uv[:, None] * u
    s = np.linalg.norm(v_perp, axis=1)
    # Where u and v are nearly parallel, set b=0; uv_sim identifies those rows.
    degenerate = s < 1e-6
    s_safe = np.where(degenerate, 1.0, s)
    b = np.einsum("ij,ij->i", w, v_perp / s_safe[:, None])
    b = np.where(degenerate, 0.0, b)

    r_norm = np.sqrt(np.clip(1.0 - a**2 - b**2, 0.0, None))
    return {
        "a": a,
        "b": b,
        "r_norm": r_norm,
        "proj_norm": np.sqrt(a**2 + b**2),
        "partner_sim": partner_sim,
        "sc": partner_sim - b,
        "uv_sim": uv,
    }


def dialogue_triple_rows(
    dialogue: Dialogue,
    emb: np.ndarray,
    corpus_mean: np.ndarray | None = None,
) -> list[dict]:
    """Triple decompositions of one dialogue, one row each.

    Triple t of role X is (X_t, Y_t, X_{t+1}), of role Y is (Y_t, X_{t+1}, Y_{t+1}):
    in both cases (own previous turn, partner's turn, own next turn).
    """
    variants = {"raw": normalize_rows(emb)}
    if corpus_mean is not None:
        variants["centered"] = center_and_normalize(emb, corpus_mean)

    rows = []
    for variant, e in variants.items():
        x, y = e[0::2], e[1::2]
        for role, speaker_id, own, partner in (
            ("X", dialogue.speaker_x, x, y[: len(x) - 1]),
            ("Y", dialogue.speaker_y, y, x[1 : len(y)]),
        ):
            n = min(len(own) - 1, len(partner))
            if n < 1:
                continue
            d = triple_decomposition(own[:n], partner[:n], own[1 : n + 1])
            for t in range(n):
                rows.append(
                    {
                        "dialogue_id": dialogue.dialogue_id,
                        "role": role,
                        "speaker_id": speaker_id,
                        "t": t + 1,
                        "variant": variant,
                        **{k: float(vals[t]) for k, vals in d.items()},
                    }
                )
    return rows


#: Sentinel for "center_corpus not given". None cannot serve, since it is the
#: valid value naming the human--human corpus.
_SELF = object()


def _centering_mean(
    dialogues: list[Dialogue],
    model: str,
    corpus: str | None,
    center_corpus: str | None | object,
) -> np.ndarray:
    """Pick the mean vector used for centering.

    By default each corpus is centered on its own mean. A derived corpus that is
    compared directly against its source, such as the replaced-speaker dialogues,
    passes center_corpus to share the source's mean: centering the two separately
    would absorb or manufacture the very difference being measured.
    """
    if center_corpus is _SELF:
        return embeddings.corpus_mean(
            [d.dialogue_id for d in dialogues], model=model, corpus=corpus
        )
    return embeddings.corpus_mean(None, model=model, corpus=center_corpus)


def compute_corpus_triples(
    dialogues: list[Dialogue],
    model: str = config.EMBEDDING_MODEL,
    corpus: str | None = None,
    center_corpus: str | None | object = _SELF,
) -> pd.DataFrame:
    """Triple decompositions for every cached dialogue of a corpus.

    Centering uses the corpus's own mean unless center_corpus names another one.
    """
    mean = _centering_mean(dialogues, model, corpus, center_corpus)
    rows = []
    for d in dialogues:
        emb = embeddings.load_embeddings(d.dialogue_id, model, corpus)
        rows.extend(dialogue_triple_rows(d, emb, corpus_mean=mean))
    return pd.DataFrame(rows)


def compute_corpus_metrics(
    dialogues: list[Dialogue],
    model: str = config.EMBEDDING_MODEL,
    corpus: str | None = None,
) -> pd.DataFrame:
    """Adjacent-turn similarities for every cached dialogue of a corpus."""
    mean = embeddings.corpus_mean(
        [d.dialogue_id for d in dialogues], model=model, corpus=corpus
    )
    rows = []
    for d in dialogues:
        emb = embeddings.load_embeddings(d.dialogue_id, model, corpus)
        rows.extend(dialogue_metric_rows(d, emb, corpus_mean=mean))
    return pd.DataFrame(rows)
