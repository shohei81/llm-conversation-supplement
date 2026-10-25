"""Geometric PID: projection coverage of a triple (x, y, z).

From unit-norm embeddings x = e(X_t), y = e(Y_t), z = e(X_{t+1}), the coverages
Q_X, Q_Y, Q_XY are determined by the three pairwise cosines alone, and yield the
atoms (R, U_self, U_other, S, N) with 1 = R + U_self + U_other + S + N exactly at
every turn.

The unique atoms are named self (source X, the speaker's own previous turn) and
other (source Y, the partner's turn) rather than X and Y, to keep them from being
read as speaker labels.

Embeddings are used raw, without centering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, embeddings
from .data import Dialogue
from .metrics import normalize_rows

#: Below this, 1 - s_xy**2 counts as parallel sources (a one-dimensional span)
#: and Q_XY falls back to max(Q_X, Q_Y).
EPS_PARALLEL = 1e-10

#: The five atoms that sum to one.
ATOMS = ["redundancy", "unique_self", "unique_other", "synergy", "novelty"]


def triple_pid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
    """Coverages and atoms per triple. x, y, z are row-normalized (n, dim).

    x is the own-side source, y the partner-side source, z the target.
    """
    s_xz = np.einsum("ij,ij->i", x, z)
    s_yz = np.einsum("ij,ij->i", y, z)
    s_xy = np.einsum("ij,ij->i", x, y)

    q_x = s_xz**2
    q_y = s_yz**2
    q_max = np.maximum(q_x, q_y)

    denom = 1.0 - s_xy**2
    parallel = denom < EPS_PARALLEL
    denom_safe = np.where(parallel, 1.0, denom)
    q_xy = (q_x + q_y - 2.0 * s_xy * s_xz * s_yz) / denom_safe
    q_xy = np.where(parallel, q_max, q_xy)
    # Clamp so rounding cannot break max(Q_X, Q_Y) <= Q_XY <= 1.
    q_xy = np.clip(q_xy, q_max, 1.0)

    redundancy = np.minimum(q_x, q_y)
    return {
        "s_xz": s_xz,
        "s_yz": s_yz,
        "s_xy": s_xy,
        "q_x": q_x,
        "q_y": q_y,
        "q_xy": q_xy,
        "q_union": q_max,
        "delta_y_given_x": q_xy - q_x,
        "delta_x_given_y": q_xy - q_y,
        "redundancy": redundancy,
        "unique_self": q_x - redundancy,
        "unique_other": q_y - redundancy,
        "synergy": q_xy - q_max,
        "novelty": 1.0 - q_xy,
        "omega": q_x + q_y - q_xy,
        "rsi": q_xy - q_x - q_y,
    }


def gaussian_mi(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """f(q) = -1/2 log(1-q) in nats. Clipped at 1-eps, where f diverges."""
    return -0.5 * np.log1p(-np.clip(q, 0.0, 1.0 - eps))


def gaussian_bridge(
    s_xz: np.ndarray, s_yz: np.ndarray, s_xy: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian MI in nats, with the Gram matrix regularized as K = (1-eps)K + eps*I.

    That leaves the diagonal at one and scales only the correlations, so the three
    cosines are scaled by (1-eps) before Q and f. Returns (f(Q_X), f(Q_Y), f(Q_XY)).
    """
    a, b, c = (1 - eps) * np.asarray(s_xz), (1 - eps) * np.asarray(s_yz), (1 - eps) * np.asarray(s_xy)
    q_x, q_y = a**2, b**2
    q_xy = (q_x + q_y - 2.0 * c * a * b) / (1.0 - c**2)
    q_xy = np.clip(q_xy, np.maximum(q_x, q_y), None)
    return gaussian_mi(q_x), gaussian_mi(q_y), gaussian_mi(q_xy)


def dialogue_pid_rows(dialogue: Dialogue, emb: np.ndarray) -> list[dict]:
    """One row per triple of one dialogue.

    Triple t of role X is (X_t, Y_t, X_{t+1}), of role Y is (Y_t, X_{t+1}, Y_{t+1}):
    in both cases (own previous turn, partner's turn, own next turn).
    """
    # The cache is float32; compute in float64 so the identity residual stays at
    # rounding level.
    e = normalize_rows(emb.astype(np.float64))
    x, y = e[0::2], e[1::2]

    rows = []
    for role, speaker_id, own, partner in (
        ("X", dialogue.speaker_x, x, y[: len(x) - 1]),
        ("Y", dialogue.speaker_y, y, x[1 : len(y)]),
    ):
        n = min(len(own) - 1, len(partner))
        if n < 1:
            continue
        d = triple_pid(own[:n], partner[:n], own[1 : n + 1])
        for t in range(n):
            rows.append(
                {
                    "dialogue_id": dialogue.dialogue_id,
                    "role": role,
                    "speaker_id": speaker_id,
                    "t": t + 1,
                    **{k: float(vals[t]) for k, vals in d.items()},
                }
            )
    return rows


def compute_corpus_pid(
    dialogues: list[Dialogue],
    model: str = config.EMBEDDING_MODEL,
    corpus: str | None = None,
    variant: str = "real",
    seed: int = 0,
) -> pd.DataFrame:
    """PID rows for every cached dialogue of a corpus.

    variant "shuffled_corpus" replaces each utterance with one drawn from another
    dialogue of the same corpus, "shuffled_all" from a pool of all corpora. The
    draw follows the same convention as novelty.compute_corpus_novelty_cos, so the
    same seed gives the same surrogate series as the matching variant there.
    """
    from . import novelty

    # Matches the position in novelty's variants=("real", "shuffled_corpus", "shuffled_all").
    seed_offset = {"shuffled_corpus": 1, "shuffled_all": 2}

    pool = pool_ids = None
    if variant == "shuffled_corpus":
        pool, pool_ids = novelty.load_corpus_pool(
            [d.dialogue_id for d in dialogues], model, corpus
        )
    elif variant == "shuffled_all":
        pool, pool_ids = novelty.load_shuffle_pool(model)

    rows = []
    for d in dialogues:
        emb = embeddings.load_embeddings(d.dialogue_id, model, corpus)
        if variant in seed_offset:
            s = (seed * 1_000_003 + d.dialogue_id) * 7 + seed_offset[variant]
            exclude = (
                d.dialogue_id if variant == "shuffled_corpus"
                else novelty.pool_id(corpus, d.dialogue_id)
            )
            emb = novelty.resample_from_corpus(
                len(emb), exclude, pool, pool_ids, np.random.default_rng(s)
            )
        rows.extend(dialogue_pid_rows(d, emb))
    return pd.DataFrame(rows)


def dialogue_means(df: pd.DataFrame) -> pd.DataFrame:
    """Atoms averaged over turns, per (dialogue, role). Linear, so the identity survives."""
    cols = ["q_x", "q_y", "q_xy", "q_union", "delta_y_given_x", "delta_x_given_y",
            *ATOMS, "omega", "rsi"]
    return df.groupby(["dialogue_id", "role"], as_index=False)[cols].mean()
