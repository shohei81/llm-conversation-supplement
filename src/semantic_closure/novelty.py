"""Adjacent cosines of the novelty vector.

The residual r of z = Proj_span(x,y) z + r is the novelty vector, and ||r||**2
equals the novelty N = 1 - Q_XY of triple_pid.

Treating the novelty vectors as a series like the utterance embeddings gives
- cos_next_self:  cosine with the same speaker's next novelty vector
- cos_next_other: cosine with the next novelty vector in time, i.e. the partner's
                  (after X's t comes Y's t; after Y's t comes X's t+1)

Cosines are scale invariant, so normalizing the novelty vectors does not change
the result; the implementation dots the normalized vectors.

cos(r_{t+1}, r_t) carries a mechanical negative bias: r_hat_{t+1} is orthogonal to
the whole utterance that is its own source, and that utterance contains r_t as a
component. Read the difference against the surrogate series
(variant="shuffled_corpus"), not the absolute value.
"""

from __future__ import annotations

import numpy as np

from . import config, embeddings
from .data import Dialogue
from .metrics import normalize_rows

#: Below this, ||r||**2 leaves the direction meaningless (z lies in the context plane).
EPS_NORM = 1e-6


def novelty_unit_vectors(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized novelty vectors r = z - Proj_span(x,y) z, one per triple.

    x, y, z are row-normalized (n, dim). Returns (r_hat, nov, valid), where
    nov = ||r||**2 equals the novelty of triple_pid, and rows with valid=False
    (z nearly inside the context plane) get a zero r_hat.
    """
    xy = np.einsum("ij,ij->i", x, y)
    y_perp = y - xy[:, None] * x
    s = np.linalg.norm(y_perp, axis=1)
    # Where x and y are parallel the span is x alone, as in triple_pid.
    parallel = s < 1e-6
    s_safe = np.where(parallel, 1.0, s)
    y_tilde = y_perp / s_safe[:, None]
    y_tilde[parallel] = 0.0

    proj = (
        np.einsum("ij,ij->i", z, x)[:, None] * x
        + np.einsum("ij,ij->i", z, y_tilde)[:, None] * y_tilde
    )
    r = z - proj
    nov = np.einsum("ij,ij->i", r, r)
    valid = nov > EPS_NORM
    r_hat = np.where(valid[:, None], r / np.sqrt(np.clip(nov, EPS_NORM, None))[:, None], 0.0)
    return r_hat, nov, valid


def _role_novelty_series(e: np.ndarray) -> dict[str, tuple | None]:
    """Novelty vector series per role, from a row-normalized utterance series.

    Item t of role X is the residual of (X_t, Y_t, X_{t+1}), the new part of X_{t+1};
    of role Y, the residual of (Y_t, X_{t+1}, Y_{t+1}), the new part of Y_{t+1}.
    """
    x, y = e[0::2], e[1::2]
    out: dict[str, tuple | None] = {}
    for role, own, partner in (("X", x, y[: len(x) - 1]), ("Y", y, x[1 : len(y)])):
        n = min(len(own) - 1, len(partner))
        out[role] = (
            novelty_unit_vectors(own[:n], partner[:n], own[1 : n + 1]) if n >= 1 else None
        )
    return out


def shuffle_pairs(emb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reorder the pairs (X_t, Y_t) at random, keeping each pair as a block.

    Preserves the marginal distribution of utterances while destroying the temporal
    structure. A trailing odd utterance is dropped.
    """
    n_pairs = len(emb) // 2
    perm = rng.permutation(n_pairs)
    idx = np.stack([2 * perm, 2 * perm + 1], axis=1).reshape(-1)
    return emb[idx]


def dialogue_novelty_cos_rows(
    dialogue: Dialogue, emb: np.ndarray, variant: str = "real", seed: int | None = None
) -> list[dict]:
    """Adjacent novelty cosines of one dialogue, one row per novelty vector.

    Columns: novelty is that vector's N = ||r||**2; cos_next_self and novelty_next
    are the cosine with the same speaker's next novelty vector and its novelty;
    cos_next_other is the cosine with the next novelty vector in time, the
    partner's; cos_uptake is the cosine with the partner's next *utterance*
    embedding, which carries no mechanical bias because r_hat is not orthogonalized
    against that utterance. Missing or invalid targets give NaN.

    variant="shuffled" shuffles the pair order within the dialogue first. The
    surrogate series for "shuffled_corpus" is passed in as emb by
    compute_corpus_novelty_cos rather than built here.
    """
    e = normalize_rows(emb.astype(np.float64))
    if variant == "shuffled":
        e = shuffle_pairs(e, np.random.default_rng(seed))
    ser = _role_novelty_series(e)
    if ser["X"] is None or ser["Y"] is None:
        return []
    r_x, nov_x, v_x = ser["X"]
    r_y, nov_y, v_y = ser["Y"]
    x, y = e[0::2], e[1::2]

    rows = []
    # other_offset: after X's t comes Y's t (offset 0); after Y's t comes X's t+1.
    # uptake: the partner's next utterance after r^X_t is Y_{t+1} = y[t+1], and
    #         after r^Y_t it is X_{t+2} = x[t+2].
    for role, speaker_id, (own, nov, v_own), (oth, v_oth), off, (utts, u_off) in (
        ("X", dialogue.speaker_x, (r_x, nov_x, v_x), (r_y, v_y), 0, (y, 1)),
        ("Y", dialogue.speaker_y, (r_y, nov_y, v_y), (r_x, v_x), 1, (x, 2)),
    ):
        for t in range(len(own)):
            if not v_own[t]:
                continue
            row = {
                "dialogue_id": dialogue.dialogue_id,
                "role": role,
                "speaker_id": speaker_id,
                "t": t + 1,
                "variant": variant,
                "novelty": float(nov[t]),
                "cos_next_self": float("nan"),
                "novelty_next": float("nan"),
                "cos_next_other": float("nan"),
                "cos_uptake": float("nan"),
            }
            if t + 1 < len(own) and v_own[t + 1]:
                row["cos_next_self"] = float(own[t] @ own[t + 1])
                row["novelty_next"] = float(nov[t + 1])
            j = t + off
            if j < len(oth) and v_oth[j]:
                row["cos_next_other"] = float(own[t] @ oth[j])
            k = t + u_off
            if k < len(utts):
                row["cos_uptake"] = float(own[t] @ utts[k])
            rows.append(row)
    return rows


def shuffle_within_speakers(emb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each speaker's utterances independently, keeping the alternation.

    shuffle_pairs moves whole pairs and so preserves the adjacency inside a pair
    (Y_t right after X_t), which is not a sufficient control for measurements that
    use the next utterance. This breaks adjacency on both sides.
    """
    out = emb.copy()
    for start in (0, 1):
        idx = np.arange(start, len(emb), 2)
        out[idx] = emb[rng.permutation(idx)]
    return out


def load_corpus_pool(
    dialogue_ids: list[int],
    model: str = config.EMBEDDING_MODEL,
    corpus: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """All utterances of one corpus, concatenated, for a pool limited to that corpus.

    Returns (pool, pool_dialogue_ids): pool is (n_utterances, dim) float32 and
    pool_dialogue_ids gives each row's source dialogue, used to exclude a dialogue
    from its own draw.
    """
    embs, ids = [], []
    for did in dialogue_ids:
        e = embeddings.load_embeddings(did, model, corpus)
        embs.append(e)
        ids.append(np.full(len(e), did, dtype=np.int64))
    return np.concatenate(embs), np.concatenate(ids)


#: Corpora mixed into the shuffling pool. This reference exists to show that
#: novelty can be driven up by unrelated utterances, so the populations are mixed
#: rather than matched.
POOL_CORPORA = (None, "llm", "llm_llm")

#: Stride that makes dialogue_id unique across corpora inside the pool.
_POOL_ID_STRIDE = 10_000_000


def pool_id(corpus: str | None, dialogue_id: int) -> int:
    """Id that is unique across corpora, so a dialogue can be excluded from its own draw."""
    return POOL_CORPORA.index(corpus) * _POOL_ID_STRIDE + dialogue_id


def load_shuffle_pool(
    model: str = config.EMBEDDING_MODEL,
    corpora: tuple[str | None, ...] = POOL_CORPORA,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffling pool mixing the utterances of all corpora.

    The dialogues come from the loaders, not from the embedding cache: the llm_llm
    cache still holds cells that were excluded from the analysis.

    Returns (pool, pool_ids), with pool_ids following pool_id() so a dialogue can be
    excluded from its own draw.
    """
    from .data import iter_dialogues, iter_llm_dialogues, iter_llm_llm_dialogues

    loader = {None: iter_dialogues, "llm": iter_llm_dialogues, "llm_llm": iter_llm_llm_dialogues}
    embs, ids = [], []
    for corpus in corpora:
        for d in loader[corpus]():
            e = embeddings.load_embeddings(d.dialogue_id, model, corpus)
            embs.append(e)
            ids.append(np.full(len(e), pool_id(corpus, d.dialogue_id), dtype=np.int64))
    return np.concatenate(embs), np.concatenate(ids)


def resample_from_corpus(
    n: int,
    dialogue_id: int,
    pool: np.ndarray,
    pool_dialogue_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Surrogate series of n utterances drawn from the pool without replacement.

    Topical coherence is destroyed along with everything else, which gives the
    reference for how far unrelated utterances can drive novelty up. Utterances of
    the dialogue itself are excluded from the draw.
    """
    ok = np.flatnonzero(pool_dialogue_ids != dialogue_id)
    if len(ok) < n:
        raise ValueError(f"need {n} utterances from other dialogues, have {len(ok)}")
    return pool[rng.choice(ok, size=n, replace=False)]


def dialogue_novelty_uptake_rows(
    dialogue: Dialogue, emb: np.ndarray, variant: str = "real", seed: int | None = None
) -> list[dict]:
    """Uptake of novelty: how much of it lands in the next utterance.

    How much of the novelty vector r_hat lands in the next utterance e_next is a
    projection onto a single vector, so it equals cos**2(r_hat, e_next), the same
    form as a coverage Q. The baseline is the mean cos**2 against every *other*
    utterance of the same dialogue, which involves no draw and so is deterministic.

    Columns, one row per novelty vector, dropping turns with no following utterance:
    - novelty  : N_t = ||r||**2 of that turn
    - take_frac: cos**2(r_hat, next utterance). The next utterance is always the
                 partner's, so this is whether the partner picked it up.
    - base_frac: mean cos**2 against the other utterances of the dialogue
    - contrast : take_frac - base_frac, the primary measure
    - take     : the absolute amount taken up, N_t * take_frac

    The baseline excludes the three utterances of the triple (x and y are orthogonal
    to r_hat by construction, z contains it) and the next utterance. variant is
    "real", "shuffled_pairs", or "shuffled_speakers", which shuffles each speaker
    independently and so breaks adjacency completely. The surrogate series for
    "shuffled_corpus" is passed in as emb by compute_corpus_novelty_uptake.
    """
    e = normalize_rows(emb.astype(np.float64))
    if variant == "shuffled_pairs":
        e = shuffle_pairs(e, np.random.default_rng(seed))
    elif variant == "shuffled_speakers":
        e = shuffle_within_speakers(e, np.random.default_rng(seed))
    ser = _role_novelty_series(e)
    if ser["X"] is None or ser["Y"] is None:
        return []

    T = len(e)
    rows = []
    # Position of the target in the full series: 2t+2 for role X, 2t+3 for role Y.
    for role, speaker_id, (own, nov, v_own), g0 in (
        ("X", dialogue.speaker_x, ser["X"], 2),
        ("Y", dialogue.speaker_y, ser["Y"], 3),
    ):
        if not len(own):
            continue
        sq = (own @ e.T) ** 2  # (n_turns, n_utterances) of cos**2
        total = sq.sum(axis=1)
        for t in range(len(own)):
            g = 2 * t + g0
            if not v_own[t] or g + 1 > T - 1:
                continue
            excluded = (g - 2, g - 1, g, g + 1)  # the triple and the next utterance
            n_base = T - len(excluded)
            if n_base < 1:
                continue
            take = float(sq[t, g + 1])
            base = float((total[t] - sq[t, list(excluded)].sum()) / n_base)
            rows.append(
                {
                    "dialogue_id": dialogue.dialogue_id,
                    "role": role,
                    "speaker_id": speaker_id,
                    "t": t + 1,
                    "variant": variant,
                    "novelty": float(nov[t]),
                    "take_frac": take,
                    "base_frac": base,
                    "contrast": take - base,
                    "take": take * float(nov[t]),
                }
            )
    return rows


def compute_corpus_novelty_uptake(
    dialogues: list[Dialogue],
    model: str = config.EMBEDDING_MODEL,
    corpus: str | None = None,
    variants: tuple[str, ...] = ("real", "shuffled_corpus"),
    seed: int = 0,
):
    """Uptake for every cached dialogue of a corpus.

    variants defaults to the real dialogues and the shuffled_corpus control.
    The within-dialogue shuffles are available too. The draw is determined by seed
    and dialogue_id, so a re-run reproduces it.
    """
    import pandas as pd

    pool = pool_ids = None
    if "shuffled_corpus" in variants:
        pool, pool_ids = load_corpus_pool([d.dialogue_id for d in dialogues], model, corpus)

    rows = []
    for d in dialogues:
        emb = embeddings.load_embeddings(d.dialogue_id, model, corpus)
        for vi, variant in enumerate(variants):
            s = (seed * 1_000_003 + d.dialogue_id) * 7 + vi
            e_in = emb
            if variant == "shuffled_corpus":
                e_in = resample_from_corpus(
                    len(emb), d.dialogue_id, pool, pool_ids, np.random.default_rng(s)
                )
            rows.extend(dialogue_novelty_uptake_rows(d, e_in, variant=variant, seed=s))
    return pd.DataFrame(rows)


def compute_corpus_novelty_cos(
    dialogues: list[Dialogue],
    model: str = config.EMBEDDING_MODEL,
    corpus: str | None = None,
    variants: tuple[str, ...] = ("real", "shuffled_corpus", "shuffled_all"),
    seed: int = 0,
):
    """Adjacent novelty cosines for every cached dialogue of a corpus.

    variants defaults to the real dialogues and two surrogate series:
    shuffled_corpus draws from a pool of the same corpus, shuffled_all from a pool
    of all corpora. The within-dialogue pair shuffle is available too. The draw is
    determined by seed, dialogue_id and the position within variants, so a re-run
    reproduces the same controls.
    """
    import pandas as pd

    pool = pool_ids = None
    if "shuffled_corpus" in variants:
        pool, pool_ids = load_corpus_pool([d.dialogue_id for d in dialogues], model, corpus)
    all_pool = all_pool_ids = None
    if "shuffled_all" in variants:
        all_pool, all_pool_ids = load_shuffle_pool(model)

    rows = []
    for d in dialogues:
        emb = embeddings.load_embeddings(d.dialogue_id, model, corpus)
        for vi, variant in enumerate(variants):
            s = (seed * 1_000_003 + d.dialogue_id) * 7 + vi
            e_in = emb
            if variant == "shuffled_corpus":
                e_in = resample_from_corpus(
                    len(emb), d.dialogue_id, pool, pool_ids, np.random.default_rng(s)
                )
            elif variant == "shuffled_all":
                e_in = resample_from_corpus(
                    len(emb), pool_id(corpus, d.dialogue_id), all_pool, all_pool_ids,
                    np.random.default_rng(s),
                )
            rows.extend(dialogue_novelty_cos_rows(d, e_in, variant=variant, seed=s))
    return pd.DataFrame(rows)
