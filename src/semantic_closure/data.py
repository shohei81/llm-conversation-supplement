"""Dialogue loaders, for RealPersonaChat and for the generated corpora.

Roles: in every dialogue the speaker who goes first is X and the other is Y.
Utterances strictly alternate, so for a 1-based pair index t,
X_t = utterances[2t-2] and Y_t = utterances[2t-1].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import config


@dataclass(frozen=True)
class Utterance:
    utterance_id: int
    speaker_id: str
    role: str  # "X" goes first, "Y" second
    pair_index: int  # 1-based pair index t
    text: str


@dataclass(frozen=True)
class Dialogue:
    dialogue_id: int
    speaker_x: str
    speaker_y: str
    utterances: list[Utterance]
    #: Dialogue system name; set for human--LLM dialogues only.
    system_id: str | None = None
    #: Speaker the persona came from; set for human--LLM dialogues only.
    interlocutor_id: str | None = None

    @property
    def n_pairs(self) -> int:
        """Number of complete (X_t, Y_t) pairs."""
        return len(self.utterances) // 2

    def texts(self) -> list[str]:
        return [u.text for u in self.utterances]


def load_dialogue(path: Path) -> Dialogue:
    raw = json.loads(path.read_text())
    utts = raw["utterances"]
    if not utts:
        raise ValueError(f"dialogue has no utterances: {path}")

    speaker_x = utts[0]["interlocutor_id"]
    others = {u["interlocutor_id"] for u in utts} - {speaker_x}
    speaker_y = others.pop() if others else ""

    utterances = []
    for i, u in enumerate(utts):
        expected = speaker_x if i % 2 == 0 else speaker_y
        if u["interlocutor_id"] != expected:
            # No consecutive same-speaker utterance exists in the corpus, but do
            # not proceed silently if that ever changes.
            raise ValueError(
                f"utterances do not alternate: dialogue={raw['dialogue_id']}, "
                f"utterance={u['utterance_id']}"
            )
        utterances.append(
            Utterance(
                utterance_id=u["utterance_id"],
                speaker_id=u["interlocutor_id"],
                role="X" if i % 2 == 0 else "Y",
                pair_index=i // 2 + 1,
                text=u["text"],
            )
        )

    return Dialogue(
        dialogue_id=raw["dialogue_id"],
        speaker_x=speaker_x,
        speaker_y=speaker_y,
        utterances=utterances,
    )


def dialogue_path(dialogue_id: int) -> Path:
    return config.dialogues_dir() / f"{dialogue_id:05d}.json"


# --- Human--LLM dialogues ----------------------------------------------------
#
# Logs built by giving GPT the RealPersonaChat personas (llm_dialogue_system/GPT-*.json).
# The JSON differs from the human--human corpus:
#   - one file holds an array of dialogues, with dialogue_id running 1..30 inside
#     each file, so ids collide across files and are made unique by mixing in the
#     file's index
#   - the speaker is the interlocutor field, "user" (human) or "system" (LLM), and
#     every dialogue strictly alternates starting from user, fixing X=human, Y=LLM

#: File order that synthetic ids are built from; this index is embedded in them.
LLM_SYSTEM_FILES = [
    "GPT-4.json",
    "GPT-4+persona.json",
    "GPT-4+personality.json",
    "GPT-4+persona+personality.json",
    "GPT-3.5+persona+personality.json",
]


def llm_synthetic_id(system_index: int, local_id: int) -> int:
    """(1-based file index, dialogue_id within the file) -> unique synthetic id.

    system_index=1 with local_id=5 gives 1005, so the system is readable in %05d.
    """
    return system_index * 1000 + local_id


def _load_llm_file(path: Path, system_index: int) -> list[Dialogue]:
    raw = json.loads(path.read_text())
    dialogues: list[Dialogue] = []
    for entry in raw:
        utts = entry["utterances"]
        if not utts:
            raise ValueError(f"dialogue has no utterances: {path} dialogue_id={entry.get('dialogue_id')}")
        utterances = []
        for i, u in enumerate(utts):
            expected = "user" if i % 2 == 0 else "system"
            if u["interlocutor"] != expected:
                raise ValueError(
                    f"user/system utterances do not alternate: {path} "
                    f"dialogue_id={entry['dialogue_id']}, utterance={u['utterance_id']}"
                )
            utterances.append(
                Utterance(
                    utterance_id=u["utterance_id"],
                    speaker_id=u["interlocutor"],  # "user" (human) / "system" (LLM)
                    role="X" if i % 2 == 0 else "Y",  # X=human, goes first; Y=LLM
                    pair_index=i // 2 + 1,
                    text=u["text"],
                )
            )
        dialogues.append(
            Dialogue(
                dialogue_id=llm_synthetic_id(system_index, entry["dialogue_id"]),
                speaker_x="user",
                speaker_y="system",
                utterances=utterances,
                system_id=entry["system_id"],
                interlocutor_id=entry.get("interlocutor_id"),
            )
        )
    return dialogues


def iter_llm_dialogues(
    dialogue_ids: list[int] | None = None,
) -> Iterator[Dialogue]:
    """Human--LLM dialogues in synthetic-id order; dialogue_ids filters by that id."""
    d = config.llm_dialogue_dir()
    all_dialogues: list[Dialogue] = []
    for idx, fname in enumerate(LLM_SYSTEM_FILES, start=1):
        p = d / fname
        if not p.exists():
            raise FileNotFoundError(f"human--LLM dialogue file not found: {p}")
        all_dialogues.extend(_load_llm_file(p, idx))
    all_dialogues.sort(key=lambda x: x.dialogue_id)
    if dialogue_ids is not None:
        keep = set(dialogue_ids)
        all_dialogues = [x for x in all_dialogues if x.dialogue_id in keep]
    yield from all_dialogues


# --- LLM--LLM dialogues (self-play) ------------------------------------------
#
# Written by generate_llm_llm.py into data/generated/llm_llm/*.json. Both speakers
# are LLMs, so they are symmetric: the labels are "A" (first) and "B" (second) and
# the roles are X=first, Y=second as in the human--human corpus. The JSON follows
# llm_dialogue_system with interlocutor replaced by "A"/"B". dialogue_id is made
# unique within the corpus at generation time as namespace*100000 + index.


def _load_llm_llm_file(path: Path) -> list[Dialogue]:
    raw = json.loads(path.read_text())
    dialogues: list[Dialogue] = []
    for entry in raw:
        utts = entry["utterances"]
        if not utts:
            raise ValueError(f"dialogue has no utterances: {path} dialogue_id={entry.get('dialogue_id')}")
        utterances = []
        for i, u in enumerate(utts):
            expected = "A" if i % 2 == 0 else "B"
            if u["interlocutor"] != expected:
                raise ValueError(
                    f"A/B utterances do not alternate: {path} "
                    f"dialogue_id={entry['dialogue_id']}, utterance={u['utterance_id']}"
                )
            utterances.append(
                Utterance(
                    utterance_id=u["utterance_id"],
                    speaker_id=u["interlocutor"],  # "A" first, "B" second; both LLMs
                    role="X" if i % 2 == 0 else "Y",
                    pair_index=i // 2 + 1,
                    text=u["text"],
                )
            )
        dialogues.append(
            Dialogue(
                dialogue_id=entry["dialogue_id"],
                speaker_x="A",
                speaker_y="B",
                utterances=utterances,
                system_id=entry.get("system_id"),
                interlocutor_id=entry.get("interlocutor_id"),
            )
        )
    return dialogues


def iter_llm_llm_dialogues(
    dialogue_ids: list[int] | None = None,
) -> Iterator[Dialogue]:
    """LLM--LLM dialogues in id order; dialogue_ids filters."""
    d = config.llm_llm_generated_dir()
    if not d.is_dir():
        raise FileNotFoundError(
            f"LLM--LLM dialogues not found: {d}\n"
            "Generate them first with scripts/generate_llm_llm.py."
        )
    all_dialogues: list[Dialogue] = []
    for p in sorted(d.glob("*.json")):
        all_dialogues.extend(_load_llm_llm_file(p))
    all_dialogues.sort(key=lambda x: x.dialogue_id)
    if dialogue_ids is not None:
        keep = set(dialogue_ids)
        all_dialogues = [x for x in all_dialogues if x.dialogue_id in keep]
    yield from all_dialogues


def iter_llm_llm_oneshot_dialogues(
    dialogue_ids: list[int] | None = None,
) -> Iterator[Dialogue]:
    """The LLM--LLM dialogues used in the paper.

    Written into data/generated/llm_llm/oneshot/ by generate_llm_llm_oneshot.py and
    generate_llm_llm_sequential.py. The generation conditions differ from the older
    cells, so they sit where iter_llm_llm_dialogues (a non-recursive glob) does not
    reach and count as a separate corpus. The JSON schema is the same.
    """
    d = config.llm_llm_generated_dir() / "oneshot"
    if not d.is_dir():
        raise FileNotFoundError(
            f"dialogues not found: {d}\n"
            "Generate them first with scripts/generate_llm_llm_sequential.py."
        )
    all_dialogues: list[Dialogue] = []
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("manifest__"):
            continue
        all_dialogues.extend(_load_llm_llm_file(p))
    all_dialogues.sort(key=lambda x: x.dialogue_id)
    if dialogue_ids is not None:
        keep = set(dialogue_ids)
        all_dialogues = [x for x in all_dialogues if x.dialogue_id in keep]
    yield from all_dialogues


# --- Dialogues with one speaker replaced -------------------------------------
#
# Written by generate_counterfactual.py into data/generated/counterfactual/*.json.
# Only the utterances of one speaker (masked_role) are replaced, by LLM output or by
# shuffled human utterances; the other speaker, the topic and the length match the
# source dialogue exactly, giving a minimal pair. The JSON uses the same A/B schema
# as LLM--LLM, so the loader is shared. entry["source_dialogue_id"] points at the
# source and entry["masked_role"] says which side was replaced.

#: Prefix of files that hold generation conditions rather than dialogues.
_CF_NON_DIALOGUE_PREFIX = "manifest__"


def _counterfactual_files() -> list[Path]:
    d = config.counterfactual_generated_dir()
    return sorted(
        p for p in d.glob("*.json")
        if not p.name.startswith(_CF_NON_DIALOGUE_PREFIX)
    )


def iter_counterfactual_dialogues(
    dialogue_ids: list[int] | None = None,
) -> Iterator[Dialogue]:
    """Replaced-speaker dialogues in id order; dialogue_ids filters."""
    d = config.counterfactual_generated_dir()
    if not d.is_dir():
        raise FileNotFoundError(
            f"replaced-speaker dialogues not found: {d}\n"
            "Generate them first with scripts/generate_counterfactual.py."
        )
    all_dialogues: list[Dialogue] = []
    for p in _counterfactual_files():
        all_dialogues.extend(_load_llm_llm_file(p))
    all_dialogues.sort(key=lambda x: x.dialogue_id)
    if dialogue_ids is not None:
        keep = set(dialogue_ids)
        all_dialogues = [x for x in all_dialogues if x.dialogue_id in keep]
    yield from all_dialogues


def load_counterfactual_meta() -> dict[int, dict]:
    """dialogue_id -> generation metadata for the replaced-speaker dialogues.

    Holds the arm, the source dialogue, the replaced side and the prompt, but no
    utterance text.
    """
    if not config.counterfactual_generated_dir().is_dir():
        return {}
    meta: dict[int, dict] = {}
    for p in _counterfactual_files():
        for entry in json.loads(p.read_text()):
            meta[entry["dialogue_id"]] = {
                k: v for k, v in entry.items() if k != "utterances"
            }
    return meta


def iter_dialogues(
    limit: int | None = None,
    dialogue_ids: list[int] | None = None,
) -> Iterator[Dialogue]:
    """Human--human dialogues in id order; limit takes the first n."""
    if dialogue_ids is not None:
        paths = [dialogue_path(i) for i in dialogue_ids]
    else:
        paths = sorted(config.dialogues_dir().glob("*.json"))
        if limit is not None:
            paths = paths[:limit]
    for p in paths:
        d = load_dialogue(p)
        # Two files in the corpus disagree with the dialogue_id inside them:
        # 10279.json duplicates 10577 and 11052.json duplicates 11838, text included.
        # Keep only files whose name matches the id, so duplicates are not counted twice.
        if int(p.stem) != d.dialogue_id:
            continue
        yield d
