"""Paths and environment variables.

Scripts are run from the repository root with `uv run`. Every value below can be
overridden in `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

#: Root of a local clone of RealPersonaChat.
RPC_DATA_DIR = Path(os.environ.get("RPC_DATA_DIR", "../real-persona-chat"))

#: Embedding cache. One subdirectory per model, since dimensions differ.
EMBEDDING_CACHE_DIR = Path(os.environ.get("EMBEDDING_CACHE_DIR", "data/embeddings"))

#: Figures and computed metrics.
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

#: Chat model used for LLM--LLM self-play generation.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

#: Generated dialogues, kept separate from the RealPersonaChat clone.
GENERATED_DIR = Path(os.environ.get("GENERATED_DIR", "data/generated"))


def dialogues_dir() -> Path:
    d = RPC_DATA_DIR / "real_persona_chat" / "dialogues"
    if not d.is_dir():
        raise FileNotFoundError(
            f"RealPersonaChat not found at {d}\n"
            "Clone it with git clone https://github.com/nu-dialogue/real-persona-chat.git "
            "and set RPC_DATA_DIR in .env to the clone."
        )
    return d


def llm_dialogue_dir() -> Path:
    """Human--LLM dialogue logs (GPT-*.json), inside the RealPersonaChat clone."""
    d = RPC_DATA_DIR / "llm_dialogue_system"
    if not d.is_dir():
        raise FileNotFoundError(
            f"Human--LLM dialogue logs not found at {d}\n"
            "They live in llm_dialogue_system/ inside the RealPersonaChat clone (RPC_DATA_DIR)."
        )
    return d


def interlocutors_path() -> Path:
    """RealPersonaChat speaker data. The source of personas for LLM--LLM generation."""
    p = RPC_DATA_DIR / "real_persona_chat" / "interlocutors.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"Speaker data not found at {p}\n"
            "It lives in interlocutors.json inside the RealPersonaChat clone (RPC_DATA_DIR)."
        )
    return p


def llm_llm_generated_dir() -> Path:
    return GENERATED_DIR / "llm_llm"


def counterfactual_generated_dir() -> Path:
    """Dialogues with one human speaker replaced."""
    return GENERATED_DIR / "counterfactual"
