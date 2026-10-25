"""Generate LLM--LLM (self-play) dialogues, and the seed and persona helpers.

The older cells were generated here; the cells used in the paper come from
generate_llm_llm_sequential.py, which imports the helpers below. Output follows the
schema of the human--LLM logs with interlocutor replaced by "A"/"B", so the same
loaders and measurements apply.

Design:
- self-play: one chat model plays both speakers, alternating viewpoints.
- Real opening and real personas: each dialogue is seeded from a RealPersonaChat
  dialogue, giving speakers A and B the personas of its two speakers and placing
  its opening utterance as A's first turn. This grounds the topics and personas in
  the human corpus.
- Utterance length is not constrained here, matching the prompts of the human--LLM
  logs, so the two LLM corpora sit in the same length regime.
- Personas only, no personality scores: the mapping from those scores to text
  cannot be reproduced from public information, whereas personas are available for
  all 233 speakers.

Run from the repository root:
    uv run python scripts/generate_llm_llm.py --dry-run
    uv run python scripts/generate_llm_llm.py
    uv run python scripts/generate_llm_llm.py --model gpt-4o-mini --n 30 --turns 20

Needs CHAT_API_KEY or OPENAI_API_KEY in .env. Dialogues already in the output file
are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from semantic_closure import config
from semantic_closure.data import Dialogue  # referenced in type comments  # noqa: F401

# Namespace of the generated dialogues: dialogue_id = NAMESPACE*100000 + index,
# unique within the corpus. Another model or condition needs another --namespace.
DEFAULT_NAMESPACE = 1


# --- Personas and seeds ------------------------------------------------------


def load_personas() -> dict[str, list[str]]:
    """interlocutor_id -> persona, as a list of sentences."""
    raw = json.loads(config.interlocutors_path().read_text())
    return {k: v["persona"] for k, v in raw.items()}


def pick_seed_dialogues(n: int) -> list[dict]:
    """n seeds taken at a fixed stride over dialogue ids, deterministically.

    Each element is {seed_dialogue_id, interlocutor_a, interlocutor_b, opening},
    where a is the speaker who goes first and opening is that speaker's utterance.
    """
    paths = sorted(config.dialogues_dir().glob("*.json"))
    if not paths:
        raise SystemExit("No RealPersonaChat dialogues found; check RPC_DATA_DIR.")
    # Stride sampling, to spread the topics. Returns fewer than n if there are fewer.
    stride = max(1, len(paths) // n)
    picked = paths[::stride][:n]

    seeds: list[dict] = []
    for p in picked:
        raw = json.loads(p.read_text())
        utts = raw.get("utterances") or []
        if not utts:
            continue
        first = utts[0]
        a_id = first["interlocutor_id"]
        others = [i for i in raw.get("interlocutors", []) if i != a_id]
        if not others:
            continue
        seeds.append(
            {
                "seed_dialogue_id": raw["dialogue_id"],
                "interlocutor_a": a_id,
                "interlocutor_b": others[0],
                "opening": first["text"],
            }
        )
    return seeds


def pick_seed_dialogues_random(n: int, rng_seed: int = 0) -> list[dict]:
    """n seeds drawn at random, with a fixed seed. Same elements as pick_seed_dialogues.

    Unlike the stride sampling above, this does not depend on the ordering. Sorted
    by id, 87.7% of adjacent RealPersonaChat dialogues share a speaker, because ids
    are assigned in blocks per speaker, so a stride over that ordering would need
    justifying.

    The population is the 13,581 dialogues that are not duplicate files and have a
    persona for both speakers. It is filtered before the draw, not after, which
    would leave fewer than n.

    The draw shuffles the population and takes the first n rather than calling
    random.sample, whose result for a larger n does not start with the result for a
    smaller one. Otherwise trying --n 2 and then running --n 150 would silently
    reassign dialogue_ids to different seeds, and partially saved utterances would
    be picked up as belonging to another seed.
    """
    paths = sorted(config.dialogues_dir().glob("*.json"))
    if not paths:
        raise SystemExit("No RealPersonaChat dialogues found; check RPC_DATA_DIR.")
    personas = load_personas()

    pool: list[dict] = []
    for p in paths:
        raw = json.loads(p.read_text())
        if int(p.stem) != raw["dialogue_id"]:  # duplicate file
            continue
        utts = raw.get("utterances") or []
        if not utts:
            continue
        first = utts[0]
        a_id = first["interlocutor_id"]
        others = [i for i in raw.get("interlocutors", []) if i != a_id]
        if not others or a_id not in personas or others[0] not in personas:
            continue
        pool.append(
            {
                "seed_dialogue_id": raw["dialogue_id"],
                "interlocutor_a": a_id,
                "interlocutor_b": others[0],
                "opening": first["text"],
            }
        )
    if len(pool) < n:
        raise SystemExit(f"only {len(pool)} seed candidates available, need {n}")
    random.Random(rng_seed).shuffle(pool)
    return pool[:n]


# --- Prompt construction -----------------------------------------------------


def _persona_block(label: str, persona: list[str]) -> str:
    lines = "\n".join(persona)
    return f"[話者{label}のプロフィール]\n{lines}"


def build_prompt(
    persona_a: list[str],
    persona_b: list[str],
    transcript: list[str],
    next_speaker: str,
) -> list[dict]:
    """Chat messages for generating the next single utterance.

    Follows the prompt of the human--LLM logs and gives both personas, with no
    length instruction.
    """
    system = (
        "あなたは日本語の雑談対話を生成するアシスタントです。"
        "話者Aと話者Bによる自然な雑談の続きを作ります。"
        f"いまは話者{next_speaker}の番です。話者{next_speaker}になりきり、次の1発話だけを"
        "日本語で出力してください。話者名やラベル、記号は付けず、発話本文のみを返します。"
    )
    body = [
        "以下は、話者Aと話者Bの対話です。",
        "",
        _persona_block("A", persona_a),
        "",
        _persona_block("B", persona_b),
        "",
        "[対話]",
    ]
    for i, text in enumerate(transcript):
        spk = "A" if i % 2 == 0 else "B"
        body.append(f"話者{spk}: {text}")
    body.append(f"話者{next_speaker}:")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(body)},
    ]


def _clean_utterance(text: str, speaker: str) -> str:
    """Reduce a model reply to one utterance: strip a speaker prefix, keep one line."""
    t = text.strip()
    for prefix in (f"話者{speaker}:", f"話者{speaker}：", f"{speaker}:", f"{speaker}："):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    # The model sometimes writes both speakers, so keep the first line only.
    t = t.splitlines()[0].strip() if t else t
    return t


# --- Chat client -------------------------------------------------------------


class OpenAIChatClient:
    def __init__(self, temperature: float, max_tokens: int) -> None:
        import os

        from openai import OpenAI

        # The generation key can be separate from the embedding key: use
        # CHAT_API_KEY when set, otherwise fall back to OPENAI_API_KEY.
        api_key = os.environ.get("CHAT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Neither CHAT_API_KEY nor OPENAI_API_KEY is set. "
                "Put a generation key in CHAT_API_KEY in .env."
            )
        self._client = OpenAI(api_key=api_key, timeout=60.0, max_retries=5)
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, messages: list[dict], model: str) -> str:
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content or ""


# --- Generation --------------------------------------------------------------


def generate_dialogue(
    client: OpenAIChatClient,
    model: str,
    dialogue_id: int,
    seed: dict,
    personas: dict[str, list[str]],
    turns: int,
) -> dict:
    persona_a = personas[seed["interlocutor_a"]]
    persona_b = personas[seed["interlocutor_b"]]
    transcript: list[str] = [seed["opening"]]  # utterance 0 is the real opening, by A

    while len(transcript) < turns:
        next_speaker = "A" if len(transcript) % 2 == 0 else "B"
        messages = build_prompt(persona_a, persona_b, transcript, next_speaker)
        text = _clean_utterance(client.complete(messages, model), next_speaker)
        if not text:
            text = "……"  # keep the alternation intact on an empty reply
        transcript.append(text)

    utterances = [
        {"utterance_id": i, "interlocutor": "A" if i % 2 == 0 else "B", "text": t}
        for i, t in enumerate(transcript)
    ]
    return {
        "dialogue_id": dialogue_id,
        "system_id": f"{model}_selfplay_persona",
        "interlocutor_id": f"{seed['interlocutor_a']}+{seed['interlocutor_b']}",
        "seed_dialogue_id": seed["seed_dialogue_id"],
        "utterances": utterances,
    }


def output_path(model: str, namespace: int) -> Path:
    safe = model.replace("/", "_")
    return config.llm_llm_generated_dir() / f"{safe}__ns{namespace}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.CHAT_MODEL, help="chat model used for self-play")
    parser.add_argument("--n", type=int, default=30, help="number of dialogues to generate")
    parser.add_argument("--turns", type=int, default=20, help="utterances per dialogue")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=300, help="output cap per utterance")
    parser.add_argument("--namespace", type=int, default=DEFAULT_NAMESPACE,
                        help="namespace of dialogue_id; change it per model or condition")
    parser.add_argument("--dry-run", action="store_true", help="show the plan without calling the API")
    args = parser.parse_args()

    personas = load_personas()
    seeds = pick_seed_dialogues(args.n)
    # Keep only seeds with a persona for both speakers.
    seeds = [s for s in seeds
             if s["interlocutor_a"] in personas and s["interlocutor_b"] in personas]
    if not seeds:
        raise SystemExit("no usable seed dialogues")

    out = output_path(args.model, args.namespace)
    existing_ids: set[int] = set()
    if out.exists():
        existing_ids = {e["dialogue_id"] for e in json.loads(out.read_text())}

    plan = []
    for i, seed in enumerate(seeds, start=1):
        did = args.namespace * 100000 + i
        if did in existing_ids:
            continue
        plan.append((did, seed))

    print(f"model {args.model}, self-play, {len(seeds)} dialogues ({len(plan)} to generate), "
          f"{args.turns} utterances, temperature {args.temperature}")
    print(f"output: {out}")

    if args.dry_run:
        # Only the order of magnitude. The input grows with each turn, since the
        # prompt carries the whole history, and prices depend on the model.
        gen_utts = len(plan) * (args.turns - 1)
        print(f"estimate: ~{gen_utts} utterances to generate; the input accumulates per turn")
        if plan:
            s = plan[0][1]
            print("\n--- first seed ---")
            print(f"seed_dialogue_id={s['seed_dialogue_id']}, "
                  f"A={s['interlocutor_a']}, B={s['interlocutor_b']}")
            print(f"opening (speaker A): {s['opening']}")
            print(f"A persona: {personas[s['interlocutor_a']][:2]} ...")
            print(f"B persona: {personas[s['interlocutor_b']][:2]} ...")
        return

    if not plan:
        print("nothing to generate; all dialogues are present")
        return

    client = OpenAIChatClient(args.temperature, args.max_tokens)
    results = list(json.loads(out.read_text())) if out.exists() else []
    out.parent.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm

    for did, seed in tqdm(plan, desc=f"generating ({args.model})"):
        d = generate_dialogue(client, args.model, did, seed, personas, args.turns)
        results.append(d)
        # Save after every dialogue, so an interrupted run resumes.
        results.sort(key=lambda e: e["dialogue_id"])
        tmp = out.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        tmp.rename(out)

    print(f"done: generated {len(plan)} dialogues into {out} ({len(results)} in total)")
    print("next: uv run python scripts/embed_llm_llm.py")


if __name__ == "__main__":
    main()
