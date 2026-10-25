"""Prompt construction and dataset assembly for the LLM--LLM cells.

Defines the generation conditions (arms), builds the prompts, and assembles model
replies into a dataset with a manifest. generate_llm_llm_sequential.py drives the
actual generation and imports from here; the prompt text itself is in Japanese,
since the corpus is.

The conditions differ along three factors: whether real human dialogues are shown
as examples, whether the prompt constrains utterance length, and whether a speaker
sees only its own persona. Length is constrained because uncontrolled generation
produced utterances nearly three times as long as the human mean of 22.9
characters.

    # 1) build the prompts (no API needed)
    uv run python scripts/generate_llm_llm_oneshot.py prompts --n 10 -o /path/prompts.json
    # 2) assemble the replies into a dataset
    uv run python scripts/generate_llm_llm_oneshot.py assemble -i /path/replies.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from typing import NamedTuple

from semantic_closure import config
from semantic_closure.data import load_dialogue

from generate_llm_llm import (
    load_personas,
    pick_seed_dialogues,
    pick_seed_dialogues_random,
)

MODEL = "claude-haiku-4-5"
TURNS = 20  # utterances per dialogue; the first is the human seed, so 19 are generated
SEED_POOL = 150  # seed pool; the first n are used
N_SHOTS = 2
SHOT_TURNS = 20  # examples are cut to the same length as the target dialogues
#: Number of example candidates. Deliberately independent of --n: if it depended
#: on n, changing n would change which example dialogue i gets, so a pilot run and
#: the real run would differ in more than their size.
POOL_SIZE = 400

class Arm(NamedTuple):
    """A generation condition. Apart from namespace, what goes into the prompt."""

    namespace: int
    examples: bool       #: show real human dialogues as examples
    length: bool         #: constrain utterance length
    own_only: bool       #: show only the speaker's own persona (sequential only)
    topic_hint: bool     #: include the line allowing topics to shift and return
    revision: int = 1    #: prompt revision; 2 was rewritten for sequential generation


#: Arms for separating the factors. no_length and no_example drop one factor each
#: from full; plain differs from the older sequential cell in generation unit alone,
#: so that unit can be read as a one-factor difference.
ARMS = {
    "full": Arm(6, True, True, False, True),
    "no_length": Arm(7, True, False, False, True),
    "no_example": Arm(8, False, True, False, True),
    "plain": Arm(10, False, False, False, True),
    #: Sequential side, one call per utterance; fills the seq half of the 2x2x2.
    "sequential": Arm(9, True, True, False, True),
    "sequential_length": Arm(11, False, True, False, True),
    "sequential_example": Arm(12, True, False, False, True),
    #: Own persona only. Sequential generation calls once per speaker, so each call
    #: can be given one persona; batch generation writes both speakers at once and
    #: cannot. Paired against the sequential arms to differ in one factor.
    "sequential_own": Arm(13, True, True, True, True),
    "sequential_plain_own": Arm(14, False, False, True, True),
    "sequential_length_own": Arm(15, False, True, True, True),
    #: No topic advice; only the length and output-format rules remain. The
    #: cross-model comparison uses this condition. The namespace here is for haiku;
    #: other models are given their own with --namespace.
    "sequential_length_own_no_hint": Arm(17, False, True, True, False),
    #: The same condition with the prompt rewritten for sequential generation
    #: (revision 2). Revision 1 reused the batch instructions, so the system message
    #: asked for a whole conversation while the user message asked for one utterance.
    "sequential_length_own_no_hint_v2": Arm(20, False, True, True, False, revision=2),
    #: Differs from the previous arm in the examples alone, so the effect of showing
    #: examples is a one-factor difference. In revision 2 each dialogue draws two
    #: real dialogues at random, in full and without personas.
    "sequential_example_length_own_no_hint_v2": Arm(23, True, True, True, False, revision=2),
}
#: Arms generated one utterance per call, through the step subcommand.
SEQUENTIAL_ARMS = {"sequential", "sequential_length", "sequential_example",
                   "sequential_own", "sequential_plain_own", "sequential_length_own",
                   "sequential_length_own_no_hint", "sequential_length_own_no_hint_v2",
                   "sequential_example_length_own_no_hint_v2"}

#: The namespaces in ARMS are for claude-haiku-4-5. Running the same condition on
#: another model would collide in dialogue_id, so pass --namespace.

_INTRO_WITH_EXAMPLES = (
    "あなたは日本語のテキストチャットで交わされる雑談を作る人です。\n"
    "実際の人間同士のチャットの例をいくつか示すので、同じ調子の雑談を1本作ってください。"
)
_INTRO_NO_EXAMPLES = (
    "あなたは日本語のテキストチャットで交わされる雑談を作る人です。\n"
    "テキストチャットらしい雑談を1本作ってください。"
)
_LENGTH_RULES = [
    "- 1発話は10〜40文字程度。平均は20文字台におさめる。",
    "- 1発話に複数の話題や長い説明を詰めない。チャットなので短くやり取りする。",
]
#: Topic advice. Dropped where the model's unprompted behavior is wanted.
_TOPIC_HINT = "- 相手の話を受けつつ、実際の雑談のように話題が移ったり戻ったりしてよい。"
#: Output format. Always included; without it speaker labels leak into the text.
_FORMAT_RULE = "- 話者名やラベル、鉤括弧は付けず、発話の本文だけを返す。"

#: Revision 2, rewritten for sequential generation. The first line keeps the role
#: of revision 1 and the second states the actual task, producing the next single
#: utterance. The length rule drops the character bounds in favour of a sentence
#: count with characters as a guide, and says nothing about topics.
_INTRO_STEP_V2 = (
    "あなたは日本語のテキストチャットで交わされる雑談を作る人です。\n"
    "対話の途中までを示すので、指示された話者の次の1発話を作ってください。"
)
#: Revision 2 with examples. It differs from the version without them only in
#: announcing the examples and asking for the same register; the task description
#: and the rules are identical.
_INTRO_STEP_V2_EXAMPLES = (
    "あなたは日本語のテキストチャットで交わされる雑談を作る人です。\n"
    f"実際の人間同士のチャットの例を{N_SHOTS}本示したあと、途中までの対話を示します。\n"
    "同じ調子で、指示された話者の次の1発話を作ってください。"
)
_LENGTH_RULE_V2 = (
    "- 1発話は1〜2文程度の簡潔な発話にする(20字前後が目安)。"
    "チャットなので短くやり取りする。"
)


def build_system(examples: bool, length: bool, topic_hint: bool = True,
                 revision: int = 1) -> str:
    if revision == 2:
        intro = _INTRO_STEP_V2_EXAMPLES if examples else _INTRO_STEP_V2
        rules = ([_LENGTH_RULE_V2] if length else []) \
            + ([_TOPIC_HINT] if topic_hint else []) + [_FORMAT_RULE]
        return intro + "\n\n守ること:\n" + "\n".join(rules)
    intro = _INTRO_WITH_EXAMPLES if examples else _INTRO_NO_EXAMPLES
    rules = (_LENGTH_RULES if length else []) \
        + ([_TOPIC_HINT] if topic_hint else []) + [_FORMAT_RULE]
    return intro + "\n\n守ること:\n" + "\n".join(rules)


#: The default system message, referenced from the other modules.
SYSTEM = build_system(examples=True, length=True)


def _persona_block(label: str, persona: list[str]) -> str:
    if not persona:  # the no-persona condition omits the block entirely
        return ""
    return f"話者{label}のプロフィール:\n" + "\n".join(f"- {s}" for s in persona)


def _transcript(texts: list[str]) -> str:
    return "\n".join(f"話者{'A' if i % 2 == 0 else 'B'}: {t}" for i, t in enumerate(texts))


def _example(path: Path, personas: dict[str, list[str]]) -> tuple[set[str], str] | None:
    """One real dialogue as (the pair of speakers, the example text with personas)."""
    raw = json.loads(path.read_text())
    if int(path.stem) != raw["dialogue_id"]:  # duplicate file, as in iter_dialogues
        return None
    d = load_dialogue(path)
    ids = raw.get("interlocutors", [])
    if len(ids) < 2 or any(i not in personas for i in ids[:2]):
        return None
    a_id = d.utterances[0].speaker_id
    b_id = next((i for i in ids if i != a_id), None)
    if b_id is None or a_id not in personas:
        return None
    texts = [u.text for u in d.utterances[:SHOT_TURNS]]
    if len(texts) < SHOT_TURNS:
        return None
    return {a_id, b_id}, "\n\n".join([
        _persona_block("A", personas[a_id]),
        _persona_block("B", personas[b_id]),
        _transcript(texts),
    ])


def build_examples_random(
    exclude: set[int], personas: dict[str, list[str]]
) -> list[tuple[set[str], str]]:
    """Example candidates for revision 2: scan every dialogue and filter only.

    Three things differ from build_examples, used by revision 1.

    - No stride sampling by id. The caller draws at random instead. Stride sampling
      was a cheap deterministic way to pick from 13,000 dialogues, but 87.7% of
      id-adjacent RealPersonaChat dialogues share a speaker, so the draw should not
      depend on the ordering.
    - No personas. The examples are bare conversation. The target side shows only
      the speaker's own persona, so showing both personas in an example would make
      the two asymmetric.
    - No truncation, so an example includes how its conversation ends.

    Each element is (the pair of speakers, the example text). The pair is needed to
    reject, at assignment time, an example featuring the two speakers of the target
    dialogue, which would leak a real conversation between them.
    """
    out: list[tuple[set[str], str]] = []
    for path in sorted(config.dialogues_dir().glob("*.json")):
        raw = json.loads(path.read_text())
        if int(path.stem) != raw["dialogue_id"]:  # duplicate file
            continue
        if raw["dialogue_id"] in exclude:
            continue
        ids = raw.get("interlocutors") or []
        utts = raw.get("utterances") or []
        if not utts or len(ids) < 2 or any(i not in personas for i in ids[:2]):
            continue
        out.append(({ids[0], ids[1]}, _transcript([u["text"] for u in utts])))
    return out


def build_examples(
    exclude: set[int], personas: dict[str, list[str]], need: int
) -> list[tuple[set[str], str]]:
    """Example candidates from dialogues outside the seed set, by stride over ids.

    Also returns the pair of speakers: another dialogue between the same pair would
    show a real conversation between the target speakers, so it has to be rejected
    at assignment time.
    """
    paths = sorted(config.dialogues_dir().glob("*.json"))
    cand = [p for p in paths if int(p.stem) not in exclude]
    stride = max(1, len(cand) // need)
    out: list[tuple[set[str], str]] = []
    for p in cand[::stride]:
        ex = _example(p, personas)
        if ex is not None:
            out.append(ex)
        if len(out) >= need:
            break
    return out


def build_head(
    seed: dict, personas: dict[str, list[str]], shots: list[str], own: str | None = None,
    revision: int = 1,
) -> str:
    """Everything up to the dialogue itself: the examples and the personas.

    own "A" or "B" includes that speaker's persona only; the default includes both.
    With no examples, revision 2 drops the heading, which exists to separate the
    examples from the target and has nothing to separate without them.
    """
    parts = []
    for i, ex in enumerate(shots, 1):
        parts += [f"[例{i}: 実際の人間同士のチャット]", ex, ""]
    if shots:
        parts.append("[ここから作ってもらう対話]")
    elif revision < 2:
        parts.append("[作ってもらう対話]")
    labels = ("A", "B") if own is None else (own,)
    blocks = [_persona_block(label, personas[seed[f"interlocutor_{label.lower()}"]]) for label in labels]
    for i, block in enumerate(b for b in blocks if b):
        if i:
            parts.append("")
        parts.append(block)
    return "\n".join(parts)


def build_prompt(seed: dict, personas: dict[str, list[str]], shots: list[str]) -> str:
    """Batch generation: show the opening utterance and ask for the rest at once."""
    return "\n".join([
        build_head(seed, personas, shots),
        "",
        f"話者A: {seed['opening']}",
        "",
        f"この続きを{TURNS - 1}発話ぶん作ってください。話者Bから始まり、"
        "話者Bと話者Aが交互に話します(最後は話者B)。"
        f"{TURNS - 1}個の発話本文を順番どおりに返してください。",
    ])


def build_step_prompt(head: str | dict[str, str], opening: str, said: list[str],
                      revision: int = 1) -> str:
    """Sequential generation: show the dialogue so far and ask for one utterance.

    said holds the utterances generated after the opening one, in order. Unlike
    batch generation the model writes one utterance at a time, without seeing how
    the conversation will continue.

    A head of {"A": ..., "B": ...} selects the head of whichever speaker is about
    to talk, for the own-persona-only condition.
    """
    texts = [opening, *said]
    nxt = "A" if len(texts) % 2 == 0 else "B"
    tail = ("発話の本文だけを返してください。" if revision >= 2
            else "発話の本文だけを返します。")
    return "\n".join([
        head[nxt] if isinstance(head, dict) else head,
        "",
        "[ここまでの対話]",
        _transcript(texts),
        f"話者{nxt}:",
        "",
        f"話者{nxt}として、次の1発話だけを作ってください。{tail}",
    ]).lstrip("\n")  # drop the leading blank line where there is no persona block


def cmd_prompts(args: argparse.Namespace) -> None:
    arm = ARMS[args.arm]
    namespace = getattr(args, "namespace", None) or arm.namespace
    use_examples, own_only = arm.examples, arm.own_only
    personas = load_personas()
    # The reduced-persona conditions draw k items per speaker at random, to see
    # what novelty does once the supplied items run out.
    k_items = getattr(args, "persona_items", None)
    if k_items is not None:
        # k = 0 leaves the persona block out altogether.
        prng = random.Random(getattr(args, "seed_rng", 0))
        personas = {sp: prng.sample(items, k_items) if len(items) > k_items else list(items)
                    for sp, items in sorted(personas.items())}
    if getattr(args, "seed_selection", "stride") == "random":
        # Simple random sample over all dialogues, so it returns exactly n.
        seeds = pick_seed_dialogues_random(args.n, getattr(args, "seed_rng", 0))
    else:
        seeds = [s for s in pick_seed_dialogues(SEED_POOL)
                 if s["interlocutor_a"] in personas
                 and s["interlocutor_b"] in personas][:args.n]
    # Exclude the seed dialogues from the example candidates: using a seed dialogue
    # as its own example would hand the model the continuation. Independent of --n,
    # so a small trial run and the real run see the same candidates.
    seed_rng = getattr(args, "seed_rng", 0)
    exclude = {s["seed_dialogue_id"]
               for s in pick_seed_dialogues_random(SEED_POOL, seed_rng)} \
        | {s["seed_dialogue_id"] for s in pick_seed_dialogues(SEED_POOL)}

    if arm.revision >= 2:
        # Draw N_SHOTS examples per dialogue. A single fixed pair would confound the
        # effect of showing examples with the effect of that particular pair. The
        # seed is fixed, so the same arguments give the same assignment.
        cand = build_examples_random(exclude, personas)
        rng = random.Random(seed_rng)
    else:
        # A different pair per dialogue. Extra candidates are built because examples
        # sharing a speaker with the target are skipped. The draw runs even for arms
        # without examples, so the cursor advances identically.
        cand = build_examples(exclude, personas, POOL_SIZE)
        rng, cursor = None, 0

    out = []
    for i, s in enumerate(seeds):
        own = {s["interlocutor_a"], s["interlocutor_b"]}
        shots = []
        while len(shots) < N_SHOTS:
            if rng is not None:
                ids, text = rng.choice(cand)
            else:
                if cursor >= len(cand):
                    raise SystemExit(f"ran out of example candidates at dialogue {i + 1}")
                ids, text = cand[cursor]
                cursor += 1
            if not (ids & own):  # skip examples featuring either target speaker
                shots.append(text)
        if not use_examples:
            shots = []
        out.append({
            "dialogue_id": namespace * 100000 + i + 1,
            "seed_dialogue_id": s["seed_dialogue_id"],
            "interlocutor_a": s["interlocutor_a"],
            "interlocutor_b": s["interlocutor_b"],
            "opening": s["opening"],
            "prompt": build_prompt(s, personas, shots),
            # In sequential generation the step subcommand builds each single-turn
            # prompt from the head; own-persona-only needs one head per speaker.
            "head": {"A": build_head(s, personas, shots, own="A", revision=arm.revision),
                     "B": build_head(s, personas, shots, own="B", revision=arm.revision)}
            if own_only else build_head(s, personas, shots, revision=arm.revision),
        })
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    chars = sum(len(r["prompt"]) for r in out) // len(out)
    print(f"wrote {args.out} (arm={args.arm}, ns{namespace}, {len(out)} dialogues, "
          f"prompts averaging {chars:,} characters)")
    print(f"system:\n{build_system(arm.examples, arm.length, arm.topic_hint, arm.revision)}")


def cmd_step(args: argparse.Namespace) -> None:
    """Write the prompt for one step of sequential generation to standard output.

    The utterances generated so far are read from dir/{dialogue_id}/{k:02d}.txt,
    with k starting at 1.
    """
    arm = ARMS[args.arm]
    rec = next(r for r in json.loads(Path(args.prompts).read_text())
               if r["dialogue_id"] == args.dialogue_id)
    d = Path(args.dir) / str(args.dialogue_id)
    said = [(d / f"{k:02d}.txt").read_text().strip()
            for k in range(1, TURNS) if (d / f"{k:02d}.txt").is_file()]
    print(build_step_prompt(rec["head"], rec["opening"], said))
    print()
    print("[system message]")
    print(build_system(arm.examples, arm.length, arm.topic_hint, arm.revision))


def cmd_assemble(args: argparse.Namespace) -> None:
    arm = ARMS[args.arm]
    namespace = getattr(args, "namespace", None) or arm.namespace
    use_examples, use_length, own_only = arm.examples, arm.length, arm.own_only
    model = getattr(args, "model", None) or MODEL
    turns = getattr(args, "turns", None) or TURNS  # cells generated at another length
    prompts = {r["dialogue_id"]: r for r in json.loads(Path(args.prompts).read_text())}
    replies = json.loads(Path(args.input).read_text())

    dialogues = []
    for did, utts in sorted((int(k), v) for k, v in replies.items()):
        meta = prompts[did]
        if len(utts) == turns:  # one utterance too many: drop the last, keeping A/B alternation
            print(f"  dialogue {did}: had {turns} utterances, dropping the last")
            utts = utts[:turns - 1]
        texts = [meta["opening"], *utts]
        if len(texts) != turns:
            raise SystemExit(f"dialogue {did}: {len(texts)} utterances, expected {turns}")
        tag = "".join(["_fewshot" if use_examples else "", "_len" if use_length else "",
                       "_ownprofile" if own_only else "",
                       "" if arm.topic_hint else "_nohint"])
        unit = "seq" if args.arm in SEQUENTIAL_ARMS else "oneshot"
        dialogues.append({
            "dialogue_id": did,
            "system_id": f"{model}_{unit}{tag}_persona",
            "interlocutor_id": f"{meta['interlocutor_a']}+{meta['interlocutor_b']}",
            "seed_dialogue_id": meta["seed_dialogue_id"],
            "utterances": [
                {"utterance_id": i, "interlocutor": "A" if i % 2 == 0 else "B", "text": t}
                for i, t in enumerate(texts)
            ],
        })

    # Written into a subdirectory, not the parent: iter_llm_llm_dialogues globs the
    # parent non-recursively, so placing these here would silently fold them into
    # the older cells, whose generation conditions differ.
    out_dir = Path("data/generated/llm_llm/oneshot")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{model.replace('/', '_')}__oneshot__ns{namespace}"
    (out_dir / f"{name}.json").write_text(json.dumps(dialogues, ensure_ascii=False, indent=1))
    (out_dir / f"manifest__{name}.json").write_text(json.dumps({
        "model": model,
        "arm": args.arm,
        "namespace": namespace,
        "n_dialogues": len(dialogues),
        "turns": turns,
        "generation": ("one call per utterance (sequential). " if args.arm in SEQUENTIAL_ARMS
                       else "one call per dialogue (batch). ") + args.via,
        "few_shot": (
            (f"{N_SHOTS} real RealPersonaChat dialogues, in full and without personas, "
             "drawn at random per dialogue. Seed dialogues, and any dialogue "
             "featuring the two target speakers, are never used as examples"
             if arm.revision >= 2 else
             f"{N_SHOTS} real RealPersonaChat dialogues, with personas and cut to "
             f"{SHOT_TURNS} utterances, varied per dialogue. Seed dialogues are "
             "never used as examples")
            if use_examples else "none"),
        "length_control": ("one or two sentences per utterance, around 20 characters"
                           if arm.revision >= 2 else "10 to 40 characters per utterance, averaging in the twenties")
                          if use_length else "none",
        "prompt_revision": arm.revision,
        "temperature": getattr(args, "temperature", None),
        "profile_visibility": "only the speaker's own persona; the partner's is never shown"
                              if own_only else "both speakers' personas",
        "profile_items": ("all 10 items" if getattr(args, "persona_items", None) is None
                          else "no persona block" if args.persona_items == 0
                          else f"{args.persona_items} of the 10 items, drawn per speaker with seed_rng"),
        "topic_hint": "included" if arm.topic_hint else "not included",
        "seed_pool": (
            f"{len(dialogues)} drawn at random with seed {getattr(args, 'seed_rng', 0)} "
            "from the 13,581 dialogues left after removing duplicate files and "
            "dialogues missing a persona for either speaker"
            if getattr(args, "seed_selection", "stride") == "random"
            else f"the first {len(dialogues)} of pick_seed_dialogues({SEED_POOL})"
        ),
        "system_prompt": build_system(use_examples, use_length, arm.topic_hint,
                                      arm.revision),
        # Call statistics: retries on an empty reply, violations of the output
        # format, and so on. Recorded only when the generator passed them in.
        **({"call_stats": args.call_stats} if getattr(args, "call_stats", None) else {}),
    }, ensure_ascii=False, indent=1))
    lens = [len(u["text"]) for d in dialogues for u in d["utterances"]]
    print(f"wrote {out_dir / f'{name}.json'} ({len(dialogues)} dialogues)")
    print(f"utterance length: mean {sum(lens) / len(lens):.1f} characters (human--human is 22.9)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    arm_help = "the generation condition; see ARMS"
    p = sub.add_parser("prompts", help="build the prompts")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--arm", default="full", choices=list(ARMS), help=arm_help)
    p.add_argument("--namespace", type=int, default=None,
                   help="namespace of dialogue_id; defaults to the arm's")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(func=cmd_prompts)
    a = sub.add_parser("assemble", help="assemble the replies into a dataset")
    a.add_argument("-i", "--input", required=True, help="JSON of {dialogue_id: [utterances]}")
    a.add_argument("--prompts", required=True)
    a.add_argument("--arm", default="full", choices=list(ARMS), help=arm_help)
    a.add_argument("--via", default="",
                   help="how the model was called; recorded in the manifest")
    a.add_argument("--model", default=None, help="the model used; defaults to claude-haiku-4-5")
    a.add_argument("--namespace", type=int, default=None,
                   help="namespace of dialogue_id; defaults to the arm's")
    a.set_defaults(func=cmd_assemble)
    st = sub.add_parser("step", help="print the prompt for one sequential step")
    st.add_argument("--prompts", required=True)
    st.add_argument("--dir", required=True, help="working directory of the generated utterances")
    st.add_argument("--dialogue-id", type=int, required=True)
    st.add_argument("--arm", default="sequential", choices=list(ARMS), help=arm_help)
    st.set_defaults(func=cmd_step)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
