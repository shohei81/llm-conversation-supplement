"""Generate the LLM--LLM dialogues used in the paper, one call per utterance.

Claude models go to the Anthropic Messages API and the rest to OpenAI Chat
Completions, in both cases as two messages, system and user, so both providers
receive exactly the same strings. Temperature scales differ, however: Anthropic
takes 0 to 1 and OpenAI 0 to 2, and the default of 1.0 is each provider's own
default rather than the same setting.

One independent call per utterance means

- the context resets every time, which is what makes the own-persona-only
  condition real. Playing A and B alternately inside one context would let a
  speaker read the partner's persona from the partner's own prompt.
- the system message carries only the rules from build_system().
- temperature can be set explicitly.

Each model needs its own dialogue_id namespace, since the same condition on
another model is different data. The namespace in the arm is for Claude, so other
models must pass --namespace.

Run from the repository root:
    uv run python scripts/generate_llm_llm_sequential.py \
        --arm sequential_length_own_no_hint_v2 --n 150 --seed-selection random
    uv run python scripts/generate_llm_llm_sequential.py --arm sequential_length_own_no_hint \
        --model gpt-5-nano --namespace 19 --n 50 --workers 3
    uv run python scripts/generate_llm_llm_sequential.py --arm sequential_own --n 2 --dry-run

Utterances are saved as they are generated, under
outputs/generate_sequential/{arm}__{model}/{dialogue_id}.json, so the same command
resumes an interrupted run. Call statistics accumulate in stats.json beside them
and stay cumulative across resumes. Once every dialogue is complete the dataset and
its manifest are written into data/generated/llm_llm/oneshot/.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from semantic_closure import config

from generate_llm_llm import _clean_utterance
from generate_llm_llm_oneshot import (
    ARMS,
    MODEL,
    SEQUENTIAL_ARMS,
    TURNS,
    build_step_prompt,
    build_system,
    cmd_assemble,
    cmd_prompts,
)

#: How the model was called, recorded in the manifest.
VIA_ANTHROPIC = "one call per utterance to the Anthropic Messages API, as a system and a user message"
VIA_OPENAI = "one call per utterance to OpenAI Chat Completions, as a system and a user message"

_CLIENT = None  # per-provider client, built in main

MAX_TOKENS = 300  # output cap per utterance
#: Retries of the same prompt on an empty reply, so 1 + EMPTY_RETRIES attempts.
EMPTY_RETRIES = 3


def reasoning_effort(model: str) -> str | None:
    """The setting that turns reasoning off, which differs by model.

    The gpt-5 models are reasoning models and by default spend around 960 reasoning
    tokens on a single utterance, sometimes returning an empty reply. Reasoning is
    turned off to match the Claude side, which runs without extended thinking.
    Valid values differ by model, and an unsupported or invalid one returns 400.
    """
    if model.startswith("gpt-5.6"):
        return "none"
    if model.startswith("gpt-5"):
        return "minimal"
    return None


#: Thresholds for a degenerate reply. At temperature 1.0 a model sometimes repeats
#: one expression indefinitely; one observed utterance ran 598 characters, repeating
#: the same two-character word 299 times. Normal utterances repeat at most 3 times
#: and reach 48 characters at the 90th percentile, so these thresholds do not catch
#: them.
DEGENERATE_LENGTH = 150
DEGENERATE_REPEATS = 8


def _max_repeat(text: str, max_unit: int = 4) -> int:
    """Largest number of consecutive repetitions of one short unit."""
    best = 1
    for n in range(1, max_unit + 1):
        i = 0
        while i + n <= len(text):
            unit = text[i:i + n]
            k = 1
            while text[i + k * n:i + (k + 1) * n] == unit:
                k += 1
            best = max(best, k)
            i += 1
    return best


def is_degenerate(text: str) -> bool:
    """Whether the reply is degenerate. It would distort the embedding badly."""
    return len(text) > DEGENERATE_LENGTH or _max_repeat(text) >= DEGENERATE_REPEATS


def drop_degenerate_dialogues(records: list[dict], out_dir: Path) -> int:
    """Discard the partial results of any dialogue holding a degenerate utterance.

    The whole dialogue is regenerated rather than that one utterance: replacing it
    alone would leave the utterances after it as replies to the discarded text.
    """
    dropped = 0
    for rec in records:
        path = out_dir / f"{rec['dialogue_id']}.json"
        if not path.is_file():
            continue
        if any(is_degenerate(t) for t in json.loads(path.read_text())):
            path.unlink()
            dropped += 1
    return dropped


def is_claude(model: str) -> bool:
    return model.startswith("claude")


def thinking_off(model: str) -> bool:
    """Whether thinking has to be disabled explicitly for this model.

    On the newer Claude models, leaving thinking unset enables adaptive thinking and
    the amount varies per utterance. Turning it off matches the reasoning-off
    setting on the gpt-5 side and claude-haiku-4-5, which does not think unless
    asked. Nothing is sent for haiku, so its requests stay as they were for the
    cells already generated.
    """
    return is_claude(model) and model != "claude-haiku-4-5"


_THINKING_TAG = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL)


def work_dir(key: str) -> Path:
    """Where partial results live, one directory per (arm, model)."""
    return config.OUTPUT_DIR / "generate_sequential" / key


def openai_client():
    """OpenAI client for generation, preferring CHAT_API_KEY over OPENAI_API_KEY."""
    import os

    from openai import OpenAI

    api_key = os.environ.get("CHAT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Neither CHAT_API_KEY nor OPENAI_API_KEY is set in .env.")
    # Leave 429 (token-per-minute limit) to the client's backoff, and lower --workers.
    return OpenAI(api_key=api_key, timeout=60.0, max_retries=8)


def anthropic_client():
    """Anthropic client for generation; ANTHROPIC_API_KEY comes from .env."""
    import os

    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set in .env.")
    # Leave 429 and 5xx to the SDK's backoff.
    return anthropic.Anthropic(timeout=60.0, max_retries=8)


def call_openai(prompt: str, system: str, model: str,
                temperature: float) -> tuple[str, int, int]:
    """The reply text, and the input and output token counts."""
    eff = reasoning_effort(model)
    extra = {"reasoning_effort": eff} if eff else {}
    resp = _CLIENT.chat.completions.create(
        model=model, temperature=temperature, max_completion_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        **extra,
    )
    u = resp.usage
    return (resp.choices[0].message.content or "",
            u.prompt_tokens if u else 0, u.completion_tokens if u else 0)


def call_anthropic(prompt: str, system: str, model: str,
                   temperature: float) -> tuple[str, int, int]:
    """The reply text, and the input and output token counts."""
    # temperature is gone from messages.create() in SDK 1.x, because the newer
    # models no longer take sampling parameters. The API itself still accepts it for
    # claude-haiku-4-5, so it goes through extra_body; it does take effect (six calls
    # gave one distinct reply at 0.0 and four at 1.0). The newer models accept only
    # the default of 1.0 and return 400 for anything else.
    kw = {"thinking": {"type": "disabled"}} if thinking_off(model) else {}
    resp = _CLIENT.messages.create(
        model=model, max_tokens=MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"temperature": temperature}, **kw,
    )
    return ("".join(b.text for b in resp.content if b.type == "text"),
            resp.usage.input_tokens, resp.usage.output_tokens)


class Stats:
    """Call statistics, guarded by a lock because worker threads update it.

    Existing counts are read from stats.json first, so a resumed run stays cumulative.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        prev = json.loads(path.read_text()) if path.is_file() else {}
        self._c = collections.Counter(prev)

    def bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._c[key] += n

    def note_format(self, raw: str, cleaned: str) -> None:
        """Count replies that broke the output format: a speaker label, or several lines.

        _clean_utterance silently fixes both, so they are recorded before it runs.
        How often a model breaks the format under the same prompt is itself part of
        the comparison between models.
        """
        lines = raw.strip().splitlines()
        if len(lines) > 1:
            self.bump("multiline")
        if lines and cleaned != lines[0].strip():
            self.bump("label_prefixed")

    def save(self) -> dict[str, int]:
        with self._lock:
            d = dict(sorted(self._c.items()))
        self._path.write_text(json.dumps(d, ensure_ascii=False, indent=1))
        return d


def generate_dialogue(rec: dict, system: str, model: str, out_dir: Path,
                      temperature: float, revision: int, stats: Stats,
                      turns: int = TURNS) -> list[str]:
    """Generate turns - 1 utterances of one dialogue, saving after each so it resumes."""
    path = out_dir / f"{rec['dialogue_id']}.json"
    said: list[str] = json.loads(path.read_text()) if path.is_file() else []
    while len(said) < turns - 1:
        speaker = "A" if len(said) % 2 == 1 else "B"  # utterance 0 is A, so B is next
        prompt = build_step_prompt(rec["head"], rec["opening"], said, revision)
        text = ""
        for attempt in range(EMPTY_RETRIES + 1):
            raw, tok_in, tok_out = (
                call_anthropic(prompt, system, model, temperature) if is_claude(model)
                else call_openai(prompt, system, model, temperature))
            stats.bump("calls")
            stats.bump("input_tokens", tok_in)
            stats.bump("output_tokens", tok_out)
            if _THINKING_TAG.search(raw):
                # With thinking disabled, a model can still leak a <thinking> tag
                # into the reply. Count it as a format violation and strip it.
                stats.bump("thinking_tag")
                raw = _THINKING_TAG.sub("", raw)
            text = _clean_utterance(raw, speaker)
            stats.note_format(raw, text)
            if text:
                break
            # Empty reply: send the same prompt again, which can differ at temperature > 0.
            stats.bump("empty_retry")
        if not text:
            stats.bump("empty_failed")  # still empty; fill in to keep the length
        said.append(text or "……")
        stats.bump("utterances")
        path.write_text(json.dumps(said, ensure_ascii=False, indent=1))
    return said


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(SEQUENTIAL_ARMS),
                        help="the generation condition, from ARMS in generate_llm_llm_oneshot.py")
    parser.add_argument("--n", type=int, default=50, help="number of dialogues to generate")
    parser.add_argument("--model", default=MODEL,
                        help="claude* goes to the Anthropic Messages API, anything else to OpenAI")
    parser.add_argument("--namespace", type=int, default=None,
                        help="namespace of dialogue_id; defaults to the arm's, and must change per model")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="each provider's own default; the scales differ, "
                             "0 to 1 for Anthropic and 0 to 2 for OpenAI")
    parser.add_argument("--seed-selection", choices=("stride", "random"), default="stride",
                        help="how seeds are chosen: stride over ids, or a simple "
                             "random sample with a fixed seed")
    parser.add_argument("--seed-rng", type=int, default=0,
                        help="seed of the draw, with --seed-selection random")
    parser.add_argument("--workers", type=int, default=8, help="dialogues generated in parallel")
    parser.add_argument("--persona-items", type=int, default=None,
                        help="persona items to supply, all 10 by default; a smaller number needs a new --namespace")
    parser.add_argument("--turns", type=int, default=TURNS,
                        help=f"utterances per dialogue, the human seed included "
                             f"(default {TURNS}); another value needs a new --namespace")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the prompt for one utterance and stop")
    args = parser.parse_args()

    arm = ARMS[args.arm]
    namespace, own_only = arm.namespace, arm.own_only
    if args.namespace is not None:
        namespace = args.namespace
    elif not is_claude(args.model):
        raise SystemExit(
            f"--model {args.model} needs an explicit --namespace: the arm's default "
            f"ns{namespace} is for Claude and would collide.")
    system = build_system(arm.examples, arm.length, arm.topic_hint, arm.revision)
    out_dir = work_dir(f"{args.arm}__{args.model}"
                       + (f"__t{args.turns}" if args.turns != TURNS else "")
                       + (f"__p{args.persona_items}" if args.persona_items is not None else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = out_dir / "prompts.json"
    cmd_prompts(argparse.Namespace(arm=args.arm, n=args.n, out=str(prompts_path),
                                   namespace=namespace,
                                   seed_selection=args.seed_selection,
                                   seed_rng=args.seed_rng, persona_items=args.persona_items))
    records = json.loads(prompts_path.read_text())

    print(f"\nmodel {args.model}, arm {args.arm} (ns{namespace}), "
          f"{len(records)} dialogues, {args.turns} utterances, {args.workers} in parallel")
    print(f"persona: {'own only' if own_only else 'both speakers'}, "
          f"working directory {out_dir}")

    if args.dry_run:
        rec = records[0]
        print("\n--- prompt for the first utterance ---")
        print(build_step_prompt(rec["head"], rec["opening"], [], arm.revision))
        return

    from tqdm import tqdm

    global _CLIENT
    _CLIENT = anthropic_client() if is_claude(args.model) else openai_client()
    stats = Stats(out_dir / "stats.json")

    # Regenerate whole dialogues holding a degenerate reply; earlier runs may have one.
    dropped = drop_degenerate_dialogues(records, out_dir)
    if dropped:
        print(f"discarding {dropped} dialogues with a degenerate reply and regenerating them")
        for _ in range(dropped):
            stats.bump("degenerate_redone")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(generate_dialogue, r, system, args.model, out_dir,
                                   args.temperature, arm.revision, stats, args.turns)
                       for r in records]
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc=f"generating ({args.arm})"):
                f.result()  # raise here; the utterances generated so far stay on disk
    finally:
        counts = stats.save()  # keep the statistics even on an interrupt
    print(f"call statistics: {counts}")

    replies = {str(r["dialogue_id"]): json.loads(
        (out_dir / f"{r['dialogue_id']}.json").read_text()) for r in records}
    replies_path = out_dir / "replies.json"
    replies_path.write_text(json.dumps(replies, ensure_ascii=False, indent=1))

    via = (VIA_ANTHROPIC if is_claude(args.model) else VIA_OPENAI) \
        + f"(temperature {args.temperature}" \
        + (f", reasoning_effort {reasoning_effort(args.model)}"
           if reasoning_effort(args.model) else "") \
        + (", thinking disabled" if thinking_off(args.model) else "") \
        + ")"
    cmd_assemble(argparse.Namespace(
        input=str(replies_path), prompts=str(prompts_path), arm=args.arm, via=via,
        model=args.model, namespace=namespace, temperature=args.temperature,
        seed_selection=args.seed_selection, seed_rng=args.seed_rng, call_stats=counts,
        turns=args.turns, persona_items=args.persona_items))


if __name__ == "__main__":
    main()
