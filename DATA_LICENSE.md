# License of the generated dialogues

The files in `data/` are LLM--LLM dialogues generated for this paper. Each dialogue
was seeded with the opening utterance and the two speakers' personas of a real
dialogue from **RealPersonaChat** (Yamashita et al., PACLIC 2023;
https://github.com/nu-dialogue/real-persona-chat), which is released under
**CC BY-SA 4.0**. Utterance 0 of every generated dialogue is that opening utterance
verbatim, and `seed_dialogue_id` / `interlocutor_id` refer to the RealPersonaChat
dialogue and speakers; the persona text itself is not stored here.

The generated dialogues are therefore a derivative work of RealPersonaChat and are
released under the same license, **CC BY-SA 4.0**
(https://creativecommons.org/licenses/by-sa/4.0/). Redistribution must keep the
attribution to RealPersonaChat and this license.

The code in `src/` and `scripts/` is released separately under the MIT license
(`LICENSE`).
