# Code and generated dialogues

Code, generated dialogues and per-dialogue values for the paper
*Why Does Human Conversation Stay Novel? A Turn-Level Decomposition of Human and LLM--LLM Dialogue*
(IEEE ICA 2026).

## Contents

| Path | What it is |
| --- | --- |
| `metrics/per_series.csv` | Per (dialogue, role) means of the five atoms and the underlying cosines, and the within-conversation novelty slope, for every group in the paper |
| `metrics/per_turn_llm.csv` | The same quantities per turn for the LLM cells |
| `metrics/groups.csv` | Group names and their generation conditions |
| `data/generated/llm_llm/oneshot/` | The generated LLM--LLM dialogues used in the paper, one JSON file per cell, with the generation manifest (prompt text, decoding settings, seed pool) alongside. See `DATA_LICENSE.md` |
| `src/`, `scripts/` | Analysis and generation code (MIT). `scripts/compute_pid.py` computes the decomposition; `scripts/generate_llm_llm_sequential.py` generated the dialogues |

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The human--human corpus
(RealPersonaChat) is not redistributed here, so clone it separately:

```bash
git clone https://github.com/nu-dialogue/real-persona-chat.git
uv sync
cp .env.example .env
```

Then edit `.env`: put an OpenAI API key in `OPENAI_API_KEY` and point `RPC_DATA_DIR`
at the RealPersonaChat clone.

## Reproducing the figures and numbers

First embed both corpora and compute the per-turn quantities. The two embedding
steps call the OpenAI API; the rest is local.

```bash
uv run python scripts/embed_human.py
uv run python scripts/embed_llm_llm.py --corpus llm_llm_oneshot
uv run python scripts/novelty_dynamics.py
uv run python scripts/novelty_dynamics.py --corpus llm_llm_oneshot
uv run python scripts/compute_pid.py
uv run python scripts/compute_pid.py --corpus llm_llm_oneshot
```

Each figure and table of the paper is then produced by one command:

| In the paper | Command | Where it lands |
| --- | --- | --- |
| Fig. 1 (decomposition profiles) | `uv run python scripts/decomposition_profile_figure.py` | `outputs/figures/decomposition_profile_main.png` |
| Fig. 2 (novelty over utterance index) | `uv run python scripts/model_comparison_figures.py --lang en` | `outputs/figures/model_comparison_150_trend.png`, alongside three further figures |
| Fig. 3 (continuation to 60 utterances) | `uv run python scripts/persona_items_continue.py --lang en` | `outputs/figures/persona_items_continue.png` |
| Table I (mean novelty and slope) | `uv run python scripts/model_comparison.py` | standard output |
| `metrics/*.csv` | `uv run python scripts/export_supplement_metrics.py --out metrics` | the CSVs in this repository |

The CSVs already in `metrics/` were produced by the last command and can be read
without recomputing anything.
