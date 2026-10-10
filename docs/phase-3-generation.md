# Phase 3 — Generation, Citation & Confidence

**Plan reference:** Notion, Phase 3 (Day 6–9). **Status: code complete; needs a
model to run for real** (see [Turning it on](#turning-it-on)).

```
5 chunks
  → grounded prompt   (cite [1], refuse rather than guess)
  → generate
  → parse claims      (sentence → citation ids)
  → verify            (judge each claim against its cited chunk)
  → score confidence  (retrieval + coverage + citation accuracy)
  → answer, or decline
```

## Files

| File | Role |
|---|---|
| `generation.py` | Prompt construction, `Claim` parsing, `GeneratedAnswer` |
| `verification.py` | `verify()`, `Verdict`, `VerificationReport` |
| `confidence.py` | `ConfidenceBreakdown`, composite scoring |
| `answering.py` | `AnswerPipeline` — the single entry point |
| `llm/` | `base.py` (the seam), `ollama.py`, `free_api.py`, `stub.py`, `registry.py` |

## The LLM seam

Everything that calls a model goes through `LLMClient` (Notion plan A.6):

```python
class LLMClient(Protocol):
    model: str
    def complete(self, system: str, prompt: str, *, temperature=0.0, max_tokens=1024) -> LLMResponse: ...
    def health(self) -> bool: ...
```

Implementations: `GeminiClient` (default, free tier), `GroqClient` (free tier),
`OllamaClient` (local), `StubLLM` (tests).

**The generator and the judge are configured independently.** That is the
plan's Appendix A.4 arrangement — whoever writes should not be the only one
grading — and it only works if the two are separate settings. Both currently
point at Gemini; pointing the judge elsewhere is a one-line `.env` change.

## Citation verification

The step most RAG systems skip. Generating `[1]` proves nothing: the model can
cite a chunk that does not support the claim, or cite a number that does not
exist.

Each claim is re-read against the chunk it points at, and a judge returns one
of:

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | The passage states or directly implies the claim |
| `PARTIAL` | Related but does not fully establish it |
| `UNSUPPORTED` | The passage does not support it |
| `DANGLING` | Cited a source number that does not exist |
| `UNCITED` | A factual sentence with no citation at all |

`DANGLING` is checked locally and never costs a judge call — pinned by
`test_verify_marks_dangling_citations_without_calling_the_judge`.

**An unparseable judge reply is scored `PARTIAL`, not `SUPPORTED`.** Failing
open there would silently inflate citation accuracy, which is exactly the
number the portfolio claim rests on.

Pleasantries are filtered before demanding a citation, so "I hope that helps."
does not count as an uncited claim.

## Confidence

Three signals, each catching a different failure:

| Signal | Weight | Catches |
|---|---|---|
| Retrieval | 0.35 | The corpus cannot answer this |
| Citation coverage | 0.25 | Model asserted things without pointing anywhere |
| Citation accuracy | 0.40 | Citations that do not hold up |

Reported separately as well as combined, because an answer can score well on
two and badly on the third. The composite drives one decision — answer or
decline — and the breakdown lets a human see why.

Below `min_answer_confidence` (0.35), the system declines and reports what it
*did* find rather than guessing. **An honest refusal is scored as a correct
non-answer, not a bad answer** — otherwise the metrics would push the system
toward guessing.

## Turning it on

Nothing here needs a model to be *tested*. Real answers need one, and the
default backend is **Gemini free tier**, for both generation and judging.

```bash
# 1. Get a key at https://aistudio.google.com/apikey
# 2. Paste it into .env:  GEMINI_API_KEY=AIzaSy...
python -m rag check                       # verifies key, model, live call
python -m rag ask "how do I roll back a deploy"
```

`check` reports failures in the order they actually occur — key present, key
valid, model reachable *with that key*, live call returns text — and names the
fix for each. If the configured model is not available to your account it lists
the ones that are.

Alternatives, same seam:

```bash
python -m rag ask "..." --backend ollama          # local, uses the GPU
python -m rag ask "..." --judge groq              # different grader
```

Without any key, `ask` warns and falls back to `StubLLM`, which extracts
sentences from the top chunk and cites them. Crude, but it exercises parsing,
verification and scoring end to end.

## Limitations

- **Never run against a real model.** No API key was available during the
  build. Every component is unit-tested and the wiring is verified against the
  stub, but the prompt has not been tuned on real output — expect to iterate
  on it.
- **Generator and judge are currently the same model.** Gemini grading Gemini
  is a weaker check than an independent grader. It is the pragmatic default for
  one free tier; switching the judge to Groq is a `.env` change, and comparing
  the two graders would itself be a result worth reporting.
- **Confidence weights (0.35/0.25/0.40) are a starting point**, not a measured
  result, and neither is the 0.35 decline threshold.
- **One judge call per citation.** A 5-citation answer costs 5 calls plus 1
  generation. A full 52-question, 3-strategy sweep is roughly 450 calls, which
  is a real constraint against any free tier — Groq's binding limit is ~8K
  tokens/min, and Gemini's is account-specific. Use `--limit` first.
