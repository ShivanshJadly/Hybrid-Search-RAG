# Phase 4 — Evaluation Framework

**Plan reference:** Notion, Phase 4 (Day 9–11). **Status: complete; numbers are
stub-generated until a real model is connected.**

## Files

| File | Role |
|---|---|
| `eval/golden_set.yaml` | 52 questions across four categories |
| `eval/dataset.py` | Loading + strict validation |
| `eval/metrics.py` | Metric computation and `EvalSummary` |
| `eval/runner.py` | `evaluate()`, `sweep()`, `write_report()` |

## The golden set

52 questions (the plan asks for 50+):

| Category | Count | What it tests |
|---|---|---|
| `lookup` | 36 | The answer sits in one place, stated plainly |
| `multihop` | 7 | Needs two or more facts, often across documents |
| `no_answer` | 5 | Plausible but unanswerable — correct behaviour is to decline |
| `ambiguous` | 4 | Underspecified — a good answer names the ambiguity |

Each row carries `expected_sources` (document names) so **retrieval is scored
independently of generation**. Matching on document name rather than chunk id
is deliberate: it means one golden set scores all three chunking strategies
without rewriting.

Validation is strict — an answerable question with no expected sources, or a
no-answer question that has them, raises rather than warns. A malformed golden
set produces plausible-looking metrics that are quietly wrong, which is worse
than a crash.

## Metrics

| Metric | Needs a judge | Measures |
|---|---|---|
| Retrieval recall | no | Did the right document reach the context window? |
| Retrieval precision | no | What share of retrieved chunks were from expected docs? |
| Correctness | yes | Does the answer match the reference? |
| Faithfulness | yes | Is it grounded in what was retrieved, correct or not? |
| Citation accuracy | yes | Do the inline citations hold up? |
| Refusal accuracy | no | Did it decline the unanswerable questions? |
| **False refusal rate** | no | Did it decline things it *should* have answered? |

Correctness and faithfulness are separate on purpose: an answer can be correct
from the model's own memory while being ungrounded in the retrieved context.
That is a hallucination that happened to be right, and it will not stay right.

**False refusal rate exists to stop the metric set being gamed.** Without it,
"decline everything" scores 100% on refusal accuracy and 100% on faithfulness.
Pinned by `test_false_refusal_rate_penalises_always_declining`.

`--no-grade` skips the two judge-backed metrics, which keeps a fast
retrieval-only sweep available and helps stay inside a free-tier quota.

## Running it

```bash
python -m rag eval                       # all strategies, all questions
python -m rag eval -s header -n 10       # one strategy, first 10
python -m rag eval --no-grade            # retrieval metrics only
python -m rag eval --judge groq          # free-tier judge
```

Reports are written to `storage/eval/eval-<timestamp>.json` with per-question
detail. A metric with no baseline to diff against is a number, not a finding.

## Current results — stub generator, do not quote these

```
strategy   recall  precision  citations  refusal  false-refusal
fixed      100.0%      50.6%      34.7%    80.0%          19.2%
header     100.0%      51.5%      34.7%    80.0%          17.0%
semantic   100.0%      54.5%      35.1%    80.0%          19.2%
```

What is and is not real here:

- **Retrieval precision is a genuine signal.** It is computed from document
  names with no model involved. Semantic chunking leads (54.5%), header second
  (51.5%), fixed last (50.6%). The gap is small and the corpus is six files —
  suggestive, not conclusive.
- **Recall at 100% means the corpus is too easy**, not that retrieval is
  solved. Six documents on distinct topics is not a discriminating test.
- **Citation accuracy of ~35% is a stub artifact.** The stub emits two
  sentences and cites once, so the second is always `UNCITED`. It measures the
  stub, not the system.
- **Correctness and faithfulness are absent** because grading with a stub judge
  that always returns `SUPPORTED` would be meaningless.

## Limitations

- **No real numbers yet.** Every judge-backed metric needs a model connected.
  This is the single biggest gap in the project.
- **Corpus is too small and too easy.** 100% recall across all three strategies
  says so. A real comparison needs a corpus where retrieval can fail.
- **One judge, no agreement check.** The plan's Appendix A.7 suggests running a
  paid judge once and reporting both numbers; judge disagreement is itself an
  interesting result and has not been measured.
- **No per-category breakdown in the summary table.** The per-question JSON has
  the data; the rollup does not slice by category yet.
