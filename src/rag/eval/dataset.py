"""Golden question set loading and validation.

Validation is strict on purpose. A malformed golden set produces plausible
looking metrics that are quietly wrong, which is worse than a crash -- so an
answerable question with no expected sources, or a no-answer question that
somehow has them, is an error rather than a warning.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATEGORIES = ("lookup", "multihop", "no_answer", "ambiguous")

DEFAULT_PATH = Path(__file__).with_name("golden_set.yaml")


@dataclass
class GoldenQuestion:
    id: str
    category: str
    question: str
    answerable: bool
    expected_answer: str | None = None
    expected_sources: list[str] = field(default_factory=list)


class InvalidGoldenSet(Exception):
    pass


def load_golden_set(path: Path | None = None) -> list[GoldenQuestion]:
    path = Path(path) if path else DEFAULT_PATH
    if not path.exists():
        raise InvalidGoldenSet(f"golden set not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise InvalidGoldenSet("golden set must be a list of questions")

    questions: list[GoldenQuestion] = []
    seen: set[str] = set()

    for i, row in enumerate(raw):
        where = row.get("id", f"index {i}")
        for required in ("id", "category", "question", "answerable"):
            if required not in row:
                raise InvalidGoldenSet(f"{where}: missing '{required}'")

        if row["id"] in seen:
            raise InvalidGoldenSet(f"duplicate id: {row['id']}")
        seen.add(row["id"])

        if row["category"] not in CATEGORIES:
            raise InvalidGoldenSet(
                f"{where}: unknown category '{row['category']}'; "
                f"expected one of {', '.join(CATEGORIES)}"
            )

        answerable = bool(row["answerable"])
        sources = row.get("expected_sources") or []

        if answerable and not sources:
            raise InvalidGoldenSet(
                f"{where}: answerable questions need expected_sources, "
                f"otherwise retrieval cannot be scored"
            )
        if not answerable and sources:
            raise InvalidGoldenSet(
                f"{where}: a no-answer question must not list expected_sources"
            )

        questions.append(
            GoldenQuestion(
                id=row["id"],
                category=row["category"],
                question=row["question"],
                answerable=answerable,
                expected_answer=row.get("expected_answer"),
                expected_sources=list(sources),
            )
        )

    if not questions:
        raise InvalidGoldenSet("golden set is empty")
    return questions


def summarise(questions: list[GoldenQuestion]) -> dict:
    counts = Counter(q.category for q in questions)
    return {
        "total": len(questions),
        "by_category": dict(counts),
        "answerable": sum(1 for q in questions if q.answerable),
        "unanswerable": sum(1 for q in questions if not q.answerable),
    }
