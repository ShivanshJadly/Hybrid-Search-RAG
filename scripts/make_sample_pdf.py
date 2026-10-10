"""Generate a multi-page PDF fixture.

Exists so the page-number citation path has something real to run against.
Requires reportlab, which is a fixture-only dependency:

    pip install reportlab
    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "data-retention-policy.pdf"

PAGES: list[list[tuple[str, str]]] = [
    [
        ("h1", "Data Retention Policy"),
        ("p", "This policy defines how long each class of data is kept, who is "
              "permitted to access it, and how deletion is verified. It applies "
              "to every production system and to any copy of production data "
              "held elsewhere, including analytics warehouses and backups."),
        ("h2", "Scope"),
        ("p", "The policy covers customer records, operational telemetry, "
              "application logs, audit trails and backups. It does not cover "
              "source code, build artefacts or internal documentation, which "
              "are governed separately by the engineering handbook."),
        ("p", "Where a contractual obligation specifies a retention period "
              "different from this policy, the contract takes precedence. Such "
              "cases must be recorded in the retention register so that the "
              "exception is visible during audit rather than discovered during "
              "one."),
    ],
    [
        ("h2", "Retention Periods"),
        ("p", "Customer records are retained for the life of the account plus "
              "seven years, driven by financial reporting obligations rather "
              "than by any product requirement."),
        ("p", "Application logs are retained for thirty days in hot storage and "
              "a further eleven months in cold storage. Cold storage is not "
              "queryable directly; retrieving from it takes up to twelve hours "
              "and requires an approved access request."),
        ("p", "Operational telemetry, meaning metrics and traces rather than "
              "logs, is retained for thirteen months. The extra month past a "
              "year exists so that year-over-year comparisons remain possible "
              "at the end of a reporting period."),
        ("p", "Audit trails are retained for seven years and are immutable. No "
              "process, including this one, may shorten audit trail retention."),
    ],
    [
        ("h2", "Deletion and Verification"),
        ("p", "Deletion requests are processed within thirty days. The request "
              "is fanned out to every system holding a copy, and each system "
              "reports completion independently. A request is only closed once "
              "every system has reported."),
        ("p", "Backups are the common source of incomplete deletion. Because "
              "backups are immutable snapshots, a record cannot be surgically "
              "removed from an existing backup. Instead the record is "
              "suppressed on restore, and the backup ages out naturally under "
              "the thirty-five day backup retention window."),
        ("h2", "Access Control"),
        ("p", "Access to customer records in production requires an approved "
              "access request tied to a named ticket. Approvals expire after "
              "eight hours. Standing access to production customer data is not "
              "granted to any individual role."),
        ("p", "Every access is logged to the audit trail with the requester, "
              "the ticket, the records touched and the justification given. "
              "Audit trail entries are reviewed monthly by the compliance "
              "function."),
    ],
]


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=LETTER,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
        title="Data Retention Policy",
    )

    flow = []
    for i, page in enumerate(PAGES):
        for kind, text in page:
            style = {"h1": "Title", "h2": "Heading2", "p": "BodyText"}[kind]
            flow.append(Paragraph(text, styles[style]))
            flow.append(Spacer(1, 10))
        if i < len(PAGES) - 1:
            flow.append(PageBreak())

    doc.build(flow)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size} bytes)")
