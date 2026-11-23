"""Read a checked-in baseline's ``report_<name>.md``: when it ran, and its
"Cases not passed by every reviewer" list (``tasks/__main__.write_report``).

Each case there is written as::

    - **must_ask_013** (claude-opus-5-5 fail 35.0, grok-4.7-medium-fast fail 40.0) Title
      - failed checks: asks_question, ends_with_question
      - claude-opus-5-5: first reason
"""

import re
from datetime import datetime
from pathlib import Path

from .eval_cases import CaseDef, category_of, expectation, request_text, verdict
from .eval_files import read_text, started_at
from .eval_models import EvalCase, EvalCheck, EvalRating

_HEADING = re.compile(r"^# .+? (\d{8}-\d{6})\s*$")
_CASE = re.compile(
    r"^- \*\*(?P<id>[A-Za-z0-9_]+)\*\* \((?P<marks>[^)]*)\) ?(?P<title>.*)$"
)
_MARK = re.compile(
    r"^(?P<label>\S+) (?P<verdict>pass|partial|fail) (?P<score>\d+(?:\.\d+)?)$"
)
_FAILED = "  - failed checks: "
_REASON = "  - "


def report_started_at(path: Path) -> datetime | None:
    lines = read_text(path).splitlines()
    match = _HEADING.match(lines[0]) if lines else None
    return started_at(match.group(1)) if match else None


def report_cases(
    path: Path, reviewers: list[str], defs: dict[str, CaseDef]
) -> list[EvalCase]:
    """The cases the report lists, each with its ratings, failed checks and
    the reasons it kept."""
    names = {r.split(":")[-1]: r for r in reviewers}
    cases: list[EvalCase] = []
    for line in read_text(path).splitlines():
        case = _CASE.match(line)
        if case:
            cases.append(_case(case, names, defs))
        elif cases and line.startswith(_FAILED):
            cases[-1].checks = _failed_checks(line[len(_FAILED) :])
        elif cases and line.startswith(_REASON):
            _add_reason(cases[-1], line[len(_REASON) :])
    return cases


def _case(
    match: re.Match[str], names: dict[str, str], defs: dict[str, CaseDef]
) -> EvalCase:
    case_id = match.group("id")
    definition = defs.get(case_id)
    ratings = [_rating(mark.strip(), names) for mark in match.group("marks").split(",")]
    ratings = [r for r in ratings if r is not None]
    primary = ratings[0] if ratings else None
    return EvalCase(
        case_id=case_id,
        category=(definition.category if definition else None) or category_of(case_id),
        title=match.group("title").strip(),
        verdict=primary.verdict if primary else None,
        score=primary.score if primary else None,
        request=request_text(definition),
        expected=expectation(definition),
        ratings=ratings,
        detail="report",
    )


def _rating(mark: str, names: dict[str, str]) -> EvalRating | None:
    match = _MARK.match(mark)
    if not match:
        return None
    label = match.group("label")
    return EvalRating(
        reviewer=names.get(label, label),
        label=label,
        verdict=verdict(match.group("verdict")),
        score=float(match.group("score")),
    )


def _failed_checks(text: str) -> list[EvalCheck]:
    if text.strip() == "none":
        return []
    return [
        EvalCheck(name=name.strip(), passed=False)
        for name in text.split(",")
        if name.strip()
    ]


def _add_reason(case: EvalCase, text: str) -> None:
    label, _, reason = text.partition(": ")
    rating = next((r for r in case.ratings if r.label == label), None)
    if rating is not None and reason:
        rating.reasons.append(reason)
