"""Compare a run with the one before it in its series: totals, categories,
and which cases got better or worse."""

from pydantic import BaseModel, Field

from .eval_files import RunSource, Summary
from .eval_models import (
    EvalCase,
    EvalCaseChange,
    EvalCategoryChange,
    EvalComparison,
    EvalRunSummary,
    EvalStats,
    EvalVerdict,
)

RANK: dict[EvalVerdict, int] = {"fail": 0, "partial": 1, "pass": 2}


class RunData(BaseModel):
    source: RunSource
    run: EvalRunSummary
    summary: Summary
    cases: list[EvalCase] = Field(default_factory=list)


def matching_reviewer(summary: Summary, reviewer: str | None) -> str | None:
    """The same reviewer when the run has it, else the run's first."""
    if reviewer and reviewer in summary.overall:
        return reviewer
    return summary.reviewers[0] if summary.reviewers else None


def stats_for(summary: Summary, reviewer: str | None) -> EvalStats | None:
    name = matching_reviewer(summary, reviewer)
    return summary.overall.get(name) if name else None


def compare(current: RunData, previous: RunData) -> EvalComparison | None:
    reviewer = matching_reviewer(current.summary, None)
    before_name = matching_reviewer(previous.summary, reviewer)
    now = stats_for(current.summary, reviewer)
    before = stats_for(previous.summary, before_name)
    if now is None or before is None:
        return None
    changes = _case_changes(current, previous, reviewer, before_name)
    return EvalComparison(
        previous_id=previous.run.id,
        previous_label=previous.run.label,
        pass_rate_delta=round(now.pass_rate - before.pass_rate, 1),
        score_delta=round(now.score - before.score, 1),
        failed_delta=now.failed - before.failed,
        categories=_categories(current.summary, previous.summary, reviewer),
        cases_compared=changes is not None,
        improved=[c for c in changes or [] if RANK[c.after] > RANK[c.before]],
        regressed=[c for c in changes or [] if RANK[c.after] < RANK[c.before]],
    )


def previous_verdicts(
    current: RunData, previous: RunData
) -> dict[str, EvalVerdict | None]:
    """Each current case's verdict in the previous run, when it is known."""
    reviewer = matching_reviewer(current.summary, None)
    known = _verdicts(previous.cases, matching_reviewer(previous.summary, reviewer))
    return {c.case_id: known.get(c.case_id) or _absent(previous) for c in current.cases}


def _categories(
    current: Summary, previous: Summary, reviewer: str | None
) -> list[EvalCategoryChange]:
    before_name = matching_reviewer(previous, reviewer)
    changes: list[EvalCategoryChange] = []
    for name, per in current.categories.items():
        now = per.get(reviewer or "") or next(iter(per.values()), None)
        before = previous.categories.get(name, {}).get(before_name or "")
        if now is None:
            continue
        changes.append(
            EvalCategoryChange(
                name=name,
                pass_rate=now.pass_rate,
                previous_pass_rate=before.pass_rate if before else None,
                pass_rate_delta=(
                    round(now.pass_rate - before.pass_rate, 1) if before else None
                ),
                score_delta=round(now.score - before.score, 1) if before else None,
            )
        )
    return sorted(
        changes,
        key=lambda c: (c.pass_rate_delta is None, c.pass_rate_delta or 0),
    )


def _case_changes(
    current: RunData,
    previous: RunData,
    reviewer: str | None,
    before_name: str | None,
) -> list[EvalCaseChange] | None:
    if "none" in (current.run.case_coverage, previous.run.case_coverage):
        return None
    now = _verdicts(current.cases, reviewer)
    before = _verdicts(previous.cases, before_name)
    info = {c.case_id: c for c in [*previous.cases, *current.cases]}
    changes: list[EvalCaseChange] = []
    for case_id in sorted(set(now) | set(before)):
        after = now.get(case_id) or _absent(current)
        prior = before.get(case_id) or _absent(previous)
        if after is None or prior is None or after == prior:
            continue
        case = info[case_id]
        changes.append(
            EvalCaseChange(
                case_id=case_id,
                title=case.title,
                category=case.category,
                before=prior,
                after=after,
            )
        )
    return changes


def _verdicts(cases: list[EvalCase], reviewer: str | None) -> dict[str, EvalVerdict]:
    verdicts: dict[str, EvalVerdict] = {}
    for case in cases:
        if case.skipped:
            continue
        rating = next((r for r in case.ratings if r.reviewer == reviewer), None)
        found = rating.verdict if rating else case.verdict
        if found:
            verdicts[case.case_id] = found
    return verdicts


def _absent(data: RunData) -> EvalVerdict | None:
    """A case a report leaves out was passed by every reviewer."""
    return "pass" if data.run.case_coverage == "not_passed" else None
