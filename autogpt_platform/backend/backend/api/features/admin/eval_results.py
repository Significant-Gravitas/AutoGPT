"""The admin Evals page's data: every saved run of the task eval and the
feature eval, and one run in full with its cases and the run before it."""

import logging
import re
from collections import defaultdict
from datetime import UTC, datetime

from .eval_cases import load_case_defs, read_cases
from .eval_commands import DEFAULT_COMMANDS, rerun_commands
from .eval_compare import RunData, compare, previous_verdicts, stats_for
from .eval_files import RunSource, Summary, discover, read_summary, started_at
from .eval_models import (
    EvalCase,
    EvalCaseCoverage,
    EvalCategoryScore,
    EvalHistoryPoint,
    EvalReviewerStats,
    EvalRunDetail,
    EvalRunList,
    EvalRunSummary,
    EvalStats,
)
from .eval_report import report_cases, report_started_at
from .eval_summary import summary_from_cases

logger = logging.getLogger(__name__)

_BASELINE_LABEL = re.compile(r"^baseline_(?:(?P<family>.+)_)?run(?P<n>\d+)$")
_NOTES: dict[EvalCaseCoverage, str | None] = {
    "all": None,
    "not_passed": "This checked-in run kept its totals and, from its report, the "
    "cases not passed by every reviewer, with two reasons each. Cases not "
    "listed were passed by every reviewer.",
    "none": "This checked-in run kept only its totals and category scores. Run "
    "the eval locally to see every case.",
}
_EMPTY = EvalStats(cases=0, score=0, pass_rate=0, passed=0, partial=0, failed=0)
_ORDER = {"fail": 0, "partial": 1, "pass": 2}


def list_runs() -> EvalRunList:
    return EvalRunList(runs=[d.run for d in _load_all()], commands=DEFAULT_COMMANDS)


def get_run(run_id: str) -> EvalRunDetail | None:
    loaded = _load_all()
    current = next((d for d in loaded if d.run.id == run_id), None)
    if current is None:
        return None
    series = sorted(
        (d for d in loaded if d.run.series == current.run.series), key=_series_key
    )
    at = series.index(current)
    data = current.model_copy(update={"cases": _cases(current)})
    previous = None
    if at > 0:
        previous = series[at - 1].model_copy(update={"cases": _cases(series[at - 1])})
    return _detail(data, previous, series)


def _detail(
    data: RunData, previous: RunData | None, series: list[RunData]
) -> EvalRunDetail:
    cases = data.cases
    if previous is not None:
        before = previous_verdicts(data, previous)
        cases = [
            c.model_copy(update={"previous_verdict": before.get(c.case_id)})
            for c in cases
        ]
    summary = data.summary
    return EvalRunDetail(
        run=data.run,
        reviewers=[_reviewer_stats(name, s) for name, s in summary.overall.items()],
        categories=_categories(summary),
        cases=sorted(cases, key=_worst_first),
        comparison=compare(data, previous) if previous else None,
        history=[_point(d, data) for d in series],
        commands=rerun_commands(data.run.suite, summary),
        seconds_per_turn=summary.seconds_per_turn,
        prompt_tokens=summary.prompt_tokens,
        completion_tokens=summary.completion_tokens,
        note=_NOTES[data.run.case_coverage],
    )


def _load_all() -> list[RunData]:
    loaded = [data for source in discover() if (data := _load(source)) is not None]
    return _picker_order(loaded)


def _load(source: RunSource) -> RunData | None:
    try:
        summary = read_summary(source)
        if summary is None and source.results_path is not None:
            summary = summary_from_cases(read_cases(source.results_path, [], {}))
        if summary is None:
            return None
        return RunData(
            source=source, run=_run_summary(source, summary), summary=summary
        )
    except (OSError, ValueError) as e:
        logger.warning(f"Eval run {source.id} could not be read: {type(e).__name__}")
        return None


def _run_summary(source: RunSource, summary: Summary) -> EvalRunSummary:
    when = started_at(source.name) if source.source == "local" else None
    if source.report_path is not None:
        when = report_started_at(source.report_path)
    coverage: EvalCaseCoverage = "none"
    if source.results_path is not None:
        coverage = "all"
    elif source.report_path is not None:
        coverage = "not_passed"
    return EvalRunSummary(
        id=source.id,
        suite=source.suite,
        source=source.source,
        label=_label(source, when),
        started_at=when,
        backend=summary.backend,
        otto_model=summary.otto,
        user_model=summary.user,
        reviewers=summary.reviewers,
        stats=stats_for(summary, None) or _EMPTY,
        case_coverage=coverage,
        series=_series(source, summary),
    )


def _cases(data: RunData) -> list[EvalCase]:
    source = data.source
    defs = load_case_defs(data.run.suite)
    if source.results_path is not None:
        return read_cases(source.results_path, data.summary.reviewers, defs)
    if source.report_path is not None:
        return report_cases(source.report_path, data.summary.reviewers, defs)
    return []


def _label(source: RunSource, when: datetime | None) -> str:
    match = _BASELINE_LABEL.match(source.name)
    if source.source == "baseline" and match:
        family = f"{match.group('family')} " if match.group("family") else ""
        return f"Baseline {family}run {match.group('n')}"
    if when is not None:
        return when.strftime("%b %d, %H:%M UTC")
    return source.name


def _series(source: RunSource, summary: Summary) -> str:
    if source.suite == "features" and summary.offline:
        return "features:offline"
    return f"{source.suite}:{summary.backend or 'openai'}:{summary.otto or 'none'}"


def _series_key(data: RunData) -> tuple[int, float, str]:
    """Oldest first: checked-in baselines by their number, then local runs."""
    if data.run.source == "baseline":
        return (0, float(_run_number(data.run.id)), data.run.id)
    return (1, _timestamp(data.run), data.run.id)


def _picker_order(runs: list[RunData]) -> list[RunData]:
    """Task eval first; local runs newest first, then the baselines, newest
    family first and highest number first."""
    newest: dict[str, float] = defaultdict(float)
    for d in runs:
        newest[d.run.series] = max(newest[d.run.series], _timestamp(d.run))

    def key(d: RunData) -> tuple[bool, bool, float, str, int]:
        local = d.run.source == "local"
        when = _timestamp(d.run) if local else newest[d.run.series]
        return (
            d.run.suite != "task",
            not local,
            -when,
            d.run.series,
            -_run_number(d.run.id),
        )

    return sorted(runs, key=key)


def _run_number(run_id: str) -> int:
    match = re.search(r"run(\d+)$", run_id)
    return int(match.group(1)) if match else 0


def _timestamp(run: EvalRunSummary) -> float:
    return (run.started_at or datetime.fromtimestamp(0, UTC)).timestamp()


def _categories(summary: Summary) -> list[EvalCategoryScore]:
    primary = summary.reviewers[0] if summary.reviewers else ""
    scores = [
        EvalCategoryScore(
            name=name,
            stats=per.get(primary) or next(iter(per.values()), _EMPTY),
            reviewers=[_reviewer_stats(n, s) for n, s in per.items()],
        )
        for name, per in summary.categories.items()
    ]
    return sorted(scores, key=lambda c: (c.stats.pass_rate, c.name))


def _reviewer_stats(name: str, stats: EvalStats) -> EvalReviewerStats:
    label = "code checks" if name == "code" else name.split(":")[-1]
    return EvalReviewerStats(reviewer=name, label=label, stats=stats)


def _point(data: RunData, current: RunData) -> EvalHistoryPoint:
    stats = stats_for(
        data.summary, current.run.reviewers[0] if current.run.reviewers else None
    )
    return EvalHistoryPoint(
        id=data.run.id,
        label=data.run.label,
        pass_rate=stats.pass_rate if stats else 0,
        score=stats.score if stats else 0,
        current=data.run.id == current.run.id,
    )


def _worst_first(case: EvalCase) -> tuple[int, float, str]:
    rank = _ORDER[case.verdict] if case.verdict else 3
    return (rank, case.score if case.score is not None else 0, case.case_id)
