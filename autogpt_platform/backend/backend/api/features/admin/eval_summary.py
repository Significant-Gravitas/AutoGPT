"""Figures for a run that has cases but no ``summary.json`` yet (a local run
still going, or one that stopped early), worked out the way the eval does."""

from collections import defaultdict

from .eval_files import Summary
from .eval_models import EvalCase, EvalStats


def summary_from_cases(cases: list[EvalCase]) -> Summary:
    ran = [c for c in cases if not c.skipped]
    reviewers = list(dict.fromkeys(r.reviewer for c in ran for r in c.ratings))
    by_category: dict[str, list[EvalCase]] = defaultdict(list)
    for case in ran:
        by_category[case.category].append(case)
    return Summary(
        reviewers=reviewers,
        overall={name: _stats(ran, name) for name in reviewers},
        categories={
            category: {name: _stats(group, name) for name in reviewers}
            for category, group in sorted(by_category.items())
        },
    )


def _stats(cases: list[EvalCase], reviewer: str) -> EvalStats:
    ratings = [r for c in cases for r in c.ratings if r.reviewer == reviewer]
    count = len(cases) or 1
    passed = sum(r.verdict == "pass" for r in ratings)
    return EvalStats(
        cases=len(cases),
        score=round(sum(r.score for r in ratings) / count, 1),
        pass_rate=round(100 * passed / count, 1),
        passed=passed,
        partial=sum(r.verdict == "partial" for r in ratings),
        failed=sum(r.verdict == "fail" for r in ratings),
    )
