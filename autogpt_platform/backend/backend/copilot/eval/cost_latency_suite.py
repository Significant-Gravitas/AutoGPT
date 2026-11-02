"""Cost + latency rollup from a batch of dream passes.

Consumes a list of ``DreamPassResult`` (each one wraps `elapsed_seconds`
and `usage`) and rolls them into the suite block written to
``results.json``. Pure data — the caller supplies the passes (synthetic
in tests, real-API runs in cloud). Keeps the harness composable: the
runner doesn't care whether the passes came from a live cluster or a
canned fixture.

Per ``dream/p0-spec.md`` §7 the report shape is::

    {
      "n": <pass count>,
      "skipped": <skipped pass count>,
      "errored": <errored pass count>,
      "latency": <latency_summary block>,
      "cost": {
        "total_usd": <sum of pass total_cost_usd, None if any unknown>,
        "mean_usd": <mean over passes with known cost>,
        "total_input_tokens": <sum>,
        "total_output_tokens": <sum>,
        "total_cache_read_tokens": <sum>,
        "total_cache_creation_tokens": <sum>
      },
      "by_execution_path": { <path>: <pass count> ... }
    }
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from pydantic.dataclasses import dataclass

from backend.copilot.dream.schemas import DreamPassResult

from .metrics import latency_summary, mean


@dataclass(frozen=True)
class CostLatencySuiteResult:
    n: int
    skipped: int
    errored: int
    latency: dict[str, float]
    cost: dict[str, float | int | None]
    by_execution_path: dict[str, int]


def run_cost_latency_suite(
    passes: Sequence[DreamPassResult],
) -> CostLatencySuiteResult:
    skipped = sum(1 for p in passes if p.skipped)
    errored = sum(1 for p in passes if p.error is not None)

    # Latency: only count passes that actually ran (skipped passes
    # short-circuit at the lock or fetch step and would distort p95).
    elapsed = [
        float(p.elapsed_seconds)
        for p in passes
        if p.elapsed_seconds is not None and not p.skipped
    ]

    total_input = sum(p.usage.total_input_tokens for p in passes if p.usage)
    total_output = sum(p.usage.total_output_tokens for p in passes if p.usage)
    total_cache_read = sum(p.usage.total_cache_read_tokens for p in passes if p.usage)
    total_cache_creation = sum(
        p.usage.total_cache_creation_tokens for p in passes if p.usage
    )

    known_costs = [
        float(p.usage.total_cost_usd)
        for p in passes
        if p.usage and p.usage.total_cost_usd is not None
    ]
    # total_usd is None if ANY pass had unknown cost — surfacing the
    # gap is more useful than reporting a confidently-wrong figure.
    any_unknown = any(
        p.usage is None or p.usage.total_cost_usd is None
        for p in passes
        if not p.skipped
    )
    total_usd: float | None = None if any_unknown else round(sum(known_costs), 6)
    mean_usd: float | None = round(mean(known_costs), 6) if known_costs else None

    path_counts = Counter(p.execution_path for p in passes if not p.skipped)

    return CostLatencySuiteResult(
        n=len(passes),
        skipped=skipped,
        errored=errored,
        latency=latency_summary(elapsed),
        cost={
            "total_usd": total_usd,
            "mean_usd": mean_usd,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_creation_tokens": total_cache_creation,
        },
        by_execution_path=dict(path_counts),
    )
