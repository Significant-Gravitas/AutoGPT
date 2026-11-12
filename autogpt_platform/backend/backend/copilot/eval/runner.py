"""Dream-eval harness — single-pass aggregator.

Composes the per-suite runners into one ``DreamEvalResult`` matching
the schema in ``dream/p0-spec.md`` §7. AgentProbe drives the live
side (runs N dream passes against a real cluster); this module is
what AgentProbe POSTs the results to / calls into. Synthetic tests
exercise the runner without any I/O.

The runner deliberately does NOT spawn dream passes itself. Coupling
the harness to the orchestrator would force CI to stand up FalkorDB /
Supabase / Redis for every run. AgentProbe (the user's TS repo) owns
the live-run side; this module owns the metrics roll-up + report
shape so both the live runs AND the CI smoke runs use the same
``results.json`` schema.

Suites available today:
  * staleness    — P0.3a heuristic precision/recall/stability
  * cost_latency — per-pass tokens + USD + p50/p95 elapsed

Deferred to future tasks (placeholder slots in the result dict):
  * ratification (P0.4 — needs the ratification scheduler, task #30)
  * web_fact_check (P0.5 — needs the webcheck module, task #32)
  * retrieval (warm-context golden query set — future)
  * demotion (precision/recall against ground-truth demotion labels —
    needs the full 50-fact fixture).

When a deferred suite gets implemented, add it to ``_run_one_suite``
and document it in the dict at the call site — no schema migration
needed (the report is open-ended).
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from typing import Any, Sequence

from pydantic import Field
from pydantic.dataclasses import dataclass
from pydantic_core import to_jsonable_python

from backend.copilot.dream.schemas import DreamPassResult

from .cost_latency_suite import run_cost_latency_suite
from .datasets import ALL_STALENESS_FIXTURES, StalenessFixture
from .staleness_suite import run_staleness_suite


@dataclass(frozen=True)
class DreamEvalConfig:
    """What suites + what fixtures one ``run_dream_eval`` call covers.

    Defaults to "all available suites with their default fixtures" so
    a caller without strong opinions still gets the full report.
    """

    run_staleness: bool = True
    run_cost_latency: bool = True
    staleness_fixtures: Sequence[StalenessFixture] = Field(
        default_factory=lambda: tuple(ALL_STALENESS_FIXTURES)
    )


@dataclass(frozen=True)
class DreamEvalResult:
    run_id: str
    ts: str
    python_version: str
    suites: dict[str, Any]

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(to_jsonable_python(self), indent=indent)


def run_dream_eval(
    *,
    run_id: str,
    passes: Sequence[DreamPassResult] = (),
    config: DreamEvalConfig | None = None,
    now: datetime | None = None,
) -> DreamEvalResult:
    """Aggregate suite results for one eval run.

    ``passes`` is the list of dream passes driven externally (by
    AgentProbe in production, or synthetic ``DreamPassResult`` objects
    in tests). Empty ``passes`` is fine — the cost/latency suite just
    reports zeros and the staleness suite runs independently against
    its fixture.

    ``run_id`` is the AgentProbe scenario id (when called from a live
    run) or a deterministic test id; carried through to ``results.json``
    so the report joins back to its source run.
    """
    config = config or DreamEvalConfig()
    now = now or datetime.now(timezone.utc)

    suites: dict[str, Any] = {}

    if config.run_staleness:
        suites["staleness"] = run_staleness_suite(config.staleness_fixtures, now=now)
    if config.run_cost_latency:
        suites["cost_latency"] = run_cost_latency_suite(passes)

    return DreamEvalResult(
        run_id=run_id,
        ts=now.isoformat().replace("+00:00", "Z"),
        python_version=platform.python_version(),
        suites={k: to_jsonable_python(v) for k, v in suites.items()},
    )
