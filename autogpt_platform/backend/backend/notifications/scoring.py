"""Interestingness scoring for finished runs.

Each completed run is scored once, at completion, from signals the platform
already has. Scoring at completion is what lets Briefing assembly be a cheap
ranked read instead of a scan: a user with one agent and a user with two
hundred agents get emails of roughly the same length, and what changes is the
compression ratio.

Signals, in the order the design calls them out:

* terminal state — blocked and failed outrank success
* first-ever success of a newly created agent
* novel output versus a no-op ("checked, nothing new")
* cost or duration anomalies against that agent's own baseline
"""

import logging

from prisma.enums import AgentExecutionStatus
from prisma.models import AgentGraphExecution

from backend.data.db import query_raw_with_schema
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[RunScoring]")

# A failed or blocked run always outranks a clean one: the Briefing leads with
# what needs a decision.
_STATUS_SCORE: dict[AgentExecutionStatus, float] = {
    AgentExecutionStatus.FAILED: 0.80,
    AgentExecutionStatus.TERMINATED: 0.70,
    AgentExecutionStatus.COMPLETED: 0.30,
}

FIRST_SUCCESS_BONUS = 0.40
NOVEL_OUTPUT_BONUS = 0.20
NO_OP_PENALTY = 0.20
ANOMALY_BONUS = 0.15
# How far past its own baseline a run has to go before it counts as an anomaly
# rather than normal variance.
ANOMALY_FACTOR = 3.0
# Enough history for a baseline to mean anything.
MIN_BASELINE_RUNS = 5
# How far back the baseline looks. Scoring runs once per finished run, so this
# has to stay a bounded read.
BASELINE_WINDOW_RUNS = 200


async def score_completed_run(graph_exec_id: str) -> float | None:
    """Score one finished run and persist it. Never raises: a scoring failure
    must not fail the run that triggered it."""
    try:
        execution = await AgentGraphExecution.prisma().find_unique(
            where={"id": graph_exec_id}
        )
        if execution is None or execution.executionStatus not in _STATUS_SCORE:
            return None

        stats = dict(execution.stats or {})
        baseline = await _agent_cost_baseline(execution.agentGraphId, graph_exec_id)
        first_success = await _is_first_success(
            execution.agentGraphId, execution.executionStatus, graph_exec_id
        )
        score = compute_score(
            status=execution.executionStatus,
            cost_cents=float(stats.get("cost") or 0),
            node_error_count=int(stats.get("node_error_count") or 0),
            has_activity=bool(stats.get("activity_status")),
            first_success=first_success,
            cost_baseline=baseline,
        )
        await AgentGraphExecution.prisma().update(
            where={"id": graph_exec_id}, data={"interestingness": score}
        )
        return score
    except Exception:
        logger.warning(
            "Could not score run %s for the briefing", graph_exec_id, exc_info=True
        )
        return None


def compute_score(
    status: AgentExecutionStatus,
    cost_cents: float,
    node_error_count: int,
    has_activity: bool,
    first_success: bool,
    cost_baseline: float | None,
) -> float:
    """Pure scoring rule, so the ranking can be reasoned about and tested
    without a database."""
    score = _STATUS_SCORE.get(status, 0.0)

    if first_success:
        score += FIRST_SUCCESS_BONUS

    # A run that produced something is worth surfacing; a run that checked and
    # found nothing new is the definition of a no-op.
    if has_activity:
        score += NOVEL_OUTPUT_BONUS
    else:
        score -= NO_OP_PENALTY

    # An unusually expensive run against the agent's own history is news even
    # when it succeeded.
    if cost_baseline and cost_cents > cost_baseline * ANOMALY_FACTOR:
        score += ANOMALY_BONUS

    if node_error_count > 0 and status is AgentExecutionStatus.COMPLETED:
        # Finished, but not cleanly — worth ranking above a silent success.
        score += ANOMALY_BONUS

    return max(0.0, score)


async def _agent_cost_baseline(graph_id: str, exclude_exec_id: str) -> float | None:
    """Mean cost of this agent's own recent runs, or None when there isn't
    enough history for "unusual" to mean anything.

    Bounded to the most recent window rather than the agent's whole history:
    this runs once per finished run, and a busy agent can have hundreds of
    thousands of them.
    """
    rows = await query_raw_with_schema(
        """
        SELECT AVG(cost) AS mean_cost, COUNT(*) AS runs
        FROM (
            SELECT ("stats"::jsonb->>'cost')::numeric AS cost
            FROM {schema_prefix}"AgentGraphExecution"
            WHERE "agentGraphId" = $1
              AND "id" <> $2
              AND "isDeleted" = false
              AND "stats" IS NOT NULL
            ORDER BY "createdAt" DESC
            LIMIT $3
        ) recent
        """,
        graph_id,
        exclude_exec_id,
        BASELINE_WINDOW_RUNS,
    )
    if not rows or int(rows[0]["runs"] or 0) < MIN_BASELINE_RUNS:
        return None
    mean = rows[0]["mean_cost"]
    return float(mean) if mean is not None else None


async def _is_first_success(
    graph_id: str, status: AgentExecutionStatus, exec_id: str
) -> bool:
    """A newly built agent working for the first time is the single most
    interesting thing that can happen in a period.

    A find_first rather than a count: we only care whether *any* earlier clean
    run exists, and stopping at the first one keeps this cheap for agents with
    a long history.
    """
    if status is not AgentExecutionStatus.COMPLETED:
        return False
    earlier = await AgentGraphExecution.prisma().find_first(
        where={
            "agentGraphId": graph_id,
            "id": {"not": exec_id},
            "isDeleted": False,
            "executionStatus": AgentExecutionStatus.COMPLETED,
        }
    )
    return earlier is None
