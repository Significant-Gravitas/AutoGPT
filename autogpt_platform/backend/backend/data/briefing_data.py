"""Aggregation for the Briefing.

The unit of aggregation is the **agent**, never the run. Runs are scored at
completion, agents are ranked here, and the email is assembled top-down under a
hard length budget — so ten thousand runs is a legitimate week without ten
thousand of anything landing in an inbox.

Everything below is one grouped read per user per period. Nothing walks the
individual executions.
"""

import logging
from datetime import datetime

from prisma.models import AgentPreset, UserBalance
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema
from backend.util.exceptions import DatabaseError
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[BriefingData]")


class AgentPeriodStats(BaseModel):
    """One agent's whole period, which is one ledger row."""

    graph_id: str
    agent: str
    runs: int
    failed: int
    credits: float
    top_score: float


class ScoredRun(BaseModel):
    """A candidate for the highlights block."""

    execution_id: str
    graph_id: str
    agent: str
    interestingness: float
    activity_status: str | None


async def get_agent_period_stats(
    user_id: str, start: datetime, end: datetime
) -> list[AgentPeriodStats]:
    """Per-agent totals for the period, ranked by interestingness.

    Ordering here is what the ledger's "six rows sorted by interestingness,
    then + N more agents" rule reads off, so it is part of the contract rather
    than a detail of the query.
    """
    try:
        rows = await query_raw_with_schema(
            """
            SELECT
                e."agentGraphId" AS graph_id,
                COALESCE(MAX(g."name"), '') AS agent,
                COUNT(*) AS runs,
                COUNT(*) FILTER (
                    WHERE e."executionStatus" IN ('FAILED', 'TERMINATED')
                ) AS failed,
                COALESCE(SUM((e."stats"::jsonb->>'cost')::numeric), 0) AS cost_cents,
                COALESCE(MAX(e."interestingness"), 0) AS top_score
            FROM {schema_prefix}"AgentGraphExecution" e
            LEFT JOIN {schema_prefix}"AgentGraph" g
                   ON g."id" = e."agentGraphId"
                  AND g."version" = e."agentGraphVersion"
            WHERE e."userId" = $1
              AND e."isDeleted" = false
              AND e."endedAt" >= $2::timestamp
              AND e."endedAt" < $3::timestamp
            GROUP BY e."agentGraphId"
            ORDER BY top_score DESC, runs DESC, graph_id
            """,
            user_id,
            start,
            end,
        )
        return [_to_agent_stats(row) for row in rows]
    except Exception as e:
        raise DatabaseError(
            f"Failed to aggregate briefing stats for user {user_id}: {e}"
        ) from e


async def get_top_scored_runs(
    user_id: str, start: datetime, end: datetime, limit: int
) -> list[ScoredRun]:
    """The period's best candidates for highlights, straight off the score
    written at completion time."""
    try:
        rows = await query_raw_with_schema(
            """
            SELECT
                e."id" AS execution_id,
                e."agentGraphId" AS graph_id,
                COALESCE(g."name", '') AS agent,
                COALESCE(e."interestingness", 0) AS interestingness,
                e."stats"::jsonb->>'activity_status' AS activity_status
            FROM {schema_prefix}"AgentGraphExecution" e
            LEFT JOIN {schema_prefix}"AgentGraph" g
                   ON g."id" = e."agentGraphId"
                  AND g."version" = e."agentGraphVersion"
            WHERE e."userId" = $1
              AND e."isDeleted" = false
              AND e."endedAt" >= $2::timestamp
              AND e."endedAt" < $3::timestamp
              AND e."executionStatus" = 'COMPLETED'
            ORDER BY e."interestingness" DESC NULLS LAST, e."endedAt" DESC
            LIMIT $4
            """,
            user_id,
            start,
            end,
            limit,
        )
        return [
            ScoredRun(
                execution_id=row["execution_id"],
                graph_id=row["graph_id"],
                agent=row["agent"] or _fallback_name(row["graph_id"]),
                interestingness=float(row["interestingness"] or 0),
                activity_status=row["activity_status"],
            )
            for row in rows
        ]
    except Exception as e:
        raise DatabaseError(
            f"Failed to read top-scored runs for user {user_id}: {e}"
        ) from e


async def count_active_agents(user_id: str) -> int:
    """Every agent the user has switched on, counted once across its versions.

    The Briefing subtracts the agents that actually ran from this to get the
    "on standby" figure, rather than passing a list of IDs into SQL.
    """
    try:
        rows = await query_raw_with_schema(
            """
            SELECT COUNT(DISTINCT "id") AS total
            FROM {schema_prefix}"AgentGraph"
            WHERE "userId" = $1
              AND "isActive" = true
            """,
            user_id,
        )
        return int(rows[0]["total"]) if rows else 0
    except Exception as e:
        raise DatabaseError(
            f"Failed to count active agents for user {user_id}: {e}"
        ) from e


async def count_scheduled_agents(user_id: str) -> int:
    """Agents configured to run on their own — the ones that would silently
    stop if the credits ran out."""
    try:
        return await AgentPreset.prisma().count(
            where={"userId": user_id, "isActive": True, "isDeleted": False}
        )
    except Exception as e:
        raise DatabaseError(
            f"Failed to count scheduled agents for user {user_id}: {e}"
        ) from e


async def get_credit_balance(user_id: str) -> float:
    """Balance in credits (the stored figure is cents)."""
    try:
        balance = await UserBalance.prisma().find_unique(where={"userId": user_id})
        return (balance.balance if balance else 0) / 100
    except Exception as e:
        raise DatabaseError(f"Failed to read balance for user {user_id}: {e}") from e


def _to_agent_stats(row: dict) -> AgentPeriodStats:
    graph_id = row["graph_id"]
    return AgentPeriodStats(
        graph_id=graph_id,
        agent=row["agent"] or _fallback_name(graph_id),
        runs=int(row["runs"]),
        failed=int(row["failed"]),
        credits=float(row["cost_cents"]) / 100,
        top_score=float(row["top_score"] or 0),
    )


def _fallback_name(graph_id: str) -> str:
    return f"Agent {graph_id[:8]}"
