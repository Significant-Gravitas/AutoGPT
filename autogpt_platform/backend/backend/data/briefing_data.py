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

from prisma.models import UserBalance
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema
from backend.data.notifications import NotificationScope
from backend.util.exceptions import DatabaseError
from backend.util.json import SafeJson
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
    graph_version: int
    library_agent_id: str | None
    organization_id: str | None
    team_id: str | None
    agent: str
    interestingness: float
    activity_status: str | None


async def get_agent_period_stats(
    user_id: str,
    start: datetime,
    end: datetime,
    authorization_scopes: list[NotificationScope],
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
                  AND g."userId" = e."userId"
                  AND g."organizationId" IS NOT DISTINCT FROM e."organizationId"
                  AND g."teamId" IS NOT DISTINCT FROM e."teamId"
            WHERE e."userId" = $1
              AND e."isDeleted" = false
              -- `endedAt` is `timestamp without time zone` holding UTC, and the
              -- driver binds these bounds as `timestamptz`. A bare `::timestamp`
              -- cast converts using the session's TimeZone, so a non-UTC session
              -- silently shifts the whole period window.
              AND e."endedAt" >= ($2::timestamptz AT TIME ZONE 'UTC')
              AND e."endedAt" < ($3::timestamptz AT TIME ZONE 'UTC')
              AND EXISTS (
                  SELECT 1
                  FROM jsonb_to_recordset($4::jsonb)
                      AS scope(organization_id TEXT, team_id TEXT)
                  WHERE scope.organization_id
                            IS NOT DISTINCT FROM e."organizationId"
                    AND scope.team_id IS NOT DISTINCT FROM e."teamId"
              )
            GROUP BY e."agentGraphId"
            ORDER BY top_score DESC, runs DESC, graph_id
            """,
            user_id,
            start,
            end,
            _scopes_json(authorization_scopes),
        )
        return [_to_agent_stats(row) for row in rows]
    except Exception as e:
        raise DatabaseError(
            f"Failed to aggregate briefing stats for user {user_id}: {e}"
        ) from e


async def get_top_scored_runs(
    user_id: str,
    start: datetime,
    end: datetime,
    limit: int,
    authorization_scopes: list[NotificationScope],
) -> list[ScoredRun]:
    """The period's best candidates for highlights, straight off the score
    written at completion time."""
    try:
        rows = await query_raw_with_schema(
            """
            SELECT
                e."id" AS execution_id,
                e."agentGraphId" AS graph_id,
                e."agentGraphVersion" AS graph_version,
                library_agent."id" AS library_agent_id,
                e."organizationId" AS organization_id,
                e."teamId" AS team_id,
                COALESCE(g."name", '') AS agent,
                COALESCE(e."interestingness", 0) AS interestingness,
                e."stats"::jsonb->>'activity_status' AS activity_status
            FROM {schema_prefix}"AgentGraphExecution" e
            LEFT JOIN {schema_prefix}"AgentGraph" g
                   ON g."id" = e."agentGraphId"
                  AND g."version" = e."agentGraphVersion"
                  AND g."userId" = e."userId"
                  AND g."organizationId" IS NOT DISTINCT FROM e."organizationId"
                  AND g."teamId" IS NOT DISTINCT FROM e."teamId"
            LEFT JOIN {schema_prefix}"LibraryAgent" library_agent
                   ON library_agent."userId" = e."userId"
                  AND library_agent."agentGraphId" = e."agentGraphId"
                  AND library_agent."agentGraphVersion" = e."agentGraphVersion"
                  AND library_agent."organizationId"
                        IS NOT DISTINCT FROM e."organizationId"
                  AND library_agent."teamId" IS NOT DISTINCT FROM e."teamId"
                  AND library_agent."isDeleted" = false
            WHERE e."userId" = $1
              AND e."isDeleted" = false
              -- `endedAt` is `timestamp without time zone` holding UTC, and the
              -- driver binds these bounds as `timestamptz`. A bare `::timestamp`
              -- cast converts using the session's TimeZone, so a non-UTC session
              -- silently shifts the whole period window.
              AND e."endedAt" >= ($2::timestamptz AT TIME ZONE 'UTC')
              AND e."endedAt" < ($3::timestamptz AT TIME ZONE 'UTC')
              AND e."executionStatus" = 'COMPLETED'
              AND EXISTS (
                  SELECT 1
                  FROM jsonb_to_recordset($4::jsonb)
                      AS scope(organization_id TEXT, team_id TEXT)
                  WHERE scope.organization_id
                            IS NOT DISTINCT FROM e."organizationId"
                    AND scope.team_id IS NOT DISTINCT FROM e."teamId"
              )
            ORDER BY e."interestingness" DESC NULLS LAST, e."endedAt" DESC
            LIMIT $5
            """,
            user_id,
            start,
            end,
            _scopes_json(authorization_scopes),
            limit,
        )
        return [
            ScoredRun(
                execution_id=row["execution_id"],
                graph_id=row["graph_id"],
                graph_version=int(row["graph_version"]),
                library_agent_id=row["library_agent_id"],
                organization_id=row["organization_id"],
                team_id=row["team_id"],
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


async def count_active_agents(
    user_id: str, authorization_scopes: list[NotificationScope]
) -> int:
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
              AND EXISTS (
                  SELECT 1
                  FROM jsonb_to_recordset($2::jsonb)
                      AS scope(organization_id TEXT, team_id TEXT)
                  WHERE scope.organization_id
                            IS NOT DISTINCT FROM "organizationId"
                    AND scope.team_id IS NOT DISTINCT FROM "teamId"
              )
            """,
            user_id,
            _scopes_json(authorization_scopes),
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
        rows = await query_raw_with_schema(
            """
            SELECT COUNT(*) AS total
            FROM {schema_prefix}"AgentPreset" preset
            LEFT JOIN {schema_prefix}"Organization" org
                   ON org."id" = preset."organizationId"
            WHERE preset."userId" = $1
              AND preset."isActive" = true
              AND preset."isDeleted" = false
              AND (
                  preset."organizationId" IS NULL
                  OR (
                      org."isPersonal" = true
                      AND org."deletedAt" IS NULL
                      AND org."bootstrapUserId" = $1
                  )
              )
            """,
            user_id,
        )
        return int(rows[0]["total"]) if rows else 0
    except Exception as e:
        raise DatabaseError(
            f"Failed to count scheduled agents for user {user_id}: {e}"
        ) from e


async def get_briefing_credit_balance(user_id: str) -> float:
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


async def get_briefing_resource_scopes(
    user_id: str, start: datetime, end: datetime
) -> list[NotificationScope]:
    try:
        rows = await query_raw_with_schema(
            """
            SELECT DISTINCT "organizationId" AS organization_id, "teamId" AS team_id
            FROM {schema_prefix}"AgentGraphExecution"
            WHERE "userId" = $1
              AND "isDeleted" = false
              AND "endedAt" >= ($2::timestamptz AT TIME ZONE 'UTC')
              AND "endedAt" < ($3::timestamptz AT TIME ZONE 'UTC')
            UNION
            SELECT DISTINCT "organizationId" AS organization_id, "teamId" AS team_id
            FROM {schema_prefix}"AgentGraph"
            WHERE "userId" = $1
              AND "isActive" = true
            """,
            user_id,
            start,
            end,
        )
        return [
            NotificationScope(
                organization_id=row["organization_id"],
                team_id=row["team_id"],
            )
            for row in rows
        ]
    except Exception as e:
        raise DatabaseError(
            f"Failed to list briefing resource scopes for user {user_id}: {e}"
        ) from e


def _scopes_json(scopes: list[NotificationScope]) -> SafeJson:
    return SafeJson(
        [
            {
                "organization_id": scope.organization_id,
                "team_id": scope.team_id,
            }
            for scope in scopes
        ]
    )
