"""Per-user execution cost aggregation models and query."""

import asyncio
from datetime import date, datetime, timezone

from prisma.enums import AgentExecutionStatus
from pydantic import BaseModel

from .db import query_raw_with_schema


class UserAgentCostRollup(BaseModel):
    graph_id: str
    cost_cents: int
    run_count: int


class UserTopRun(BaseModel):
    execution_id: str
    graph_id: str
    cost_cents: int
    started_at: datetime
    status: AgentExecutionStatus
    duration_seconds: float
    node_error_count: int


class UserDailyCost(BaseModel):
    date: date
    cost_cents: int
    run_count: int
    success_count: int = 0
    failed_count: int = 0
    review_count: int = 0


class UserExecutionCostSummary(BaseModel):
    total_cents: int
    run_count: int
    billable_run_count: int
    failed_cost_cents: int
    success_run_count: int = 0
    failed_run_count: int = 0
    review_run_count: int = 0
    total_duration_seconds: float = 0
    # Denominator for "avg run duration": `run_count` also covers QUEUED /
    # RUNNING rows and rows whose stats never got a `walltime`, which SUM
    # silently skips — dividing by it biases the average downward.
    duration_run_count: int = 0
    by_agent: list[UserAgentCostRollup]
    top_runs: list[UserTopRun]
    daily: list[UserDailyCost]


_MAX_BY_AGENT_ROWS = 50
_MAX_TOP_RUNS = 50

_BASE_WHERE = (
    '"userId" = $1'
    ' AND "isDeleted" = false'
    ' AND "createdAt" >= $2::timestamp'
    ' AND "createdAt" <= $3::timestamp'
    " AND COALESCE((stats->>'is_dry_run')::boolean, false) = false"
)


async def get_user_cost_summary(
    *,
    user_id: str,
    since: datetime | None = None,
    until: datetime | None = None,
    top_runs_limit: int = 10,
) -> UserExecutionCostSummary:
    """Aggregate per-user execution costs from AgentGraphExecution.stats JSON.

    Defaults to the current calendar month (UTC) when `since`/`until` are not provided.
    Dry-run executions are excluded.

    Window filters use `createdAt` so the existing
    `@@index([userId, isDeleted, createdAt])` is hit on `AgentGraphExecution`
    — bucketing by `startedAt` instead would force a full per-user scan for
    heavy users.
    """
    # Hard cap top_runs_limit here too, not just at the FastAPI layer, since
    # this function is callable directly and an unbounded LIMIT would scan
    # the whole per-user history on the cost-ordered `top_runs` query.
    top_runs_limit = max(1, min(_MAX_TOP_RUNS, top_runs_limit))

    now = datetime.now(timezone.utc)
    if since is None:
        since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if until is None:
        until = now

    params = (user_id, since, until)
    totals, by_agent, top_runs, daily = await asyncio.gather(
        _fetch_totals(params),
        _fetch_by_agent(params),
        _fetch_top_runs(params, top_runs_limit),
        _fetch_daily(params),
    )
    return UserExecutionCostSummary(
        total_cents=int(totals.get("total_cents") or 0),
        run_count=int(totals.get("run_count") or 0),
        billable_run_count=int(totals.get("billable_run_count") or 0),
        failed_cost_cents=int(totals.get("failed_cost_cents") or 0),
        success_run_count=int(totals.get("success_run_count") or 0),
        failed_run_count=int(totals.get("failed_run_count") or 0),
        review_run_count=int(totals.get("review_run_count") or 0),
        total_duration_seconds=float(totals.get("total_duration_seconds") or 0),
        duration_run_count=int(totals.get("duration_run_count") or 0),
        by_agent=by_agent,
        top_runs=top_runs,
        daily=daily,
    )


async def _fetch_totals(params: tuple[str, datetime, datetime]) -> dict:
    rows = await query_raw_with_schema(
        "SELECT"
        "  COALESCE(SUM((stats->>'cost')::numeric), 0)::bigint AS total_cents,"
        "  COUNT(*)::bigint AS run_count,"
        "  COUNT(*) FILTER (WHERE COALESCE((stats->>'cost')::numeric, 0) > 0)::bigint"
        "    AS billable_run_count,"
        "  COALESCE(SUM(CASE WHEN \"executionStatus\" IN ('FAILED', 'TERMINATED')"
        "    THEN (stats->>'cost')::numeric ELSE 0 END), 0)::bigint"
        "    AS failed_cost_cents,"
        "  COUNT(*) FILTER (WHERE \"executionStatus\" = 'COMPLETED')::bigint"
        "    AS success_run_count,"
        "  COUNT(*) FILTER (WHERE \"executionStatus\" IN ('FAILED', 'TERMINATED'))::bigint"
        "    AS failed_run_count,"
        "  COUNT(*) FILTER (WHERE \"executionStatus\" = 'REVIEW')::bigint"
        "    AS review_run_count,"
        "  COALESCE(SUM((stats->>'walltime')::numeric), 0)::float"
        "    AS total_duration_seconds,"
        "  COUNT(*) FILTER (WHERE (stats->>'walltime') IS NOT NULL)::bigint"
        "    AS duration_run_count"
        ' FROM {schema_prefix}"AgentGraphExecution"'
        f" WHERE {_BASE_WHERE}",
        *params,
    )
    return rows[0] if rows else {}


async def _fetch_by_agent(
    params: tuple[str, datetime, datetime],
) -> list[UserAgentCostRollup]:
    rows = await query_raw_with_schema(
        "SELECT"
        '  "agentGraphId" AS graph_id,'
        "  COALESCE(SUM((stats->>'cost')::numeric), 0)::bigint AS cost_cents,"
        "  COUNT(*)::bigint AS run_count"
        ' FROM {schema_prefix}"AgentGraphExecution"'
        f" WHERE {_BASE_WHERE}"
        '  GROUP BY "agentGraphId"'
        "  ORDER BY cost_cents DESC"
        f"  LIMIT {_MAX_BY_AGENT_ROWS}",
        *params,
    )
    return [
        UserAgentCostRollup(
            graph_id=r["graph_id"],
            cost_cents=int(r.get("cost_cents") or 0),
            run_count=int(r.get("run_count") or 0),
        )
        for r in rows
    ]


async def _fetch_top_runs(
    params: tuple[str, datetime, datetime], limit: int
) -> list[UserTopRun]:
    rows = await query_raw_with_schema(
        "SELECT"
        "  id AS execution_id,"
        '  "agentGraphId" AS graph_id,'
        "  COALESCE((stats->>'cost')::numeric, 0)::bigint AS cost_cents,"
        '  COALESCE("startedAt", "createdAt") AS started_at,'
        '  "executionStatus" AS status,'
        "  COALESCE((stats->>'walltime')::numeric, 0) AS duration_seconds,"
        "  COALESCE((stats->>'node_error_count')::int, 0) AS node_error_count"
        ' FROM {schema_prefix}"AgentGraphExecution"'
        f" WHERE {_BASE_WHERE}"
        "  AND COALESCE((stats->>'cost')::numeric, 0) > 0"
        '  ORDER BY cost_cents DESC, "createdAt" DESC'
        "  LIMIT $4",
        *params,
        limit,
    )
    return [
        UserTopRun(
            execution_id=r["execution_id"],
            graph_id=r["graph_id"],
            cost_cents=int(r.get("cost_cents") or 0),
            started_at=r["started_at"],
            status=AgentExecutionStatus(r["status"]),
            duration_seconds=float(r.get("duration_seconds") or 0),
            node_error_count=int(r.get("node_error_count") or 0),
        )
        for r in rows
    ]


async def _fetch_daily(
    params: tuple[str, datetime, datetime],
) -> list[UserDailyCost]:
    rows = await query_raw_with_schema(
        "SELECT"
        '  ("createdAt" AT TIME ZONE \'UTC\')::date AS "date",'
        "  COALESCE(SUM((stats->>'cost')::numeric), 0)::bigint AS cost_cents,"
        "  COUNT(*)::bigint AS run_count,"
        "  COUNT(*) FILTER (WHERE \"executionStatus\" = 'COMPLETED')::bigint"
        "    AS success_count,"
        "  COUNT(*) FILTER (WHERE \"executionStatus\" IN ('FAILED', 'TERMINATED'))::bigint"
        "    AS failed_count,"
        "  COUNT(*) FILTER (WHERE \"executionStatus\" = 'REVIEW')::bigint"
        "    AS review_count"
        ' FROM {schema_prefix}"AgentGraphExecution"'
        f" WHERE {_BASE_WHERE}"
        "  GROUP BY (\"createdAt\" AT TIME ZONE 'UTC')::date"
        '  ORDER BY "date" ASC',
        *params,
    )
    return [
        UserDailyCost(
            date=r["date"],
            cost_cents=int(r.get("cost_cents") or 0),
            run_count=int(r.get("run_count") or 0),
            success_count=int(r.get("success_count") or 0),
            failed_count=int(r.get("failed_count") or 0),
            review_count=int(r.get("review_count") or 0),
        )
        for r in rows
    ]
