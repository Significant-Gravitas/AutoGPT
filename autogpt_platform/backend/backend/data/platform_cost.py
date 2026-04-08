import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from prisma.models import PlatformCostLog as PrismaLog
from prisma.types import PlatformCostLogCreateInput
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema
from backend.util.cache import cached
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

MICRODOLLARS_PER_USD = 1_000_000

# Dashboard query limits — keep in sync with the SQL queries below
MAX_PROVIDER_ROWS = 500
MAX_USER_ROWS = 100

# Default date range for dashboard queries when no start date is provided.
# Prevents full-table scans on large deployments.
DEFAULT_DASHBOARD_DAYS = 30


def usd_to_microdollars(cost_usd: float | None) -> int | None:
    """Convert a USD amount (float) to microdollars (int). None-safe."""
    if cost_usd is None:
        return None
    return round(cost_usd * MICRODOLLARS_PER_USD)


class PlatformCostEntry(BaseModel):
    user_id: str
    graph_exec_id: str | None = None
    node_exec_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    block_id: str | None = None
    block_name: str | None = None
    provider: str
    credential_id: str | None = None
    cost_microdollars: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    data_size: int | None = None
    duration: float | None = None
    model: str | None = None
    tracking_type: str | None = None
    tracking_amount: float | None = None
    metadata: dict[str, Any] | None = None


async def log_platform_cost(entry: PlatformCostEntry) -> None:
    await PrismaLog.prisma().create(
        data=PlatformCostLogCreateInput(
            userId=entry.user_id,
            graphExecId=entry.graph_exec_id,
            nodeExecId=entry.node_exec_id,
            graphId=entry.graph_id,
            nodeId=entry.node_id,
            blockId=entry.block_id,
            blockName=entry.block_name,
            # Normalize to lowercase so the (provider, createdAt) index is always
            # used without LOWER() on the read side.
            provider=entry.provider.lower(),
            credentialId=entry.credential_id,
            costMicrodollars=entry.cost_microdollars,
            inputTokens=entry.input_tokens,
            outputTokens=entry.output_tokens,
            dataSize=entry.data_size,
            duration=entry.duration,
            model=entry.model,
            trackingType=entry.tracking_type,
            trackingAmount=entry.tracking_amount,
            metadata=SafeJson(entry.metadata or {}),
        )
    )


# Bound the number of concurrent cost-log DB inserts to prevent unbounded
# task/connection growth under sustained load or DB slowness.
_log_semaphore = asyncio.Semaphore(50)


async def log_platform_cost_safe(entry: PlatformCostEntry) -> None:
    """Fire-and-forget wrapper that never raises."""
    try:
        async with _log_semaphore:
            await log_platform_cost(entry)
    except Exception:
        logger.exception(
            "Failed to log platform cost for user=%s provider=%s block=%s",
            entry.user_id,
            entry.provider,
            entry.block_name,
        )


def _mask_email(email: str | None) -> str | None:
    """Mask an email address to reduce PII exposure in admin API responses.

    Turns 'user@example.com' into 'us***@example.com'.
    Handles short local parts gracefully (e.g. 'a@b.com' → 'a***@b.com').
    """
    if not email:
        return email
    at = email.find("@")
    if at < 0:
        return "***"
    local = email[:at]
    domain = email[at:]
    visible = local[:2] if len(local) >= 2 else local[:1]
    return f"{visible}***{domain}"


class ProviderCostSummary(BaseModel):
    provider: str
    tracking_type: str | None = None
    total_cost_microdollars: int
    total_input_tokens: int
    total_output_tokens: int
    total_duration_seconds: float = 0.0
    total_tracking_amount: float = 0.0
    request_count: int


class UserCostSummary(BaseModel):
    user_id: str | None = None
    email: str | None = None
    total_cost_microdollars: int
    total_input_tokens: int
    total_output_tokens: int
    request_count: int


class CostLogRow(BaseModel):
    id: str
    created_at: datetime
    user_id: str | None = None
    email: str | None = None
    graph_exec_id: str | None = None
    node_exec_id: str | None = None
    block_name: str
    provider: str
    tracking_type: str | None = None
    cost_microdollars: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration: float | None = None
    model: str | None = None


class PlatformCostDashboard(BaseModel):
    by_provider: list[ProviderCostSummary]
    by_user: list[UserCostSummary]
    total_cost_microdollars: int
    total_requests: int
    total_users: int


def _build_where(
    start: datetime | None,
    end: datetime | None,
    provider: str | None,
    user_id: str | None,
    table_alias: str = "",
) -> tuple[str, list[Any]]:
    prefix = f"{table_alias}." if table_alias else ""
    clauses: list[str] = []
    params: list[Any] = []
    idx = 1

    if start:
        clauses.append(f'{prefix}"createdAt" >= ${idx}::timestamptz')
        params.append(start)
        idx += 1
    if end:
        clauses.append(f'{prefix}"createdAt" <= ${idx}::timestamptz')
        params.append(end)
        idx += 1
    if provider:
        # Provider names are normalized to lowercase at write time so a plain
        # equality check is sufficient and the (provider, createdAt) index is used.
        clauses.append(f'{prefix}"provider" = ${idx}')
        params.append(provider.lower())
        idx += 1
    if user_id:
        clauses.append(f'{prefix}"userId" = ${idx}')
        params.append(user_id)
        idx += 1

    return (" AND ".join(clauses) if clauses else "TRUE", params)


@cached(ttl_seconds=30)
async def get_platform_cost_dashboard(
    start: datetime | None = None,
    end: datetime | None = None,
    provider: str | None = None,
    user_id: str | None = None,
) -> PlatformCostDashboard:
    """Aggregate platform cost logs for the admin dashboard.

    Note: by_provider rows are keyed on (provider, tracking_type). A single
    provider can therefore appear in multiple rows if it has entries with
    different billing models (e.g. "openai" with both "tokens" and "cost_usd"
    if pricing is later added for some entries). Frontend treats each row
    independently rather than as a provider primary key.

    Defaults to the last DEFAULT_DASHBOARD_DAYS days when no start date is
    provided to avoid full-table scans on large deployments.
    """
    if start is None:
        start = datetime.now(timezone.utc) - timedelta(days=DEFAULT_DASHBOARD_DAYS)
    where_p, params_p = _build_where(start, end, provider, user_id, "p")

    by_provider_rows, by_user_rows, total_user_rows = await asyncio.gather(
        query_raw_with_schema(
            f"""
            SELECT
                p."provider",
                p."trackingType" AS tracking_type,
                COALESCE(SUM(p."costMicrodollars"), 0)::bigint AS total_cost,
                COALESCE(SUM(p."inputTokens"), 0)::bigint AS total_input_tokens,
                COALESCE(SUM(p."outputTokens"), 0)::bigint AS total_output_tokens,
                COALESCE(SUM(p."duration"), 0)::float AS total_duration,
                COALESCE(SUM(p."trackingAmount"), 0)::float AS total_tracking_amount,
                COUNT(*)::bigint AS request_count
            FROM {{schema_prefix}}"PlatformCostLog" p
            WHERE {where_p}
            GROUP BY p."provider", p."trackingType"
            ORDER BY total_cost DESC
            LIMIT {MAX_PROVIDER_ROWS}
            """,
            *params_p,
        ),
        query_raw_with_schema(
            f"""
            SELECT
                p."userId" AS user_id,
                u."email",
                COALESCE(SUM(p."costMicrodollars"), 0)::bigint AS total_cost,
                COALESCE(SUM(p."inputTokens"), 0)::bigint AS total_input_tokens,
                COALESCE(SUM(p."outputTokens"), 0)::bigint AS total_output_tokens,
                COUNT(*)::bigint AS request_count
            FROM {{schema_prefix}}"PlatformCostLog" p
            LEFT JOIN {{schema_prefix}}"User" u ON u."id" = p."userId"
            WHERE {where_p}
            GROUP BY p."userId", u."email"
            ORDER BY total_cost DESC
            LIMIT {MAX_USER_ROWS}
            """,
            *params_p,
        ),
        query_raw_with_schema(
            f"""
            SELECT COUNT(DISTINCT p."userId")::bigint AS cnt
            FROM {{schema_prefix}}"PlatformCostLog" p
            WHERE {where_p}
            """,
            *params_p,
        ),
    )

    # Use the exact COUNT(DISTINCT userId) so total_users is not capped at
    # MAX_USER_ROWS (which would silently report 100 for >100 active users).
    total_users = int(total_user_rows[0]["cnt"]) if total_user_rows else 0
    total_cost = sum(r["total_cost"] for r in by_provider_rows)
    total_requests = sum(r["request_count"] for r in by_provider_rows)

    return PlatformCostDashboard(
        by_provider=[
            ProviderCostSummary(
                provider=r["provider"],
                tracking_type=r.get("tracking_type"),
                total_cost_microdollars=r["total_cost"],
                total_input_tokens=r["total_input_tokens"],
                total_output_tokens=r["total_output_tokens"],
                total_duration_seconds=r.get("total_duration", 0.0),
                total_tracking_amount=r.get("total_tracking_amount", 0.0),
                request_count=r["request_count"],
            )
            for r in by_provider_rows
        ],
        by_user=[
            UserCostSummary(
                user_id=r.get("user_id"),
                email=_mask_email(r.get("email")),
                total_cost_microdollars=r["total_cost"],
                total_input_tokens=r["total_input_tokens"],
                total_output_tokens=r["total_output_tokens"],
                request_count=r["request_count"],
            )
            for r in by_user_rows
        ],
        total_cost_microdollars=total_cost,
        total_requests=total_requests,
        total_users=total_users,
    )


async def get_platform_cost_logs(
    start: datetime | None = None,
    end: datetime | None = None,
    provider: str | None = None,
    user_id: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[CostLogRow], int]:
    if start is None:
        start = datetime.now(tz=timezone.utc) - timedelta(days=DEFAULT_DASHBOARD_DAYS)
    where_sql, params = _build_where(start, end, provider, user_id, "p")

    offset = (page - 1) * page_size
    limit_idx = len(params) + 1
    offset_idx = len(params) + 2

    count_rows, rows = await asyncio.gather(
        query_raw_with_schema(
            f"""
            SELECT COUNT(*)::bigint AS cnt
            FROM {{schema_prefix}}"PlatformCostLog" p
            WHERE {where_sql}
            """,
            *params,
        ),
        query_raw_with_schema(
            f"""
            SELECT
                p."id",
                p."createdAt" AS created_at,
                p."userId" AS user_id,
                u."email",
                p."graphExecId" AS graph_exec_id,
                p."nodeExecId" AS node_exec_id,
                p."blockName" AS block_name,
                p."provider",
                p."trackingType" AS tracking_type,
                p."costMicrodollars" AS cost_microdollars,
                p."inputTokens" AS input_tokens,
                p."outputTokens" AS output_tokens,
                p."duration",
                p."model"
            FROM {{schema_prefix}}"PlatformCostLog" p
            LEFT JOIN {{schema_prefix}}"User" u ON u."id" = p."userId"
            WHERE {where_sql}
            ORDER BY p."createdAt" DESC, p."id" DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
            """,
            *params,
            page_size,
            offset,
        ),
    )
    total = count_rows[0]["cnt"] if count_rows else 0

    logs = [
        CostLogRow(
            id=r["id"],
            created_at=r["created_at"],
            user_id=r.get("user_id"),
            email=_mask_email(r.get("email")),
            graph_exec_id=r.get("graph_exec_id"),
            node_exec_id=r.get("node_exec_id"),
            block_name=r["block_name"],
            provider=r["provider"],
            tracking_type=r.get("tracking_type"),
            cost_microdollars=r.get("cost_microdollars"),
            input_tokens=r.get("input_tokens"),
            output_tokens=r.get("output_tokens"),
            duration=r.get("duration"),
            model=r.get("model"),
        )
        for r in rows
    ]
    return logs, total
