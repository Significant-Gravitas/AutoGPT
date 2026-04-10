import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from prisma.models import PlatformCostLog as PrismaLog
from prisma.models import User as PrismaUser
from prisma.types import PlatformCostLogCreateInput, PlatformCostLogWhereInput
from pydantic import BaseModel

from backend.util.cache import cached
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

MICRODOLLARS_PER_USD = 1_000_000

# Dashboard query limits
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
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
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
            cacheReadTokens=entry.cache_read_tokens,
            cacheCreationTokens=entry.cache_creation_tokens,
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
    model: str | None = None
    total_cost_microdollars: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
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
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None


class PlatformCostDashboard(BaseModel):
    by_provider: list[ProviderCostSummary]
    by_user: list[UserCostSummary]
    total_cost_microdollars: int
    total_requests: int
    total_users: int


def _si(row: dict, field: str) -> int:
    """Extract an integer from a Prisma group_by _sum dict.

    Prisma Python serialises BigInt/Int aggregate sums as strings; coerce to int.
    """
    return int((row.get("_sum") or {}).get(field) or 0)


def _sf(row: dict, field: str) -> float:
    """Extract a float from a Prisma group_by _sum dict."""
    return float((row.get("_sum") or {}).get(field) or 0.0)


def _ca(row: dict) -> int:
    """Extract _count._all from a Prisma group_by row."""
    c = row.get("_count") or {}
    return int(c.get("_all") or 0) if isinstance(c, dict) else int(c or 0)


def _build_prisma_where(
    start: datetime | None,
    end: datetime | None,
    provider: str | None,
    user_id: str | None,
    model: str | None = None,
    block_name: str | None = None,
    tracking_type: str | None = None,
) -> PlatformCostLogWhereInput:
    """Build a Prisma WhereInput for PlatformCostLog filters."""
    where: PlatformCostLogWhereInput = {}

    if start and end:
        where["createdAt"] = {"gte": start, "lte": end}
    elif start:
        where["createdAt"] = {"gte": start}
    elif end:
        where["createdAt"] = {"lte": end}

    if provider:
        where["provider"] = provider.lower()

    if user_id:
        where["userId"] = user_id

    if model:
        where["model"] = model

    if block_name:
        # Case-insensitive match — mirrors the original LOWER() SQL filter.
        where["blockName"] = {"equals": block_name, "mode": "insensitive"}

    if tracking_type:
        where["trackingType"] = tracking_type

    return where


@cached(ttl_seconds=30)
async def get_platform_cost_dashboard(
    start: datetime | None = None,
    end: datetime | None = None,
    provider: str | None = None,
    user_id: str | None = None,
    model: str | None = None,
    block_name: str | None = None,
    tracking_type: str | None = None,
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

    where = _build_prisma_where(
        start, end, provider, user_id, model, block_name, tracking_type
    )

    sum_fields = {
        "costMicrodollars": True,
        "inputTokens": True,
        "outputTokens": True,
        "cacheReadTokens": True,
        "cacheCreationTokens": True,
        "duration": True,
        "trackingAmount": True,
    }

    # Run all four aggregation queries in parallel.
    by_provider_groups, by_user_groups, total_user_groups, total_agg_groups = (
        await asyncio.gather(
            # (provider, trackingType, model) aggregation — no ORDER BY in ORM;
            # sort by total cost descending in Python after fetch.
            PrismaLog.prisma().group_by(
                by=["provider", "trackingType", "model"],
                where=where,
                sum=sum_fields,
                count=True,
            ),
            # userId aggregation — emails fetched separately below.
            PrismaLog.prisma().group_by(
                by=["userId"],
                where=where,
                sum=sum_fields,
                count=True,
            ),
            # Distinct user count: group by userId, count groups.
            PrismaLog.prisma().group_by(
                by=["userId"],
                where=where,
                count=True,
            ),
            # Total aggregate: group by provider (no limit) to sum across all
            # matching rows. Summed in Python to get grand totals.
            PrismaLog.prisma().group_by(
                by=["provider"],
                where=where,
                sum={"costMicrodollars": True},
                count=True,
            ),
        )
    )

    # Sort by_provider by total cost descending and cap at MAX_PROVIDER_ROWS.
    by_provider_groups.sort(key=lambda r: _si(r, "costMicrodollars"), reverse=True)
    by_provider_groups = by_provider_groups[:MAX_PROVIDER_ROWS]

    # Sort by_user by total cost descending and cap at MAX_USER_ROWS.
    by_user_groups.sort(key=lambda r: _si(r, "costMicrodollars"), reverse=True)
    by_user_groups = by_user_groups[:MAX_USER_ROWS]

    # Batch-fetch emails for the users in by_user.
    user_ids = [r["userId"] for r in by_user_groups if r.get("userId") is not None]
    email_by_user_id: dict[str, str | None] = {}
    if user_ids:
        users = await PrismaUser.prisma().find_many(
            where={"id": {"in": user_ids}},
        )
        email_by_user_id = {u.id: u.email for u in users}

    # Total distinct users — exclude the NULL-userId group (deleted users).
    total_users = len([g for g in total_user_groups if g.get("userId") is not None])

    # Grand totals — sum across all provider groups (no LIMIT applied above).
    total_cost = sum(_si(r, "costMicrodollars") for r in total_agg_groups)
    total_requests = sum(_ca(r) for r in total_agg_groups)

    return PlatformCostDashboard(
        by_provider=[
            ProviderCostSummary(
                provider=r["provider"],
                tracking_type=r.get("trackingType"),
                model=r.get("model"),
                total_cost_microdollars=_si(r, "costMicrodollars"),
                total_input_tokens=_si(r, "inputTokens"),
                total_output_tokens=_si(r, "outputTokens"),
                total_cache_read_tokens=_si(r, "cacheReadTokens"),
                total_cache_creation_tokens=_si(r, "cacheCreationTokens"),
                total_duration_seconds=_sf(r, "duration"),
                total_tracking_amount=_sf(r, "trackingAmount"),
                request_count=_ca(r),
            )
            for r in by_provider_groups
        ],
        by_user=[
            UserCostSummary(
                user_id=r.get("userId"),
                email=_mask_email(email_by_user_id.get(r.get("userId") or "")),
                total_cost_microdollars=_si(r, "costMicrodollars"),
                total_input_tokens=_si(r, "inputTokens"),
                total_output_tokens=_si(r, "outputTokens"),
                request_count=_ca(r),
            )
            for r in by_user_groups
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
    model: str | None = None,
    block_name: str | None = None,
    tracking_type: str | None = None,
) -> tuple[list[CostLogRow], int]:
    if start is None:
        start = datetime.now(tz=timezone.utc) - timedelta(days=DEFAULT_DASHBOARD_DAYS)

    where = _build_prisma_where(
        start, end, provider, user_id, model, block_name, tracking_type
    )
    offset = (page - 1) * page_size

    total, rows = await asyncio.gather(
        PrismaLog.prisma().count(where=where),
        PrismaLog.prisma().find_many(
            where=where,
            include={"User": True},
            order=[{"createdAt": "desc"}, {"id": "desc"}],
            take=page_size,
            skip=offset,
        ),
    )

    logs = [
        CostLogRow(
            id=r.id,
            created_at=r.createdAt,
            user_id=r.userId,
            email=_mask_email(r.User.email if r.User else None),
            graph_exec_id=r.graphExecId,
            node_exec_id=r.nodeExecId,
            block_name=r.blockName or "",
            provider=r.provider,
            tracking_type=r.trackingType,
            cost_microdollars=r.costMicrodollars,
            input_tokens=r.inputTokens,
            output_tokens=r.outputTokens,
            cache_read_tokens=getattr(r, "cacheReadTokens", None),
            cache_creation_tokens=getattr(r, "cacheCreationTokens", None),
            duration=r.duration,
            model=r.model,
        )
        for r in rows
    ]
    return logs, total


EXPORT_MAX_ROWS = 100_000


async def get_platform_cost_logs_for_export(
    start: datetime | None = None,
    end: datetime | None = None,
    provider: str | None = None,
    user_id: str | None = None,
    model: str | None = None,
    block_name: str | None = None,
    tracking_type: str | None = None,
) -> tuple[list[CostLogRow], bool]:
    """Return all matching rows up to EXPORT_MAX_ROWS.

    Returns (rows, truncated) where truncated=True means the result was capped
    and the caller should warn the user that not all rows are included.
    """
    if start is None:
        start = datetime.now(tz=timezone.utc) - timedelta(days=DEFAULT_DASHBOARD_DAYS)

    where = _build_prisma_where(
        start, end, provider, user_id, model, block_name, tracking_type
    )

    rows = await PrismaLog.prisma().find_many(
        where=where,
        include={"User": True},
        order=[{"createdAt": "desc"}, {"id": "desc"}],
        take=EXPORT_MAX_ROWS + 1,
    )

    truncated = len(rows) > EXPORT_MAX_ROWS
    rows = rows[:EXPORT_MAX_ROWS]

    return [
        CostLogRow(
            id=r.id,
            created_at=r.createdAt,
            user_id=r.userId,
            email=_mask_email(r.User.email if r.User else None),
            graph_exec_id=r.graphExecId,
            node_exec_id=r.nodeExecId,
            block_name=r.blockName or "",
            provider=r.provider,
            tracking_type=r.trackingType,
            cost_microdollars=r.costMicrodollars,
            input_tokens=r.inputTokens,
            output_tokens=r.outputTokens,
            cache_read_tokens=getattr(r, "cacheReadTokens", None),
            cache_creation_tokens=getattr(r, "cacheCreationTokens", None),
            duration=r.duration,
            model=r.model,
        )
        for r in rows
    ], truncated


# ---------------------------------------------------------------------------
# Helpers kept for backward-compatibility with existing tests.
# New code should not use these — use _build_prisma_where instead.
# ---------------------------------------------------------------------------


def _build_where(
    start: datetime | None,
    end: datetime | None,
    provider: str | None,
    user_id: str | None,
    table_alias: str = "",
    model: str | None = None,
    block_name: str | None = None,
    tracking_type: str | None = None,
) -> tuple[str, list[Any]]:
    """Legacy SQL WHERE builder — retained so existing unit tests still pass.

    Only used by tests that verify the SQL-string generation logic. All
    production code uses _build_prisma_where instead.
    """
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
        clauses.append(f'{prefix}"provider" = ${idx}')
        params.append(provider.lower())
        idx += 1
    if user_id:
        clauses.append(f'{prefix}"userId" = ${idx}')
        params.append(user_id)
        idx += 1
    if model:
        clauses.append(f'{prefix}"model" = ${idx}')
        params.append(model)
        idx += 1
    if block_name:
        clauses.append(f'LOWER({prefix}"blockName") = LOWER(${idx})')
        params.append(block_name)
        idx += 1
    if tracking_type:
        clauses.append(f'{prefix}"trackingType" = ${idx}')
        params.append(tracking_type)
        idx += 1

    return (" AND ".join(clauses) if clauses else "TRUE", params)
