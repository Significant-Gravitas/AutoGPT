import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from prisma.models import PlatformCostLog as PrismaLog
from prisma.models import User as PrismaUser
from prisma.types import PlatformCostLogCreateInput, PlatformCostLogWhereInput
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema
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
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    request_count: int
    cost_bearing_request_count: int = 0


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


class CostBucket(BaseModel):
    bucket: str
    count: int


class PlatformCostDashboard(BaseModel):
    by_provider: list[ProviderCostSummary]
    by_user: list[UserCostSummary]
    total_cost_microdollars: int
    total_requests: int
    total_users: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_input_tokens_per_request: float = 0.0
    avg_output_tokens_per_request: float = 0.0
    avg_cost_microdollars_per_request: float = 0.0
    cost_p50_microdollars: float = 0.0
    cost_p75_microdollars: float = 0.0
    cost_p95_microdollars: float = 0.0
    cost_p99_microdollars: float = 0.0
    cost_buckets: list[CostBucket] = []


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


def _build_raw_where(
    start: datetime | None,
    end: datetime | None,
    provider: str | None,
    user_id: str | None,
    model: str | None = None,
    block_name: str | None = None,
    tracking_type: str | None = None,
) -> tuple[str, list]:
    """Build a parameterised WHERE clause for raw SQL queries.

    Mirrors the filter logic of ``_build_prisma_where`` so there is a single
    source of truth for which columns are filtered and how. The first clause
    always restricts to ``cost_usd`` tracking type unless *tracking_type* is
    explicitly provided by the caller.
    """
    params: list = []
    clauses: list[str] = []
    idx = 1

    # Always filter by tracking type — defaults to cost_usd for percentile /
    # bucket queries that only make sense on cost-denominated rows.
    tt = tracking_type if tracking_type is not None else "cost_usd"
    clauses.append(f'"trackingType" = ${idx}')
    params.append(tt)
    idx += 1

    if start is not None:
        clauses.append(f'"createdAt" >= ${idx}::timestamptz')
        params.append(start)
        idx += 1

    if end is not None:
        clauses.append(f'"createdAt" <= ${idx}::timestamptz')
        params.append(end)
        idx += 1

    if provider is not None:
        clauses.append(f'"provider" = ${idx}')
        params.append(provider.lower())
        idx += 1

    if user_id is not None:
        clauses.append(f'"userId" = ${idx}')
        params.append(user_id)
        idx += 1

    if model is not None:
        clauses.append(f'"model" = ${idx}')
        params.append(model)
        idx += 1

    if block_name is not None:
        clauses.append(f'LOWER("blockName") = LOWER(${idx})')
        params.append(block_name)
        idx += 1

    return (" AND ".join(clauses), params)


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

    # For per-user tracking-type breakdown we intentionally omit the
    # tracking_type filter so cost_usd and tokens rows are always present.
    # This ensures cost_bearing_request_count is correct even when the caller
    # is filtering the main view by a different tracking_type.
    where_no_tracking_type = _build_prisma_where(
        start, end, provider, user_id, model, block_name, tracking_type=None
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

    # Build parameterised WHERE clause for the raw SQL percentile/bucket
    # queries.  Uses _build_raw_where so filter logic is shared with
    # _build_prisma_where and only maintained in one place.
    # Always force tracking_type=None here so _build_raw_where defaults to
    # "cost_usd" — percentile and histogram queries only make sense on
    # cost-denominated rows, regardless of what the caller is filtering.
    raw_where, raw_params = _build_raw_where(
        start, end, provider, user_id, model, block_name, tracking_type=None
    )

    # Queries that always run regardless of tracking_type filter.
    common_queries = [
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
        # Per-user cost-bearing request count: group by (userId, trackingType)
        # so we can compute the correct denominator for per-user avg cost.
        # Uses where_no_tracking_type so cost_usd rows are always included
        # even when the caller filters the main view by a different tracking_type.
        PrismaLog.prisma().group_by(
            by=["userId", "trackingType"],
            where=where_no_tracking_type,
            count=True,
        ),
        # Distinct user count: group by userId, count groups.
        PrismaLog.prisma().group_by(
            by=["userId"],
            where=where,
            count=True,
        ),
        # Total aggregate (filtered): group by (provider, trackingType) so we can
        # compute cost-bearing and token-bearing denominators for avg stats.
        PrismaLog.prisma().group_by(
            by=["provider", "trackingType"],
            where=where,
            sum={
                "costMicrodollars": True,
                "inputTokens": True,
                "outputTokens": True,
            },
            count=True,
        ),
        # Percentile distribution of cost per request (respects all filters).
        query_raw_with_schema(
            "SELECT"
            "  percentile_cont(0.5) WITHIN GROUP"
            '    (ORDER BY "costMicrodollars") as p50,'
            "  percentile_cont(0.75) WITHIN GROUP"
            '    (ORDER BY "costMicrodollars") as p75,'
            "  percentile_cont(0.95) WITHIN GROUP"
            '    (ORDER BY "costMicrodollars") as p95,'
            "  percentile_cont(0.99) WITHIN GROUP"
            '    (ORDER BY "costMicrodollars") as p99'
            ' FROM {schema_prefix}"PlatformCostLog"'
            f" WHERE {raw_where}",
            *raw_params,
        ),
        # Histogram buckets for cost distribution (respects all filters).
        # NULL costMicrodollars is excluded explicitly to prevent such rows
        # from falling through all WHEN clauses into the ELSE '$10+' bucket.
        query_raw_with_schema(
            "SELECT"
            "  CASE"
            '    WHEN "costMicrodollars" < 500000'
            "      THEN '$0-0.50'"
            '    WHEN "costMicrodollars" < 1000000'
            "      THEN '$0.50-1'"
            '    WHEN "costMicrodollars" < 2000000'
            "      THEN '$1-2'"
            '    WHEN "costMicrodollars" < 5000000'
            "      THEN '$2-5'"
            '    WHEN "costMicrodollars" < 10000000'
            "      THEN '$5-10'"
            "    ELSE '$10+'"
            "  END as bucket,"
            "  COUNT(*) as count"
            ' FROM {schema_prefix}"PlatformCostLog"'
            f' WHERE {raw_where} AND "costMicrodollars" IS NOT NULL'
            " GROUP BY bucket"
            ' ORDER BY MIN("costMicrodollars")',
            *raw_params,
        ),
    ]

    # Only run the unfiltered aggregate query when tracking_type is set;
    # when tracking_type is None, the filtered query already contains all
    # tracking types and reusing it avoids a redundant full aggregation.
    if tracking_type is not None:
        common_queries.append(
            # Total aggregate (no tracking_type filter): used to compute
            # cost_bearing_requests and token_bearing_requests denominators so
            # global avg stats remain meaningful when the caller filters the
            # main view by a specific tracking_type (e.g. 'tokens').
            PrismaLog.prisma().group_by(
                by=["provider", "trackingType"],
                where=where_no_tracking_type,
                sum={
                    "costMicrodollars": True,
                    "inputTokens": True,
                    "outputTokens": True,
                },
                count=True,
            )
        )

    results = await asyncio.gather(*common_queries)

    # Unpack results by name for clarity.
    by_provider_groups = results[0]
    by_user_groups = results[1]
    by_user_tracking_groups = results[2]
    total_user_groups = results[3]
    total_agg_groups = results[4]
    percentile_rows = results[5]
    bucket_rows = results[6]
    # When tracking_type is None, the filtered and unfiltered queries are
    # identical — reuse total_agg_groups to avoid the extra DB round-trip.
    total_agg_no_tracking_type_groups = (
        results[7] if tracking_type is not None else total_agg_groups
    )

    # Compute token grand-totals from the unfiltered aggregate so they remain
    # consistent with the avg-token stats (which also use unfiltered data).
    # Using by_provider_groups here would give 0 tokens when tracking_type='cost_usd'
    # because cost_usd rows carry no token data, contradicting non-zero averages.
    total_input_tokens = sum(
        _si(r, "inputTokens")
        for r in total_agg_no_tracking_type_groups
        if r.get("trackingType") == "tokens"
    )
    total_output_tokens = sum(
        _si(r, "outputTokens")
        for r in total_agg_no_tracking_type_groups
        if r.get("trackingType") == "tokens"
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

    # Extract percentile values from the raw query result.
    pctl = percentile_rows[0] if percentile_rows else {}
    cost_p50 = float(pctl.get("p50") or 0)
    cost_p75 = float(pctl.get("p75") or 0)
    cost_p95 = float(pctl.get("p95") or 0)
    cost_p99 = float(pctl.get("p99") or 0)

    # Build cost bucket list.
    cost_buckets: list[CostBucket] = [
        CostBucket(bucket=r["bucket"], count=int(r["count"])) for r in bucket_rows
    ]

    # Avg-stat numerators and denominators are derived from the unfiltered
    # aggregate so they remain meaningful when the caller filters by a specific
    # tracking_type.  Example: filtering by 'tokens' excludes cost_usd rows from
    # total_agg_groups, so avg_cost would always be 0 if we used that; using
    # total_agg_no_tracking_type_groups gives the correct cost_usd total/count.
    avg_cost_total = sum(
        _si(r, "costMicrodollars")
        for r in total_agg_no_tracking_type_groups
        if r.get("trackingType") == "cost_usd"
    )
    cost_bearing_requests = sum(
        _ca(r)
        for r in total_agg_no_tracking_type_groups
        if r.get("trackingType") == "cost_usd"
    )
    avg_input_total = sum(
        _si(r, "inputTokens")
        for r in total_agg_no_tracking_type_groups
        if r.get("trackingType") == "tokens"
    )
    avg_output_total = sum(
        _si(r, "outputTokens")
        for r in total_agg_no_tracking_type_groups
        if r.get("trackingType") == "tokens"
    )
    # Token-bearing request count: only rows where trackingType == "tokens".
    # Token averages must use this denominator; cost_usd rows do not carry tokens.
    token_bearing_requests = sum(
        _ca(r)
        for r in total_agg_no_tracking_type_groups
        if r.get("trackingType") == "tokens"
    )

    # Per-user cost-bearing request count: used for per-user avg cost so the
    # denominator matches the numerator (cost_usd rows only, per user).
    user_cost_bearing_counts: dict[str, int] = {}
    for r in by_user_tracking_groups:
        if r.get("trackingType") == "cost_usd" and r.get("userId"):
            uid = r["userId"]
            user_cost_bearing_counts[uid] = user_cost_bearing_counts.get(uid, 0) + _ca(
                r
            )

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
                total_cache_read_tokens=_si(r, "cacheReadTokens"),
                total_cache_creation_tokens=_si(r, "cacheCreationTokens"),
                request_count=_ca(r),
                cost_bearing_request_count=user_cost_bearing_counts.get(
                    r.get("userId") or "", 0
                ),
            )
            for r in by_user_groups
        ],
        total_cost_microdollars=total_cost,
        total_requests=total_requests,
        total_users=total_users,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        avg_input_tokens_per_request=(
            avg_input_total / token_bearing_requests
            if token_bearing_requests > 0
            else 0.0
        ),
        avg_output_tokens_per_request=(
            avg_output_total / token_bearing_requests
            if token_bearing_requests > 0
            else 0.0
        ),
        avg_cost_microdollars_per_request=(
            avg_cost_total / cost_bearing_requests if cost_bearing_requests > 0 else 0.0
        ),
        cost_p50_microdollars=cost_p50,
        cost_p75_microdollars=cost_p75,
        cost_p95_microdollars=cost_p95,
        cost_p99_microdollars=cost_p99,
        cost_buckets=cost_buckets,
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
