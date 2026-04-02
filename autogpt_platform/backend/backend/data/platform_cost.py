import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from backend.data.db import query_raw_with_schema

logger = logging.getLogger(__name__)


class PlatformCostEntry(BaseModel):
    user_id: str
    graph_exec_id: str | None = None
    node_exec_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    block_id: str
    block_name: str
    provider: str
    credential_id: str
    cost_microdollars: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    data_size: int | None = None
    duration: float | None = None
    model: str | None = None
    metadata: dict[str, Any] | None = None


async def log_platform_cost(entry: PlatformCostEntry) -> None:
    await query_raw_with_schema(
        """
        INSERT INTO {schema_prefix}"PlatformCostLog"
            ("id", "createdAt", "userId", "graphExecId", "nodeExecId",
             "graphId", "nodeId", "blockId", "blockName", "provider",
             "credentialId", "costMicrodollars", "inputTokens", "outputTokens",
             "dataSize", "duration", "model", "metadata")
        VALUES (
            gen_random_uuid(), NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9,
            $10, $11, $12, $13, $14, $15, $16::jsonb
        )
        """,
        entry.user_id,
        entry.graph_exec_id,
        entry.node_exec_id,
        entry.graph_id,
        entry.node_id,
        entry.block_id,
        entry.block_name,
        entry.provider,
        entry.credential_id,
        entry.cost_microdollars,
        entry.input_tokens,
        entry.output_tokens,
        entry.data_size,
        entry.duration,
        entry.model,
        _json_or_none(entry.metadata),
    )


async def log_platform_cost_safe(entry: PlatformCostEntry) -> None:
    """Fire-and-forget wrapper that never raises."""
    try:
        await log_platform_cost(entry)
    except Exception:
        logger.exception(
            "Failed to log platform cost for user=%s provider=%s block=%s",
            entry.user_id,
            entry.provider,
            entry.block_name,
        )


def _json_or_none(data: dict[str, Any] | None) -> str | None:
    if data is None:
        return None
    return json.dumps(data)


class ProviderCostSummary(BaseModel):
    provider: str
    total_cost_microdollars: int
    total_input_tokens: int
    total_output_tokens: int
    request_count: int


class UserCostSummary(BaseModel):
    user_id: str
    email: str | None = None
    total_cost_microdollars: int
    total_input_tokens: int
    total_output_tokens: int
    request_count: int


class CostLogRow(BaseModel):
    id: str
    created_at: datetime
    user_id: str
    email: str | None = None
    graph_exec_id: str | None = None
    node_exec_id: str | None = None
    block_name: str
    provider: str
    cost_microdollars: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
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
        clauses.append(f'{prefix}"provider" = ${idx}')
        params.append(provider)
        idx += 1
    if user_id:
        clauses.append(f'{prefix}"userId" = ${idx}')
        params.append(user_id)
        idx += 1

    return (" AND ".join(clauses) if clauses else "TRUE", params)


async def get_platform_cost_dashboard(
    start: datetime | None = None,
    end: datetime | None = None,
    provider: str | None = None,
    user_id: str | None = None,
) -> PlatformCostDashboard:
    where_p, params_p = _build_where(start, end, provider, user_id, "p")

    by_provider_rows, user_count_rows, by_user_rows = await asyncio.gather(
        query_raw_with_schema(
            f"""
            SELECT
                p."provider",
                COALESCE(SUM(p."costMicrodollars"), 0)::bigint AS total_cost,
                COALESCE(SUM(p."inputTokens"), 0)::bigint AS total_input_tokens,
                COALESCE(SUM(p."outputTokens"), 0)::bigint AS total_output_tokens,
                COUNT(*)::bigint AS request_count
            FROM {{schema_prefix}}"PlatformCostLog" p
            WHERE {where_p}
            GROUP BY p."provider"
            ORDER BY total_cost DESC
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
            LIMIT 100
            """,
            *params_p,
        ),
    )

    total_users = user_count_rows[0]["cnt"] if user_count_rows else 0
    total_cost = sum(r["total_cost"] for r in by_provider_rows)
    total_requests = sum(r["request_count"] for r in by_provider_rows)

    return PlatformCostDashboard(
        by_provider=[
            ProviderCostSummary(
                provider=r["provider"],
                total_cost_microdollars=r["total_cost"],
                total_input_tokens=r["total_input_tokens"],
                total_output_tokens=r["total_output_tokens"],
                request_count=r["request_count"],
            )
            for r in by_provider_rows
        ],
        by_user=[
            UserCostSummary(
                user_id=r["user_id"],
                email=r.get("email"),
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
    where_sql, params = _build_where(start, end, provider, user_id, "p")

    count_rows = await query_raw_with_schema(
        f"""
        SELECT COUNT(*)::bigint AS cnt
        FROM {{schema_prefix}}"PlatformCostLog" p
        WHERE {where_sql}
        """,
        *params,
    )
    total = count_rows[0]["cnt"] if count_rows else 0

    offset = (page - 1) * page_size
    limit_idx = len(params) + 1
    offset_idx = len(params) + 2
    rows = await query_raw_with_schema(
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
            p."costMicrodollars" AS cost_microdollars,
            p."inputTokens" AS input_tokens,
            p."outputTokens" AS output_tokens,
            p."model"
        FROM {{schema_prefix}}"PlatformCostLog" p
        LEFT JOIN {{schema_prefix}}"User" u ON u."id" = p."userId"
        WHERE {where_sql}
        ORDER BY p."createdAt" DESC
        LIMIT ${limit_idx} OFFSET ${offset_idx}
        """,
        *params,
        page_size,
        offset,
    )

    logs = [
        CostLogRow(
            id=r["id"],
            created_at=r["created_at"],
            user_id=r["user_id"],
            email=r.get("email"),
            graph_exec_id=r.get("graph_exec_id"),
            node_exec_id=r.get("node_exec_id"),
            block_name=r["block_name"],
            provider=r["provider"],
            cost_microdollars=r.get("cost_microdollars"),
            input_tokens=r.get("input_tokens"),
            output_tokens=r.get("output_tokens"),
            model=r.get("model"),
        )
        for r in rows
    ]
    return logs, total
