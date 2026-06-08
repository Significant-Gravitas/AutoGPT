"""Bot analytics read/aggregate queries for the admin dashboard.

Read-only counterpart to ``bot_analytics`` (writes). Runs in-process against
Prisma via ``query_raw_with_schema`` — the same direct-call pattern as
``platform_cost`` — so the admin REST routes call these without RPC.

Privacy: these queries only ever aggregate counts, durations and server-level
rows. There is no message content or user identity anywhere in the schema.
"""

import logging
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from backend.data.db import query_raw_with_schema

logger = logging.getLogger(__name__)


class BotAnalyticsSummary(BaseModel):
    platform: str | None
    window_days: int
    live_servers: int = Field(
        description=(
            "Current count of servers the bot is in right now. "
            "Point-in-time gauge — intentionally ignores window_days. "
            "Use the server-count timeseries endpoint for the windowed growth curve."
        ),
    )
    messages_received: int
    replies_sent: int
    commands_used: int
    stream_errors: int
    avg_reply_ms: float | None
    error_rate: float


class BotTimeseriesPoint(BaseModel):
    date: datetime
    messages: int
    replies: int
    errors: int


class BotServerCountPoint(BaseModel):
    date: datetime
    server_count: int


class BotServerActivity(BaseModel):
    server_id: str
    name: str | None
    messages: int
    commands: int


class BotCommandUsage(BaseModel):
    command: str
    uses: int


class BotGuildInfo(BaseModel):
    platform: str
    server_id: str
    name: str | None
    joined_at: datetime
    left_at: datetime | None
    active: bool


def _since(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def _platform_filter(platform: str | None, params: list, *, alias: str = "") -> str:
    """Append a platform value to ``params`` and return a SQL fragment."""
    if not platform:
        return ""
    params.append(platform)
    col = f'{alias}"platform"' if alias else '"platform"'
    return f"AND {col}::text = ${len(params)}"


async def get_bot_analytics_summary(
    platform: str | None = None, days: int = 30
) -> BotAnalyticsSummary:
    event_params: list = [_since(days)]
    event_filter = _platform_filter(platform, event_params)
    event_rows = await query_raw_with_schema(
        f"""
        SELECT
            count(*) FILTER (WHERE "eventType" = 'message_received') AS messages,
            count(*) FILTER (WHERE "eventType" = 'reply_sent') AS replies,
            count(*) FILTER (WHERE "eventType" = 'command_used') AS commands,
            count(*) FILTER (WHERE "eventType" = 'stream_error') AS errors,
            avg("durationMs") FILTER (WHERE "eventType" = 'reply_sent') AS avg_reply_ms
        FROM {{schema_prefix}}"BotEvent"
        WHERE "createdAt" >= $1::timestamptz {event_filter}
        """,
        *event_params,
    )
    row = event_rows[0] if event_rows else {}

    guild_params: list = []
    guild_filter = _platform_filter(platform, guild_params)
    guild_rows = await query_raw_with_schema(
        f"""
        SELECT count(*) AS live
        FROM {{schema_prefix}}"BotGuild"
        WHERE "leftAt" IS NULL {guild_filter}
        """,
        *guild_params,
    )

    messages = int(row.get("messages") or 0)
    errors = int(row.get("errors") or 0)
    avg_reply_ms = row.get("avg_reply_ms")
    return BotAnalyticsSummary(
        platform=platform,
        window_days=days,
        live_servers=int(guild_rows[0]["live"]) if guild_rows else 0,
        messages_received=messages,
        replies_sent=int(row.get("replies") or 0),
        commands_used=int(row.get("commands") or 0),
        stream_errors=errors,
        avg_reply_ms=float(avg_reply_ms) if avg_reply_ms is not None else None,
        error_rate=(errors / messages) if messages else 0.0,
    )


async def get_bot_message_timeseries(
    platform: str | None = None, days: int = 30
) -> list[BotTimeseriesPoint]:
    params: list = [_since(days)]
    platform_filter = _platform_filter(platform, params)
    # date_trunc on a timestamptz uses the session's time zone by default —
    # convert to UTC explicitly via AT TIME ZONE so days are bucketed by UTC
    # regardless of the connection's TZ (mirrors platform_cost's pattern).
    rows = await query_raw_with_schema(
        f"""
        SELECT
            date_trunc('day', "createdAt" AT TIME ZONE 'UTC') AT TIME ZONE 'UTC' AS day,
            count(*) FILTER (WHERE "eventType" = 'message_received') AS messages,
            count(*) FILTER (WHERE "eventType" = 'reply_sent') AS replies,
            count(*) FILTER (WHERE "eventType" = 'stream_error') AS errors
        FROM {{schema_prefix}}"BotEvent"
        WHERE "createdAt" >= $1::timestamptz {platform_filter}
        GROUP BY day
        ORDER BY day
        """,
        *params,
    )
    return [
        BotTimeseriesPoint(
            date=row["day"],
            messages=int(row["messages"]),
            replies=int(row["replies"]),
            errors=int(row["errors"]),
        )
        for row in rows
    ]


async def get_bot_server_timeseries(
    platform: str | None = None, days: int = 30
) -> list[BotServerCountPoint]:
    """Cumulative live-server count at the end of each day — the curve to watch
    against the sharding threshold."""
    params: list = [days]
    joined_filter = _platform_filter(platform, params, alias="g.")
    left_filter = joined_filter  # same param ($2) reused in both subqueries
    # The BotGuild.joinedAt / leftAt columns are `timestamp` (no tz) storing
    # naive UTC values (Prisma's default for DateTime). `now() AT TIME ZONE 'utc'`
    # also produces a naive UTC `timestamp`, so date_trunc here uses neither the
    # session TZ nor any implicit cast — comparisons stay naive-to-naive and the
    # daily buckets line up with UTC days regardless of session TZ.
    rows = await query_raw_with_schema(
        f"""
        SELECT
            d.day,
            (
                SELECT count(*) FROM {{schema_prefix}}"BotGuild" g
                WHERE g."joinedAt" < d.day + interval '1 day' {joined_filter}
            ) - (
                SELECT count(*) FROM {{schema_prefix}}"BotGuild" g
                WHERE g."leftAt" IS NOT NULL
                    AND g."leftAt" < d.day + interval '1 day' {left_filter}
            ) AS server_count
        FROM generate_series(
            date_trunc('day', now() AT TIME ZONE 'utc') - ($1::int - 1) * interval '1 day',
            date_trunc('day', now() AT TIME ZONE 'utc'),
            interval '1 day'
        ) AS d(day)
        ORDER BY d.day
        """,
        *params,
    )
    return [
        BotServerCountPoint(
            date=row["day"], server_count=max(0, int(row["server_count"]))
        )
        for row in rows
    ]


async def get_bot_top_servers(
    platform: str | None = None, days: int = 30, limit: int = 20
) -> list[BotServerActivity]:
    params: list = [_since(days)]
    platform_filter = _platform_filter(platform, params, alias="e.")
    params.append(limit)
    rows = await query_raw_with_schema(
        f"""
        SELECT
            e."serverId" AS server_id,
            g."name" AS name,
            count(*) FILTER (WHERE e."eventType" = 'message_received') AS messages,
            count(*) FILTER (WHERE e."eventType" = 'command_used') AS commands
        FROM {{schema_prefix}}"BotEvent" e
        LEFT JOIN {{schema_prefix}}"BotGuild" g
            ON g."serverId" = e."serverId" AND g."platform" = e."platform"
        WHERE e."createdAt" >= $1::timestamptz AND e."serverId" IS NOT NULL {platform_filter}
        GROUP BY e."serverId", g."name"
        ORDER BY messages DESC
        LIMIT ${len(params)}::int
        """,
        *params,
    )
    return [
        BotServerActivity(
            server_id=row["server_id"],
            name=row["name"],
            messages=int(row["messages"]),
            commands=int(row["commands"]),
        )
        for row in rows
    ]


async def get_bot_command_usage(
    platform: str | None = None, days: int = 30
) -> list[BotCommandUsage]:
    params: list = [_since(days)]
    platform_filter = _platform_filter(platform, params)
    rows = await query_raw_with_schema(
        f"""
        SELECT "commandName" AS command, count(*) AS uses
        FROM {{schema_prefix}}"BotEvent"
        WHERE "eventType" = 'command_used'
            AND "commandName" IS NOT NULL
            AND "createdAt" >= $1::timestamptz {platform_filter}
        GROUP BY "commandName"
        ORDER BY uses DESC
        """,
        *params,
    )
    return [
        BotCommandUsage(command=row["command"], uses=int(row["uses"])) for row in rows
    ]


async def list_bot_guilds(
    platform: str | None = None, include_left: bool = False, limit: int = 500
) -> list[BotGuildInfo]:
    params: list = []
    platform_filter = _platform_filter(platform, params)
    active_filter = "" if include_left else 'AND "leftAt" IS NULL'
    params.append(limit)
    rows = await query_raw_with_schema(
        f"""
        SELECT "platform", "serverId" AS server_id, "name", "joinedAt", "leftAt"
        FROM {{schema_prefix}}"BotGuild"
        WHERE 1 = 1 {platform_filter} {active_filter}
        ORDER BY ("leftAt" IS NULL) DESC, "joinedAt" DESC
        LIMIT ${len(params)}::int
        """,
        *params,
    )
    return [
        BotGuildInfo(
            platform=row["platform"],
            server_id=row["server_id"],
            name=row["name"],
            joined_at=row["joinedAt"],
            left_at=row["leftAt"],
            active=row["leftAt"] is None,
        )
        for row in rows
    ]
