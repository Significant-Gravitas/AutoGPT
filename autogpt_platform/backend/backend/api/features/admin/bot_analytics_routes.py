"""Admin read API for bot usage analytics.

All endpoints are admin-gated (router-level ``requires_admin_user``) and
read-only. They surface aggregate counts, server presence and performance
metrics for every live bot — never message content or user identity (the
underlying schema stores neither).
"""

import logging

from autogpt_libs.auth import requires_admin_user
from fastapi import APIRouter, Query, Security

from backend.data.bot_analytics_reads import (
    BotAnalyticsSummary,
    BotCommandUsage,
    BotGuildInfo,
    BotServerActivity,
    BotServerCountPoint,
    BotTimeseriesPoint,
    get_bot_analytics_summary,
    get_bot_command_usage,
    get_bot_message_timeseries,
    get_bot_server_timeseries,
    get_bot_top_servers,
    list_bot_guilds,
)
from backend.platform_linking.models import Platform

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/bot-analytics",
    tags=["bot-analytics"],
    dependencies=[Security(requires_admin_user)],
)


def _platform_value(platform: Platform | None) -> str | None:
    return platform.value if platform else None


@router.get("/summary", response_model=BotAnalyticsSummary, summary="Bot Usage Summary")
async def bot_summary(
    platform: Platform | None = Query(None),
    days: int = Query(30, ge=1, le=365),
) -> BotAnalyticsSummary:
    return await get_bot_analytics_summary(_platform_value(platform), days)


@router.get(
    "/timeseries",
    response_model=list[BotTimeseriesPoint],
    summary="Bot Message Timeseries",
)
async def bot_timeseries(
    platform: Platform | None = Query(None),
    days: int = Query(30, ge=1, le=365),
) -> list[BotTimeseriesPoint]:
    return await get_bot_message_timeseries(_platform_value(platform), days)


@router.get(
    "/server-timeseries",
    response_model=list[BotServerCountPoint],
    summary="Bot Server-Count Timeseries (sharding curve)",
)
async def bot_server_timeseries(
    platform: Platform | None = Query(None),
    days: int = Query(30, ge=1, le=365),
) -> list[BotServerCountPoint]:
    return await get_bot_server_timeseries(_platform_value(platform), days)


@router.get(
    "/top-servers",
    response_model=list[BotServerActivity],
    summary="Top Servers by Activity",
)
async def bot_top_servers(
    platform: Platform | None = Query(None),
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(20, ge=1, le=100),
) -> list[BotServerActivity]:
    return await get_bot_top_servers(_platform_value(platform), days, limit)


@router.get(
    "/command-usage",
    response_model=list[BotCommandUsage],
    summary="Command Usage Breakdown",
)
async def bot_command_usage(
    platform: Platform | None = Query(None),
    days: int = Query(30, ge=1, le=365),
) -> list[BotCommandUsage]:
    return await get_bot_command_usage(_platform_value(platform), days)


@router.get("/guilds", response_model=list[BotGuildInfo], summary="Bot Server Roster")
async def bot_guilds(
    platform: Platform | None = Query(None),
    include_left: bool = Query(False),
    limit: int = Query(500, ge=1, le=2000),
) -> list[BotGuildInfo]:
    return await list_bot_guilds(_platform_value(platform), include_left, limit)
