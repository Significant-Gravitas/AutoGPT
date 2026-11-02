"""Bot usage analytics writes (privacy-preserving).

Records discrete bot usage events and per-server presence for the admin
analytics page. NEVER stores message content — only counts, bounded enums,
timestamps and numeric metrics.

Directly accessed by the ``DatabaseManager`` pod (which holds the Prisma
connection). Other services (notably the out-of-process bot) reach these
through ``backend.data.db_accessors.bot_analytics_db`` so calls are
transparently routed via ``DatabaseManagerAsyncClient``.
"""

import logging
from datetime import datetime, timezone

from prisma.models import BotEvent, BotGuild

from backend.platform_linking.models import BotEventInput, BotGuildInput, Platform

logger = logging.getLogger(__name__)


async def record_bot_event(event: BotEventInput) -> None:
    await BotEvent.prisma().create(
        data={
            "platform": event.platform.value,
            "eventType": event.event_type,
            "serverId": event.server_id,
            "channelType": event.channel_type,
            "commandName": event.command_name,
            "errorKind": event.error_kind,
            "charCount": event.char_count,
            "durationMs": event.duration_ms,
        }
    )


async def record_guild_joined(guild: BotGuildInput) -> None:
    await BotGuild.prisma().upsert(
        where={
            "platform_serverId": {
                "platform": guild.platform.value,
                "serverId": guild.server_id,
            }
        },
        data={
            "create": {
                "platform": guild.platform.value,
                "serverId": guild.server_id,
                "name": guild.name,
            },
            "update": {
                "name": guild.name,
                "leftAt": None,
                "lastSeenAt": datetime.now(timezone.utc),
            },
        },
    )


async def mark_guild_left(platform: Platform, server_id: str) -> None:
    await BotGuild.prisma().update_many(
        where={
            "platform": platform.value,
            "serverId": server_id,
            "leftAt": None,
        },
        data={"leftAt": datetime.now(timezone.utc)},
    )


async def sync_guild_presence(platform: Platform, guilds: list[BotGuildInput]) -> None:
    """Full presence reconcile (on connect): upsert every server the bot is in
    now and mark any previously-joined server it is no longer in as left."""
    for guild in guilds:
        await record_guild_joined(guild)

    present = {guild.server_id for guild in guilds}
    joined = await BotGuild.prisma().find_many(
        where={"platform": platform.value, "leftAt": None}
    )
    stale_ids = [row.id for row in joined if row.serverId not in present]
    if stale_ids:
        await BotGuild.prisma().update_many(
            where={"id": {"in": stale_ids}},
            data={"leftAt": datetime.now(timezone.utc)},
        )
