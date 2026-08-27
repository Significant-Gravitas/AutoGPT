"""Proactive (backend → platform) message delivery.

Turns an *already-authenticated* "post this to a channel" request into
adapter calls. Two concerns live here, both platform-agnostic:

- **Authorization** — a user may only post into channels that belong to a
  server they've linked (``BotBackend.list_linked_server_ids``). Channel
  resolution is funneled through that allowlist so an unauthorized target can
  never be reached, whether referenced by ID or by name.
- **Resolution** — a human channel reference (``#announcements``, a bare
  name, or a raw snowflake ID) is mapped to a concrete channel ID.

The adapter owns everything platform-specific (enumerating channels, sending,
thread creation); this module never imports ``discord``.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Literal, Optional, TypeVar

from pydantic import BaseModel

from backend.copilot.bot.adapters.base import ChannelInfo, PlatformAdapter
from backend.copilot.bot.bot_backend import BotBackend

logger = logging.getLogger(__name__)
OUTBOUND_ACTION_TIMEOUT_SECONDS = 30
OUTBOUND_LEASE_POLL_SECONDS = 0.1
T = TypeVar("T")


class PlatformLinkLeaseLostError(RuntimeError):
    pass


class DeliveryResult(BaseModel):
    """Outcome of a proactive post, shaped for a tool/LLM to relay.

    ``error`` is a stable machine code (e.g. ``not_authorized``) so the
    caller can phrase its own user-facing message; ``ok`` is the only field
    callers must branch on.
    """

    ok: bool
    kind: Literal["message", "thread", "dm"]
    channel_id: Optional[str] = None
    ref_id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None


async def list_channels(
    adapter: PlatformAdapter,
    api: BotBackend,
    platform: str,
    user_id: str,
) -> list[ChannelInfo]:
    """List channels ``user_id`` may post to via the bot on ``platform``."""
    server_ids = tuple(sorted(set(await api.list_linked_server_ids(platform, user_id))))
    if not server_ids:
        return []
    channels: list[ChannelInfo] = []
    for server_id in server_ids:
        async with _outbound_link_lease(
            api, platform, user_id, server_id, None
        ) as lease_id:
            if lease_id is None:
                continue
            try:
                listed = await _run_outbound_action(
                    api,
                    lease_id,
                    adapter.list_text_channels((server_id,)),
                )
            except (TimeoutError, PlatformLinkLeaseLostError) as error:
                logger.warning("Channel enumeration aborted: %s", error)
                continue
            channels.extend(
                channel for channel in listed if channel.server_id == server_id
            )
    return channels


async def deliver_message(
    adapter: PlatformAdapter,
    api: BotBackend,
    platform: str,
    user_id: str,
    channel: str,
    content: str,
) -> DeliveryResult:
    """Post ``content`` to ``channel`` (name or ID) as a standalone message."""
    if not content or not content.strip():
        return DeliveryResult(ok=False, kind="message", error="empty_content")
    channel_id, server_id, error = await _resolve_target(
        adapter, api, platform, user_id, channel
    )
    if channel_id is None:
        return DeliveryResult(ok=False, kind="message", error=error)
    assert server_id is not None
    async with _outbound_link_lease(
        api, platform, user_id, server_id, None
    ) as lease_id:
        if lease_id is None:
            return DeliveryResult(ok=False, kind="message", error="not_authorized")
        try:
            ref = await _run_outbound_action(
                api,
                lease_id,
                adapter.post_channel_message(channel_id, content),
            )
        except (TimeoutError, PlatformLinkLeaseLostError) as error:
            logger.warning("Outbound platform message aborted: %s", error)
            ref = None
    if ref is None:
        return DeliveryResult(
            ok=False, kind="message", channel_id=channel_id, error="send_failed"
        )
    return DeliveryResult(
        ok=True, kind="message", channel_id=channel_id, ref_id=ref.id, url=ref.url
    )


async def deliver_dm(
    adapter: PlatformAdapter,
    api: BotBackend,
    platform: str,
    user_id: str,
    content: str,
) -> DeliveryResult:
    """Send ``content`` to ``user_id``'s own DM with the bot.

    Authorization is the DM link itself: the target is always the calling
    user's linked platform account, never a caller-supplied recipient — so a
    user can only ever DM themself.
    """
    if not content or not content.strip():
        return DeliveryResult(ok=False, kind="dm", error="empty_content")
    platform_user_id = await api.get_dm_user_id(platform, user_id)
    if platform_user_id is None:
        return DeliveryResult(ok=False, kind="dm", error="no_dm_link")
    async with _outbound_link_lease(
        api,
        platform,
        user_id,
        None,
        platform_user_id,
    ) as lease_id:
        if lease_id is None:
            return DeliveryResult(ok=False, kind="dm", error="no_dm_link")
        try:
            channel_id = await _run_outbound_action(
                api,
                lease_id,
                adapter.open_dm_channel(platform_user_id),
            )
        except (TimeoutError, PlatformLinkLeaseLostError) as error:
            logger.warning("Outbound DM channel open aborted: %s", error)
            channel_id = None
        if channel_id is None:
            return DeliveryResult(ok=False, kind="dm", error="dm_unavailable")
        try:
            ref = await _run_outbound_action(
                api,
                lease_id,
                adapter.post_channel_message(channel_id, content),
            )
        except (TimeoutError, PlatformLinkLeaseLostError) as error:
            logger.warning("Outbound DM delivery aborted: %s", error)
            ref = None
    if ref is None:
        return DeliveryResult(
            ok=False, kind="dm", channel_id=channel_id, error="send_failed"
        )
    return DeliveryResult(
        ok=True, kind="dm", channel_id=channel_id, ref_id=ref.id, url=ref.url
    )


async def create_thread(
    adapter: PlatformAdapter,
    api: BotBackend,
    platform: str,
    user_id: str,
    channel: str,
    thread_name: str,
    content: str,
) -> DeliveryResult:
    """Create a standalone thread in ``channel`` and post ``content`` in it."""
    if not content or not content.strip():
        return DeliveryResult(ok=False, kind="thread", error="empty_content")
    channel_id, server_id, error = await _resolve_target(
        adapter, api, platform, user_id, channel
    )
    if channel_id is None:
        return DeliveryResult(ok=False, kind="thread", error=error)
    assert server_id is not None
    async with _outbound_link_lease(
        api, platform, user_id, server_id, None
    ) as lease_id:
        if lease_id is None:
            return DeliveryResult(ok=False, kind="thread", error="not_authorized")
        try:
            ref = await _run_outbound_action(
                api,
                lease_id,
                adapter.create_channel_thread(channel_id, thread_name, content),
            )
        except (TimeoutError, PlatformLinkLeaseLostError) as error:
            logger.warning("Outbound platform thread creation aborted: %s", error)
            ref = None
    if ref is None:
        return DeliveryResult(
            ok=False, kind="thread", channel_id=channel_id, error="thread_failed"
        )
    return DeliveryResult(
        ok=True, kind="thread", channel_id=channel_id, ref_id=ref.id, url=ref.url
    )


async def _resolve_target(
    adapter: PlatformAdapter,
    api: BotBackend,
    platform: str,
    user_id: str,
    channel: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve a channel reference to an authorized channel ID.

    Returns ``(channel_id, None)`` on success or ``(None, error_code)``. Both
    the ID path and the name path are constrained to the user's linked
    servers, so resolution and authorization are the same step.
    """
    server_ids = tuple(await api.list_linked_server_ids(platform, user_id))
    if not server_ids:
        return None, None, "no_linked_servers"

    ref = channel.strip().lstrip("#").strip()
    if not ref:
        return None, None, "channel_not_found"

    if adapter.looks_like_channel_id(ref):
        guild_id = await adapter.get_channel_server_id(ref)
        if guild_id is None:
            return None, None, "channel_not_found"
        if guild_id not in server_ids:
            return None, None, "not_authorized"
        return ref, guild_id, None

    # `c.server_id in server_ids` is redundant with passing `server_ids` to
    # the adapter, but re-asserting it here keeps authorization correct even
    # if an adapter ever returns channels outside the requested servers.
    matches = [
        c
        for c in await adapter.list_text_channels(server_ids)
        if c.name.lower() == ref.lower() and c.server_id in server_ids
    ]
    if not matches:
        return None, None, "channel_not_found"
    if len(matches) > 1:
        return None, None, "ambiguous_channel"
    return matches[0].id, matches[0].server_id, None


@asynccontextmanager
async def _outbound_link_lease(
    api: BotBackend,
    platform: str,
    user_id: str,
    platform_server_id: str | None,
    platform_user_id: str | None,
) -> AsyncIterator[str | None]:
    lease_id = await api.acquire_platform_link_lease(
        platform,
        platform_server_id,
        platform_user_id or "",
    )
    try:
        if platform_server_id is not None:
            active = platform_server_id in await api.list_linked_server_ids(
                platform, user_id
            )
        else:
            active = await api.get_dm_user_id(platform, user_id) == platform_user_id
        yield lease_id if active else None
    finally:
        released = await api.release_platform_link_lease(lease_id)
        if not released:
            logger.error("Outbound platform link lease expired during delivery")


async def _run_outbound_action(
    api: BotBackend,
    lease_id: str,
    action: Awaitable[T],
) -> T:
    task = asyncio.ensure_future(action)
    try:
        async with asyncio.timeout(OUTBOUND_ACTION_TIMEOUT_SECONDS):
            while not task.done():
                await asyncio.wait({task}, timeout=OUTBOUND_LEASE_POLL_SECONDS)
                if task.done():
                    break
                if not await api.is_platform_link_lease_active(lease_id):
                    raise PlatformLinkLeaseLostError(
                        "platform link authorization expired"
                    )
            result = await task
            if not await api.is_platform_link_lease_active(lease_id):
                raise PlatformLinkLeaseLostError("platform link authorization expired")
            return result
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
