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

import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel

from backend.copilot.bot.adapters.base import ChannelInfo, PlatformAdapter
from backend.copilot.bot.bot_backend import BotBackend

logger = logging.getLogger(__name__)

# Discord snowflakes are 17-19 digits today; allow a little slack so the ID
# path stays robust as the epoch advances rather than silently treating a
# valid ID as a channel name.
_SNOWFLAKE = re.compile(r"^\d{15,21}$")


class DeliveryResult(BaseModel):
    """Outcome of a proactive post, shaped for a tool/LLM to relay.

    ``error`` is a stable machine code (e.g. ``not_authorized``) so the
    caller can phrase its own user-facing message; ``ok`` is the only field
    callers must branch on.
    """

    ok: bool
    kind: Literal["message", "thread"]
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
    server_ids = tuple(await api.list_linked_server_ids(platform, user_id))
    if not server_ids:
        return []
    # Re-filter the adapter's output against the linked-server allowlist so the
    # picker can never surface a channel from an unlinked server, even if an
    # adapter over-returns. Mirrors the same guard in `_resolve_target`.
    allowed = set(server_ids)
    return [
        channel
        for channel in await adapter.list_text_channels(server_ids)
        if channel.server_id in allowed
    ]


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
    channel_id, error = await _resolve_target(adapter, api, platform, user_id, channel)
    if channel_id is None:
        return DeliveryResult(ok=False, kind="message", error=error)
    ref = await adapter.post_channel_message(channel_id, content)
    if ref is None:
        return DeliveryResult(
            ok=False, kind="message", channel_id=channel_id, error="send_failed"
        )
    return DeliveryResult(
        ok=True, kind="message", channel_id=channel_id, ref_id=ref.id, url=ref.url
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
    channel_id, error = await _resolve_target(adapter, api, platform, user_id, channel)
    if channel_id is None:
        return DeliveryResult(ok=False, kind="thread", error=error)
    ref = await adapter.create_channel_thread(channel_id, thread_name, content)
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
) -> tuple[Optional[str], Optional[str]]:
    """Resolve a channel reference to an authorized channel ID.

    Returns ``(channel_id, None)`` on success or ``(None, error_code)``. Both
    the ID path and the name path are constrained to the user's linked
    servers, so resolution and authorization are the same step.
    """
    server_ids = tuple(await api.list_linked_server_ids(platform, user_id))
    if not server_ids:
        return None, "no_linked_servers"

    ref = channel.strip().lstrip("#").strip()
    if not ref:
        return None, "channel_not_found"

    if _SNOWFLAKE.match(ref):
        guild_id = await adapter.get_channel_server_id(ref)
        if guild_id is None:
            return None, "channel_not_found"
        if guild_id not in server_ids:
            return None, "not_authorized"
        return ref, None

    # `c.server_id in server_ids` is redundant with passing `server_ids` to
    # the adapter, but re-asserting it here keeps authorization correct even
    # if an adapter ever returns channels outside the requested servers.
    matches = [
        c
        for c in await adapter.list_text_channels(server_ids)
        if c.name.lower() == ref.lower() and c.server_id in server_ids
    ]
    if not matches:
        return None, "channel_not_found"
    if len(matches) > 1:
        return None, "ambiguous_channel"
    return matches[0].id, None
