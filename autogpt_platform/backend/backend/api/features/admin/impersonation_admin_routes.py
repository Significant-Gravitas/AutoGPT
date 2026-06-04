"""Admin endpoint that emits a Discord audit alert when an admin begins
impersonating a user from the admin dashboard.

Design (see plan): the alert GATES impersonation. The frontend awaits this
endpoint before swapping identity, so:

- No Discord bot token configured -> skip the alert and allow the swap (200,
  alerted=False). Keeps non-Discord / self-hosted deployments working.
- Token configured + alert delivered -> allow the swap (200, alerted=True).
- Token configured + alert NOT delivered -> 502, which blocks the swap. We would
  rather kill an admin's ability to impersonate than allow it with no audit
  alert.

`discord_send_alert` does NOT raise on every failure -- a bad token raises, but a
missing/misconfigured channel merely returns a status string. So delivery is
treated as successful ONLY when the returned status is the block's success
sentinel ("Message sent"); any other status raises here. A bounded timeout keeps
a slow Discord gateway login from holding the request (and the admin's click)
open, and never crashes the service.
"""

import asyncio
import logging
from typing import Optional

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Body, HTTPException, Security
from pydantic import BaseModel

from backend.data.user import get_user_email_by_id
from backend.util.metrics import DiscordChannel, discord_send_alert
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Success sentinel returned by SendDiscordMessageBlock on a delivered message
# (backend/blocks/discord/bot_blocks.py). Anything else means "not delivered".
_DISCORD_SENT_STATUS = "Message sent"

# discord_send_alert opens a full discord.Client (client.start) per call, so a
# send can take a few seconds; bound it generously to avoid false blocks.
_DISCORD_ALERT_TIMEOUT_SECONDS = 10.0

router = APIRouter(
    prefix="/admin",
    tags=["admin", "impersonation"],
    dependencies=[Security(requires_admin_user)],
)


class ImpersonationNotifyRequest(BaseModel):
    target_user_id: str


class ImpersonationNotifyResponse(BaseModel):
    # True when the Discord alert was delivered. False only when no bot token is
    # configured (alert intentionally skipped). A failed-but-configured alert
    # never returns False -- it raises 502 so the caller blocks the swap.
    alerted: bool


@router.post(
    "/impersonation/notify",
    response_model=ImpersonationNotifyResponse,
    summary="Notify Impersonation Start",
)
async def notify_impersonation_start(
    body: ImpersonationNotifyRequest = Body(...),
    admin_user_id: str = Security(get_user_id),
) -> ImpersonationNotifyResponse:
    """Emit a Discord audit alert when an admin starts impersonating a user.

    Admin-only (router dependency). The alert gates impersonation: callers must
    treat a non-2xx response as "do not impersonate".
    """
    target_user_id = body.target_user_id

    # Always record a server-side trail, independent of Discord and of outcome.
    logger.info(
        f"Admin impersonation requested: {admin_user_id} acting as user "
        f"{target_user_id} (via admin dashboard)"
    )

    if not settings.secrets.discord_bot_token:
        logger.debug("Discord bot token not configured; skipping impersonation alert")
        return ImpersonationNotifyResponse(alerted=False)

    try:
        await _send_impersonation_alert(admin_user_id, target_user_id)
    except Exception:
        logger.warning(
            f"Failed to deliver impersonation Discord alert "
            f"(admin={admin_user_id} target={target_user_id})",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail="Could not deliver impersonation audit alert; impersonation blocked",
        )

    return ImpersonationNotifyResponse(alerted=True)


async def _send_impersonation_alert(admin_user_id: str, target_user_id: str) -> None:
    """Send the Discord alert. Raises if it was not actually delivered."""
    admin_email = await _resolve_email(admin_user_id)
    target_email = await _resolve_email(target_user_id)

    content = (
        "🕵️ **Admin impersonation started**\n"
        f"Admin: {admin_email or 'unknown'} (`{admin_user_id}`)\n"
        f"Now viewing as: {target_email or 'unknown'} (`{target_user_id}`)\n"
        "Source: admin dashboard"
    )

    status = await asyncio.wait_for(
        discord_send_alert(content, DiscordChannel.PLATFORM),
        timeout=_DISCORD_ALERT_TIMEOUT_SECONDS,
    )
    if status != _DISCORD_SENT_STATUS:
        raise RuntimeError(f"Discord alert not delivered (status: {status!r})")


async def _resolve_email(user_id: str) -> Optional[str]:
    """Best-effort email lookup for a friendlier alert. Never raises."""
    try:
        return await get_user_email_by_id(user_id)
    except Exception:
        logger.warning("Failed to resolve email for user %s", user_id, exc_info=True)
        return None
