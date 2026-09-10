"""Pending-install markers: who installed a workspace before linking a DM.

The closed-loop install flow needs the Bots settings page to say "you added
the bot to workspace X — one step left" before any account link exists. The
OAuth install is the only moment that ties a platform user to a workspace, so
the callback drops a short-lived marker here and the settings/confirm routes
read it back. Redis with a TTL: this is journey state, not a record.
"""

import json
import logging

from pydantic import BaseModel, ValidationError

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_TTL_SECONDS = 7 * 24 * 3600


class PendingSlackInstall(BaseModel):
    team_id: str
    team_name: str | None = None
    app_id: str | None = None


def bot_dm_url(app_id: str, team_id: str) -> str:
    """Deep link that lands the user in the bot's DM in the Slack client."""
    return f"https://slack.com/app_redirect?app={app_id}&team={team_id}"


def _key(user_id: str) -> str:
    return f"copilot:bot:slack:pending-install:{user_id}"


async def mark_pending(user_id: str, install: PendingSlackInstall) -> None:
    """Best-effort: losing the marker degrades UX, never the flow."""
    try:
        redis = await get_redis_async()
        await redis.set(_key(user_id), install.model_dump_json(), ex=_TTL_SECONDS)
    except Exception:
        logger.warning("Could not record pending Slack install", exc_info=True)


async def get_pending(user_id: str) -> PendingSlackInstall | None:
    try:
        redis = await get_redis_async()
        raw = await redis.get(_key(user_id))
    except Exception:
        logger.warning("Could not read pending Slack install", exc_info=True)
        return None
    if not raw:
        return None
    try:
        return PendingSlackInstall.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError):
        return None


async def clear_pending(user_id: str) -> None:
    try:
        redis = await get_redis_async()
        await redis.delete(_key(user_id))
    except Exception:
        logger.warning("Could not clear pending Slack install", exc_info=True)
