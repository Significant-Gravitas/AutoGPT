"""Per-user heartbeat settings and the pure rules around them.

Stored as one JSON object under ``User.metadata["copilot_heartbeat"]``, so no
migration: the column already exists and the data layer stores the dict as
given. Everything that interprets it lives here.
"""

import logging
import re
from datetime import datetime, time
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.copilot.config import CopilotLLMModel
from backend.data.db_accessors import user_db
from backend.util.timezone_utils import get_user_timezone_or_utc

logger = logging.getLogger(__name__)

ChatPlatformName = Literal["discord", "slack", "telegram", "teams"]

DEFAULT_INTERVAL_MINUTES = 30
MIN_INTERVAL_MINUTES = 10
MAX_INTERVAL_MINUTES = 24 * 60
MAX_CHECKLIST_CHARS = 8_000
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class HeartbeatDelivery(BaseModel):
    """Where an alert goes."""

    # The user's main AutoPilot thread, where the morning briefing lands.
    copilot_thread: bool = True
    # The WebSocket notification, which the notification bus also fans out
    # as a web push.
    push: bool = True
    # Linked chat platforms to DM the alert to (the bot must be linked).
    chat_platforms: list[ChatPlatformName] = Field(default_factory=list)


class HeartbeatConfig(BaseModel):
    enabled: bool = False
    interval_minutes: int = Field(
        default=DEFAULT_INTERVAL_MINUTES,
        ge=MIN_INTERVAL_MINUTES,
        le=MAX_INTERVAL_MINUTES,
    )
    # Local wall-clock window, ``HH:MM``. ``start == end`` means all day; a
    # window whose end is before its start runs across midnight.
    active_hours_start: str = "08:00"
    active_hours_end: str = "22:00"
    # IANA name; None follows the user's profile timezone.
    timezone: str | None = None
    # The cheap tier by default: most beats end in NO_REPLY.
    model_tier: CopilotLLMModel = "standard"
    delivery: HeartbeatDelivery = Field(default_factory=HeartbeatDelivery)
    # Markdown, like OpenClaw's HEARTBEAT.md.
    checklist: str = Field(default="", max_length=MAX_CHECKLIST_CHARS)

    @field_validator("active_hours_start", "active_hours_end")
    @classmethod
    def _hhmm(cls, value: str) -> str:
        if not _HHMM.match(value):
            raise ValueError("must be HH:MM in 24-hour time")
        return value

    @field_validator("timezone")
    @classmethod
    def _iana(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ValueError(f"unknown timezone {value!r}") from e
        return value


async def load_config(user_id: str) -> HeartbeatConfig:
    """The user's settings; the defaults (off) when none are saved or the
    stored object no longer validates."""
    stored = await user_db().get_user_copilot_heartbeat(user_id)
    if not stored:
        return HeartbeatConfig()
    try:
        return HeartbeatConfig.model_validate(stored)
    except ValidationError:
        logger.warning(
            "Heartbeat settings for user %s no longer validate; treating as off",
            user_id[:12],
            exc_info=True,
        )
        return HeartbeatConfig()


async def save_config(user_id: str, config: HeartbeatConfig) -> None:
    # Only the REST server writes, and it holds a Prisma connection.
    from backend.data.user import set_user_copilot_heartbeat

    await set_user_copilot_heartbeat(user_id, config.model_dump(mode="json"))


async def resolve_timezone(user_id: str, config: HeartbeatConfig) -> str:
    """The config's own timezone, else the profile's, else UTC."""
    if config.timezone:
        return config.timezone
    try:
        user = await user_db().get_user_by_id(user_id)
    except Exception:
        logger.warning(
            "Could not read the timezone of user %s; heartbeat uses UTC",
            user_id[:12],
            exc_info=True,
        )
        return "UTC"
    return get_user_timezone_or_utc(getattr(user, "timezone", None))


def in_active_hours(config: HeartbeatConfig, now_local: datetime) -> bool:
    """Whether ``now_local`` (already in the user's zone) is inside the window.

    The start is inclusive and the end exclusive, so 08:00-22:00 runs a beat
    at 08:00 and none at 22:00.
    """
    start = _parse_hhmm(config.active_hours_start)
    end = _parse_hhmm(config.active_hours_end)
    current = now_local.time().replace(second=0, microsecond=0)
    if start == end:
        return True
    if start < end:
        return start <= current < end
    return current >= start or current < end


# Lines that carry no instruction: headings, comments, blank list items and
# unticked empty checkboxes, as a fresh HEARTBEAT.md template has.
_EMPTY_LINE = re.compile(r"^(#+\s.*|#+|[-*+]\s*(\[[ xX]?\])?\s*|>\s*)$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def checklist_is_empty(checklist: str) -> bool:
    """True when nothing in the checklist asks for anything, so a beat would
    only ever answer NO_REPLY and is not worth a model call."""
    text = _HTML_COMMENT.sub("", checklist or "")
    for raw in text.splitlines():
        line = raw.strip()
        if line and not _EMPTY_LINE.match(line):
            return False
    return True


def _parse_hhmm(value: str) -> time:
    hours, minutes = value.split(":")
    return time(int(hours), int(minutes))
