"""Server-side per-session consent for Local PC computer use."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Literal

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

ComputerUseConsent = Literal["pending", "approved", "denied"]

_CONSENT_KEY_PREFIX = "copilot:local-executor:computer-use-consent:"
_CONSENT_TTL_SECONDS = 60 * 60 * 24 * 30


async def get_computer_use_consent(
    session_id: str,
    user_id: str | None,
    *,
    machine_id: str | None = None,
    driver: str | None = None,
    tools: Iterable[str] | None = None,
) -> ComputerUseConsent:
    """Return consent only when the approval matches this machine, its
    computer-use driver release, and the tools it offers."""
    if not user_id:
        return "pending"
    try:
        value = await (await get_redis_async()).get(_consent_key(session_id, user_id))
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        record = json.loads(value) if isinstance(value, str) else None
    except Exception:
        logger.exception(
            "Failed to read Local PC computer-use consent for session %s",
            session_id[:12],
        )
        return "pending"

    if not isinstance(record, dict):
        return "pending"
    state = record.get("state")
    if state == "denied":
        return "denied"
    if state != "approved":
        return "pending"

    scoped_machine_id, scoped_driver, scoped_tools = _normalize_scope(
        machine_id, driver, tools
    )
    if (
        not scoped_machine_id
        or record.get("machine_id") != scoped_machine_id
        or record.get("driver") != scoped_driver
        or record.get("tools") != scoped_tools
    ):
        return "pending"
    return "approved"


async def is_computer_use_approved(
    session_id: str,
    user_id: str | None,
    *,
    machine_id: str | None = None,
    driver: str | None = None,
    tools: Iterable[str] | None = None,
) -> bool:
    """Return whether this session may expose Local PC computer-use tools."""
    return (
        await get_computer_use_consent(
            session_id, user_id, machine_id=machine_id, driver=driver, tools=tools
        )
        == "approved"
    )


async def set_computer_use_consent(
    session_id: str,
    user_id: str,
    *,
    approved: bool,
    machine_id: str | None = None,
    driver: str | None = None,
    tools: Iterable[str] | None = None,
) -> ComputerUseConsent:
    """Persist a decision bound to the connected machine's advertised scope."""
    state: ComputerUseConsent = "approved" if approved else "denied"
    scoped_machine_id, scoped_driver, scoped_tools = _normalize_scope(
        machine_id, driver, tools
    )
    if approved and not (scoped_machine_id and scoped_driver and scoped_tools):
        raise ValueError(
            "Computer-use approval requires a connected machine and its tools"
        )
    record = json.dumps(
        {
            "state": state,
            "machine_id": scoped_machine_id or None,
            "driver": scoped_driver,
            "tools": scoped_tools,
        },
        separators=(",", ":"),
    )
    redis = await get_redis_async()
    await redis.setex(_consent_key(session_id, user_id), _CONSENT_TTL_SECONDS, record)
    return state


def _consent_key(session_id: str, user_id: str) -> str:
    return f"{_CONSENT_KEY_PREFIX}{user_id}:{session_id}"


def _normalize_scope(
    machine_id: str | None,
    driver: str | None,
    tools: Iterable[str] | None,
) -> tuple[str, str, list[str]]:
    return machine_id or "", driver or "", normalize_tool_names(tools)


def normalize_tool_names(tools: Iterable[str] | None) -> list[str]:
    return sorted({tool for tool in tools or () if isinstance(tool, str) and tool})
