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
    features_coarse: Iterable[str] | None = None,
    features: Iterable[str] | None = None,
) -> ComputerUseConsent:
    """Return consent only when approval matches this machine and feature set."""
    if not user_id:
        return "pending"
    try:
        value = await (await get_redis_async()).get(_consent_key(session_id, user_id))
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if value == "denied":
            record = {"state": "denied"}
        elif value == "approved":
            record = None
        else:
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

    scoped_machine_id, scoped_features_coarse, scoped_features = _normalize_scope(
        machine_id, features_coarse, features
    )
    stored_features_coarse = record.get("features_coarse")
    stored_features = record.get("features")
    if (
        not scoped_machine_id
        or record.get("machine_id") != scoped_machine_id
        or not isinstance(stored_features_coarse, list)
        or stored_features_coarse != scoped_features_coarse
        or not isinstance(stored_features, list)
        or stored_features != scoped_features
    ):
        return "pending"
    return "approved"


async def is_computer_use_approved(
    session_id: str,
    user_id: str | None,
    *,
    machine_id: str | None = None,
    features_coarse: Iterable[str] | None = None,
    features: Iterable[str] | None = None,
) -> bool:
    """Return whether this session may expose Local PC computer-use tools."""
    return (
        await get_computer_use_consent(
            session_id,
            user_id,
            machine_id=machine_id,
            features_coarse=features_coarse,
            features=features,
        )
        == "approved"
    )


async def set_computer_use_consent(
    session_id: str,
    user_id: str,
    *,
    approved: bool,
    machine_id: str | None = None,
    features_coarse: Iterable[str] | None = None,
    features: Iterable[str] | None = None,
) -> ComputerUseConsent:
    """Persist a decision bound to the connected machine's advertised scope."""
    state: ComputerUseConsent = "approved" if approved else "denied"
    scoped_machine_id, scoped_features_coarse, scoped_features = _normalize_scope(
        machine_id, features_coarse, features
    )
    if approved and not scoped_machine_id:
        raise ValueError("Computer-use approval requires a connected machine")
    record = json.dumps(
        {
            "state": state,
            "machine_id": scoped_machine_id or None,
            "features_coarse": scoped_features_coarse,
            "features": scoped_features,
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
    features_coarse: Iterable[str] | None,
    features: Iterable[str] | None,
) -> tuple[str, list[str], list[str]]:
    return (
        machine_id or "",
        _normalize_features(features_coarse),
        _normalize_features(features),
    )


def _normalize_features(features: Iterable[str] | None) -> list[str]:
    return sorted(
        {feature for feature in features or () if isinstance(feature, str) and feature}
    )
