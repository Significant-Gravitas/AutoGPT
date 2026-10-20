"""Calendar usage namespaces published atomically with paid entitlement."""

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from backend.data.db_accessors import pro_activation_db
from backend.data.redis_client import get_redis_async

if TYPE_CHECKING:
    from backend.data.pro_activation import UsageActivationState


class UsageActivationUnavailable(RuntimeError):
    """No authoritative, completed entitlement/usage snapshot is available."""


async def get_ready_usage_state(user_id: str) -> "UsageActivationState":
    try:
        state = await pro_activation_db().get_usage_activation_state(user_id)
    except Exception as exc:
        raise UsageActivationUnavailable("Usage activation state unavailable") from exc
    if state.user_id != user_id or not state.ready:
        raise UsageActivationUnavailable("Usage activation is processing")
    return state


def usage_keys(user_id: str, generation: str | None, now: datetime) -> tuple[str, str]:
    owner = f"{user_id}:{generation}" if generation else user_id
    year, week, _ = now.isocalendar()
    return (
        f"copilot:cost:daily:{owner}:{now.strftime('%Y-%m-%d')}",
        f"copilot:cost:weekly:{owner}:{year}-W{week:02d}",
    )


async def verify_activation_usage(user_id: str, generation: str) -> None:
    """Retry-safe readiness probe; never erase or initialize paid counters."""
    redis = await get_redis_async()
    keys = usage_keys(user_id, generation, datetime.now(UTC))
    counters = await asyncio.gather(*(redis.get(key) for key in keys))
    if any(int(value or 0) < 0 for value in counters):
        raise UsageActivationUnavailable("Invalid paid usage counters")
