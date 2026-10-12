"""Best-effort steps of closing an account: bounded, logged, never raised."""

import asyncio
import logging
from collections.abc import Awaitable
from typing import TypeVar

from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# The most any one call may take. These run inside the deletion request, and
# the scheduler RPC client would otherwise wait up to its 300 s default.
STEP_TIMEOUT_SECONDS = 10.0


async def attempt_step(what: str, user_id: str, step: Awaitable[T]) -> T | None:
    """Await one call, bounded by STEP_TIMEOUT_SECONDS; log it if it fails."""
    try:
        return await asyncio.wait_for(step, timeout=STEP_TIMEOUT_SECONDS)
    except NotFoundError:
        logger.info(
            f"Account deletion did not need to {what} for user {user_id}: already gone"
        )
    except Exception as e:
        log_step_failure(what, user_id, e)
    return None


def log_step_failure(what: str, user_id: str, error: Exception) -> None:
    logger.error(
        f"Account deletion could not {what} for user {user_id}: "
        f"{type(error).__name__}: {error}"
    )
