"""Turn an unreachable or slow scheduler into a fast, stable 503.

The scheduler is a singleton that restarts on every deploy. While it is down,
schedule routes used to answer 500 "All connection attempts failed", or wait
out the 300s RPC timeout until the proxy gave up with a 504. Routes wrap their
scheduler calls in :func:`read_schedules` or :func:`write_schedule` so the user
gets a 503 with a code the frontend can show calmly and retry.
"""

import asyncio
import logging
from typing import Awaitable, Literal, TypeVar

import httpx
from fastapi import HTTPException
from pydantic import BaseModel

from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

T = TypeVar("T")

RETRY_AFTER_SECONDS = 15

SchedulerErrorCode = Literal["scheduler_unavailable", "scheduler_unconfirmed"]

# The connection was never made, so the scheduler never saw the request.
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class SchedulerUnavailableDetail(BaseModel):
    """Body ``detail`` of a scheduler 503.

    ``scheduler_unavailable``: nothing was changed; retrying is safe.
    ``scheduler_unconfirmed``: a write reached the scheduler but no answer came
    back in time, so it may or may not have been applied; re-read before retrying.
    """

    code: SchedulerErrorCode
    message: str


async def read_schedules(call: Awaitable[T], timeout: float | None = None) -> T:
    """Await a scheduler read, or raise a 503 within ``timeout`` seconds."""
    bound = timeout or settings.config.scheduler_request_timeout_seconds
    try:
        return await asyncio.wait_for(call, timeout=bound)
    except (asyncio.TimeoutError, httpx.TransportError) as e:
        raise _scheduler_503(
            "scheduler_unavailable",
            "Schedules are temporarily unavailable. Try again in a moment.",
            e,
        ) from e


async def write_schedule(call: Awaitable[T]) -> T:
    """Await a scheduler write made through the interactive scheduler client.

    A scheduler that can't be reached fails within the client's connect
    timeout, and nothing was sent. Once it has the request, the write is not
    cut short: a deadline here would turn slow-but-successful writes (a cold
    scheduler takes 25s+ to create one) into failures, and on a self-hosted
    install nothing else would. Behind the hosted proxy a write that outlives
    it ends as a 504, which the frontend treats like ``scheduler_unconfirmed``.
    """
    try:
        return await call
    except _NOT_SENT as e:
        raise _scheduler_503(
            "scheduler_unavailable",
            "Schedules are temporarily unavailable, so nothing was changed. "
            "Try again in a moment.",
            e,
        ) from e
    except httpx.TransportError as e:
        raise _scheduler_503(
            "scheduler_unconfirmed",
            "The scheduler didn't confirm the change in time. Refresh to see "
            "whether it was saved before trying again.",
            e,
        ) from e


def _scheduler_503(
    code: SchedulerErrorCode, message: str, cause: BaseException
) -> HTTPException:
    logger.warning(f"Scheduler call failed ({code}): {type(cause).__name__}: {cause}")
    return HTTPException(
        status_code=503,
        detail=SchedulerUnavailableDetail(code=code, message=message).model_dump(),
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )
