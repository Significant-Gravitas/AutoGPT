"""
V2 External API - Idempotent run creation

Starting a run costs the caller money, and a timed-out request says nothing
about whether it started. Without a key the only safe retry is no retry.

The caller sends `Idempotency-Key: <a value they choose>`. The first request to
claim it runs; a later request with the same key gets back the run the first one
created. A request that arrives while the first is still in flight is a 409 —
there is no run to hand back yet.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Header, HTTPException
from starlette import status

from backend.data import execution as execution_db
from backend.data.redis_client import get_redis_async

from .models import AgentGraphRun
from .tenancy import TenantContext, in_tenant

logger = logging.getLogger(__name__)

IDEMPOTENCY_HEADER = "Idempotency-Key"

# Long enough to cover any sane retry window, short enough that a key is not a
# permanent reservation on a name the caller picked.
_TTL_SECONDS = 24 * 60 * 60
_IN_FLIGHT = "in-flight"


def idempotency_key(
    key: Optional[str] = Header(
        default=None,
        alias=IDEMPOTENCY_HEADER,
        max_length=255,
        description=(
            "Retry-safety token. Repeating a request with the same value returns "
            "the run the first one started instead of starting another. Scoped to "
            "the caller; expires after 24 hours."
        ),
    ),
) -> Optional[str]:
    return (key or "").strip() or None


@asynccontextmanager
async def idempotent_run(key: Optional[str], user_id: str) -> AsyncIterator["RunClaim"]:
    """Claim `key` for this request, or report the run it already produced.

    An unreachable key store degrades to no idempotency rather than to a refusal:
    the caller loses retry safety, which is where a caller without a key already is.
    """
    claim = RunClaim(key=key, user_id=user_id)
    if key is None:
        yield claim
        return

    try:
        redis = await get_redis_async()
        claimed = await redis.set(claim.redis_key, _IN_FLIGHT, nx=True, ex=_TTL_SECONDS)
    except Exception as e:
        logger.warning(f"Idempotency store unavailable, proceeding without it: {e}")
        yield claim
        return

    if not claimed:
        claim.existing_run_id = await claim.resolve_existing()
        yield claim
        return

    claim.holds_key = True
    try:
        yield claim
    except Exception:
        # No run came of it, so the key must not stay claimed against a retry.
        await claim.release()
        raise


async def replayed_run(claim: "RunClaim", auth: TenantContext) -> AgentGraphRun:
    """The run a previous request with this key started."""
    return AgentGraphRun.from_internal(
        in_tenant(
            await execution_db.get_graph_execution(
                user_id=auth.user_id,
                execution_id=claim.existing_run_id or "",
                organization_id=auth.organization_id,
            ),
            auth,
            f"Run #{claim.existing_run_id}",
        )
    )


class RunClaim:
    """One request's hold on an idempotency key."""

    def __init__(self, key: Optional[str], user_id: str) -> None:
        self.key = key
        self.user_id = user_id
        self.holds_key = False
        self.existing_run_id: Optional[str] = None

    @property
    def redis_key(self) -> str:
        return f"v2:idem:{self.user_id}:{self.key}"

    async def resolve_existing(self) -> str:
        """The run id the first request recorded, or 409 while it is still running."""
        try:
            redis = await get_redis_async()
            stored = await redis.get(self.redis_key)
        except Exception as e:
            logger.warning(f"Idempotency store unavailable on replay: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cannot tell whether this request already ran; retry shortly.",
            )
        if stored in (None, _IN_FLIGHT):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A request with {IDEMPOTENCY_HEADER} '{self.key}' is still "
                "in flight. Retry once it completes.",
            )
        return stored

    async def record(self, run_id: str) -> None:
        """Point the key at the run, so a later retry gets this one back."""
        if not self.holds_key:
            return
        try:
            redis = await get_redis_async()
            await redis.set(self.redis_key, run_id, ex=_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Could not record idempotent run {run_id}: {e}")

    async def release(self) -> None:
        if not self.holds_key:
            return
        try:
            redis = await get_redis_async()
            await redis.delete(self.redis_key)
        except Exception as e:
            logger.warning(f"Could not release idempotency key: {e}")
