"""One erasure of an account at a time, across every process.

The claim (`request.claim_request`) lets a run resume an erasure that
stopped, so it does not keep two live runs apart: the hourly job on two
database manager replicas, or the CLI beside the job. Both would read the
account once and act on it twice (the confirmation email sent twice, say).
A Redis lease per account, held from before the account is read until the
erasure ends, keeps them apart. It outlives any erasure (`LEASE_TTL_SECONDS`)
and expires on its own after a crash, so the next run can resume.
"""

import logging
import uuid

from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock

logger = logging.getLogger(__name__)

LEASE_KEY_PREFIX = "account_erasure:lease:"
LEASE_TTL_SECONDS = 60 * 60


async def acquire_erasure_lease(user_id: str) -> AsyncClusterLock | None:
    """The lease, or None while another run holds it. Raises when Redis
    can't answer: erasing without knowing is what the lease prevents."""
    lock = AsyncClusterLock(
        redis=await get_redis_async(),
        key=f"{LEASE_KEY_PREFIX}{user_id}",
        owner_id=uuid.uuid4().hex,
        timeout=LEASE_TTL_SECONDS,
    )
    owner = await lock.try_acquire()
    if owner is None:
        raise RuntimeError("account erasure lease unavailable")
    if owner != lock.owner_id:
        return None
    return lock


async def release_erasure_lease(lock: AsyncClusterLock) -> None:
    try:
        await lock.release()
    except Exception:
        logger.warning("Account erasure lease release failed; its TTL clears it")
