"""One erasure of an account at a time, across every process.

The claim (`request.claim_request`) lets a run resume an erasure that
stopped, so it does not keep two live runs apart: the hourly job on two
database manager replicas, or the CLI beside the job. Both would read the
account once and act on it twice (the confirmation email sent twice, say).
A Redis lease per account, held from before the account is read until the
erasure ends, keeps them apart. Its lifetime is short (`LEASE_TTL_SECONDS`),
so a run that crashed or was killed frees it within minutes and the next run
can resume. A live run keeps it: a heartbeat renews it in the background
(`renew_in_background`), since one step (the files sweep of a large account)
can outlast the lifetime, and the erasure renews it again before each step
(`keep_erasure_lease`) and stops if it was lost.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator

from redis.exceptions import RedisClusterException, RedisError

from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock

LEASE_KEY_PREFIX = "account_erasure:lease:"
LEASE_TTL_SECONDS = 5 * 60
# Every minute: five renewals per lifetime, so a beat or two lost to a Redis
# blip doesn't drop the lease, and above the lock's own rate limit (a tenth of
# the lifetime), under which a renewal only checks the owner.
HEARTBEAT_SECONDS = 60

logger = logging.getLogger(__name__)


class ErasureLeaseUnavailable(RuntimeError):
    """Redis can't answer, so whether another run holds the lease is unknown,
    and nothing is erased."""


async def acquire_erasure_lease(user_id: str) -> AsyncClusterLock | None:
    """The lease, or None while another run holds it. Raises
    `ErasureLeaseUnavailable` when Redis can't answer: erasing without
    knowing is what the lease prevents."""
    try:
        lock = AsyncClusterLock(
            redis=await get_redis_async(),
            key=f"{LEASE_KEY_PREFIX}{user_id}",
            owner_id=uuid.uuid4().hex,
            timeout=LEASE_TTL_SECONDS,
        )
    except (RedisError, RedisClusterException, OSError) as e:
        # A client not connected yet connects here, and raises if it can't:
        # the cluster client raises RedisClusterException, not a RedisError.
        raise ErasureLeaseUnavailable("account erasure lease unavailable") from e
    owner = await lock.try_acquire()
    if owner is None:
        raise ErasureLeaseUnavailable("account erasure lease unavailable")
    if owner != lock.owner_id:
        return None
    return lock


async def keep_erasure_lease(lock: AsyncClusterLock) -> bool:
    """Renew the lease; False when it lapsed or another run took it over, and
    the erasure must stop."""
    return await lock.refresh()


@contextlib.asynccontextmanager
async def renew_in_background(lock: AsyncClusterLock) -> AsyncIterator[None]:
    """Renew the lease every `HEARTBEAT_SECONDS` while the block runs. A
    renewal that fails is logged and the next beat tries again; the check
    before each step is what stops an erasure whose lease is gone."""
    task = asyncio.create_task(_beat(lock))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _beat(lock: AsyncClusterLock) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        try:
            renewed = await lock.refresh()
        except Exception as e:
            # Never ends the heartbeat, nor reaches the erasure's result.
            logger.warning(f"Account erasure lease renewal raised {type(e).__name__}")
            renewed = False
        if not renewed:
            logger.warning(
                f"Account erasure lease {lock.key} could not be renewed; the "
                "erasure stops before its next step if it was lost"
            )


async def release_erasure_lease(lock: AsyncClusterLock) -> None:
    """Owner-checked: a lease another run took over is left to it. A release
    that fails is swallowed by the lock, and the TTL clears the key."""
    await lock.release()
