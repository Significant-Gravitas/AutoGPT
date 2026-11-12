"""One erasure of an account at a time, across every process.

The claim (`request.claim_request`) lets a run resume an erasure that
stopped, so it does not keep two live runs apart: the hourly job on two
database manager replicas, or the CLI beside the job. Both would read the
account once and act on it twice (the confirmation email sent twice, say).
A Redis lease per account, held from before the account is read until the
erasure ends, keeps them apart. The erasure renews it before each step
(`keep_erasure_lease`) and stops if it was lost; after a crash it expires on
its own (`LEASE_TTL_SECONDS`), so the next run can resume.
"""

import uuid

from redis.exceptions import RedisClusterException, RedisError

from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock

LEASE_KEY_PREFIX = "account_erasure:lease:"
LEASE_TTL_SECONDS = 60 * 60


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


async def release_erasure_lease(lock: AsyncClusterLock) -> None:
    """Owner-checked: a lease another run took over is left to it. A release
    that fails is swallowed by the lock, and the TTL clears the key."""
    await lock.release()
