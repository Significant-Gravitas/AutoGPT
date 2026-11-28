"""Credits each running credit-billed turn holds, per ledger, so turns that
run at once share a balance instead of each being offered all of it. A turn
holds a little to start and more as it spends (``credit_billing``)."""

import time
from collections.abc import Awaitable, Callable

from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import as_str, hash_compare_and_set

_VERSION = "version"
_HOLD_ATTEMPTS = 8


class LedgerHolds:
    """One Redis hash per ledger: a field per running turn, holding the
    credits it may still debit and when its hold lapses, plus a version that
    every hold moves, so two holds never both count the same credits."""

    def __init__(self, ledger_id: str, lifetime_seconds: int) -> None:
        self.key = f"copilot:credit_turn_holds:{{{ledger_id}}}"
        self.lifetime_seconds = lifetime_seconds

    async def hold(
        self,
        turn_id: str,
        *,
        read_balance: Callable[[], Awaitable[int]],
        credits: int,
        at_least: int = 1,
    ) -> int:
        """Add up to *credits* to the turn's hold, from what the balance has
        left after every live hold, its own included; 0, adding nothing, when
        that is under *at_least*."""
        redis = await get_redis_async()
        # The compare-and-set needs a version to compare against; the key
        # lapses even when no hold on it ever succeeds.
        pipe = redis.pipeline(transaction=True)
        pipe.hsetnx(self.key, _VERSION, "0")
        pipe.expire(self.key, self.lifetime_seconds)
        await pipe.execute()
        for _ in range(_HOLD_ATTEMPTS):
            fields = await redis.hgetall(self.key)
            # After the holds: a debit lands before its hold drops, so this
            # order can only under-count what is free.
            balance = await read_balance()
            version = as_str(fields.pop(_VERSION, None)) or "0"
            now = time.time()
            own = _live(fields.pop(turn_id, None), now)
            held = own + sum(_live(raw, now) for raw in fields.values())
            added = min(balance - held, credits)
            if added < at_least:
                return 0
            if await hash_compare_and_set(
                redis,
                self.key,
                _VERSION,
                expected=version,
                new=str(int(version) + 1),
                updates={turn_id: _format(own + added, now + self.lifetime_seconds)},
            ):
                await redis.expire(self.key, self.lifetime_seconds)
                return added
        raise RuntimeError("Credit hold could not be taken")

    async def lower(self, turn_id: str, credits: int) -> None:
        """What the turn may still debit, after a debit has landed."""
        redis = await get_redis_async()
        await redis.hset(
            self.key,
            turn_id,
            _format(max(0, credits), time.time() + self.lifetime_seconds),
        )

    async def release(self, turn_id: str) -> None:
        redis = await get_redis_async()
        await redis.hdel(self.key, turn_id)


def _live(raw: bytes | str | None, now: float) -> int:
    """The credits a field holds, or 0 once it has lapsed."""
    credits, _, expires_at = (as_str(raw) or "").partition(":")
    return int(credits) if credits and float(expires_at) > now else 0


def _format(credits: int, expires_at: float) -> str:
    return f"{credits}:{expires_at:.0f}"
