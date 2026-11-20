"""Invalidations this process couldn't record in Redis.

A versioned cache entry is trusted while the token it was written under is
current. When an invalidation fails to replace the token, a write already on
its way when the change landed is stored under the old, unchanged token, so
until any such entry has expired this process must read the key as if Redis
were out. These marks say which keys, and until when.
"""

UNCONFIRMED_MAXSIZE = 2048


class UnconfirmedInvalidations:
    """Live marks, never dropped: with no room left, every key is covered
    until the latest mark would have expired.

    Callers pass the time (monotonic seconds), so the clock stays theirs.
    """

    def __init__(self, *, lifetime: float, maxsize: int = UNCONFIRMED_MAXSIZE):
        self.lifetime = lifetime
        self.maxsize = maxsize
        self.until: dict[str, float] = {}
        self.everyone_until = 0.0

    def add(self, key: str, now: float) -> None:
        until = now + self.lifetime
        self._purge(now)
        # Re-inserted at the end: every mark has the same lifetime, so
        # insertion order stays expiry order.
        self.until.pop(key, None)
        if len(self.until) >= self.maxsize:
            self.everyone_until = max(self.everyone_until, until)
            return
        self.until[key] = until

    def clear(self, key: str) -> None:
        self.until.pop(key, None)

    def covers(self, key: str, now: float) -> bool:
        if now < self.everyone_until:
            return True
        until = self.until.get(key)
        return until is not None and until > now

    def _purge(self, now: float) -> None:
        for key, until in list(self.until.items()):
            if until > now:
                break
            del self.until[key]
