"""How many credentialed requests a box, and a user, may make.

The swap stops a credential leaving the box; it does not stop a box using it.
A compromised session could walk an entire account through the provider's
API, so every request that gets a value swapped in is counted: per box (its
owner, a session or an expert) and per user, in a fixed window.  Past either
limit the request is not sent, and the box gets an answer it can read instead
of a failure it would retry.

Counted in Redis, which every replica shares, and failing closed: if the
count cannot be taken, the request goes out with no value in it.  Only
credentialed requests count; everything else a box does is untouched.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Optional, Protocol

logger = logging.getLogger(__name__)

KEY_PREFIX = "swap:quota:"


class RedisCounter(Protocol):
    def incr(self, name: str) -> Awaitable[Any]: ...

    def expire(self, name: str, time: int) -> Awaitable[Any]: ...


@dataclass(frozen=True)
class QuotaVerdict:
    """*reason* is empty when the request may go; otherwise ``quota-exceeded``
    (with the *scope* whose *limit* it passed) or ``quota-unavailable``."""

    reason: str = ""
    scope: str = ""
    limit: int = 0
    window: int = 0
    retry_after: int = 0

    @property
    def allowed(self) -> bool:
        return not self.reason

    def message(self) -> str:
        """What the box reads instead of the provider's answer."""
        if self.reason == "quota-unavailable":
            return (
                "swap-proxy: request not sent. The request quota could not be "
                "checked, so no credential was attached. Try again shortly.\n"
            )
        who = "this sandbox" if self.scope == "box" else "this user's sandboxes"
        return (
            f"swap-proxy: request not sent. {who.capitalize()} made "
            f"{self.limit} requests with the user's connected accounts in "
            f"{self.window} s, the limit. It resets in {self.retry_after} s. "
            "Do not retry in a loop: wait, or tell the user the task needs "
            "more requests than it is allowed.\n"
        )


class RequestQuota:
    """*per_box* and *per_user* requests per *window* seconds; 0 turns one
    off."""

    def __init__(
        self, redis: RedisCounter, *, per_box: int, per_user: int, window: int
    ):
        self._redis = redis
        self._per_box = per_box
        self._per_user = per_user
        self._window = max(window, 1)

    async def take(
        self, owner_label: str, user_id: Optional[str], *, now: Optional[float] = None
    ) -> QuotaVerdict:
        """Count one credentialed request for this box and user."""
        now = time.time() if now is None else now
        index = int(now) // self._window
        retry_after = self._window - int(now) % self._window
        scopes = [("box", owner_label, self._per_box)]
        if user_id:
            scopes.append(("user", user_id, self._per_user))
        for scope, ident, limit in scopes:
            if limit <= 0:
                continue
            key = f"{KEY_PREFIX}{scope}:{ident}:{index}"
            try:
                count = int(await self._redis.incr(key))
                if count == 1:
                    await self._redis.expire(key, self._window)
            except Exception:
                logger.warning("Quota count failed; refusing the swap", exc_info=True)
                return QuotaVerdict(reason="quota-unavailable")
            if count > limit:
                return QuotaVerdict(
                    reason="quota-exceeded",
                    scope=scope,
                    limit=limit,
                    window=self._window,
                    retry_after=retry_after,
                )
        return QuotaVerdict()
