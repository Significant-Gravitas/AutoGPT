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


# One round trip, atomic, over every scope at once: each counter is checked
# first, and only if all are under their limit are all incremented (a new key
# gets its expiry in the same step).  A request refused on one scope spends
# nothing on another.  Returns 0, or the 1-based index of the scope that
# refused.  KEYS[i] pairs with ARGV[i + 1] (its limit); ARGV[1] is the window.
_TAKE_SCRIPT = """
for i, key in ipairs(KEYS) do
  local count = tonumber(redis.call('GET', key) or '0')
  if count >= tonumber(ARGV[i + 1]) then return i end
end
for _, key in ipairs(KEYS) do
  if redis.call('INCR', key) == 1 then redis.call('EXPIRE', key, ARGV[1]) end
end
return 0
"""


class RedisCounter(Protocol):
    def eval(
        self, script: str, numkeys: int, *keys_and_args: Any
    ) -> Awaitable[Any]: ...


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
    off.  Anything else that would weaken the quota without saying so (a
    negative limit, a window under a second) is refused here."""

    def __init__(
        self, redis: RedisCounter, *, per_box: int, per_user: int, window: int
    ):
        if per_box < 0 or per_user < 0:
            raise ValueError("Quota limits must be 0 (off) or more")
        if window <= 0:
            raise ValueError("The quota window must be at least a second")
        self._redis = redis
        self._per_box = per_box
        self._per_user = per_user
        self._window = window

    async def take(
        self, owner_label: str, user_id: Optional[str], *, now: Optional[float] = None
    ) -> QuotaVerdict:
        """Count one credentialed request for this box and user."""
        now = time.time() if now is None else now
        index = int(now) // self._window
        retry_after = self._window - int(now) % self._window
        # The user id is the hash tag of both keys: a script may only touch
        # keys of one cluster slot, and it is what both scopes share.
        tag = "{" + (user_id or owner_label) + "}"
        scopes = [("box", f"{tag}:box:{owner_label}", self._per_box)]
        if user_id:
            scopes.append(("user", f"{tag}:user", self._per_user))
        scopes = [scope for scope in scopes if scope[2] > 0]
        if not scopes:
            return QuotaVerdict()
        keys = [f"{KEY_PREFIX}{key}:{index}" for _, key, _ in scopes]
        limits = [limit for _, _, limit in scopes]
        try:
            refused = int(
                await self._redis.eval(
                    _TAKE_SCRIPT, len(keys), *keys, self._window, *limits
                )
            )
        except Exception:
            logger.warning("Quota count failed; refusing the swap", exc_info=True)
            return QuotaVerdict(reason="quota-unavailable")
        if refused:
            scope, _, limit = scopes[refused - 1]
            return QuotaVerdict(
                reason="quota-exceeded",
                scope=scope,
                limit=limit,
                window=self._window,
                retry_after=retry_after,
            )
        return QuotaVerdict()
