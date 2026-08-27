"""Reading what an xAI refusal actually means.

This is a classifier rather than a lookup because xAI's error envelope is
flat and its ``code`` field carries four different kinds of thing::

    {"code": <string|number>, "error": "<message>"}

``code`` may hold a kebab gRPC code (``permission-denied``), a namespaced
well-known code (``subscription:free-usage-exhausted``), a bare integer, or
*a sentence*. xAI's own parser has a test named
``sentence_shaped_codes_never_become_prefixes``, which is the tell: their
client cannot trust the field either. And when a machine-readable code is
not in ``code``, it is appended to the message inline as ``[WKE=ns:code]``.

So classifying on ``code`` alone is wrong in both directions -- it misses
codes that arrived in the message, and it treats prose as an identifier.

Two status rules that look backwards and are not:

- **402 is always a spending block**, with no message filter.
- **403 is a spending block only if the body says "run out of credits"**,
  and is *never* an authentication failure. That second half matters: xAI's
  own client refuses to re-auth on a 403, because doing so can race their
  ``invalid_grant_threshold`` and destroy a perfectly good stored
  credential. A 403 here must not trigger a token refresh.

There is also no rate-limit header to read: only ``Retry-After``, which is
clamped because the value can be far longer than a user will sit through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

# xAI's client clamps this. A provider is free to ask for an hour; a chat
# waiting on it is just a hang, and telling the user to come back is better
# than pretending to retry.
MAX_RETRY_AFTER_SECONDS = 120

# Emitted inline in the message when the machine code is not in ``code``.
_INLINE_CODE = re.compile(r"\[WKE=([a-z0-9_.-]+:[a-z0-9_.-]+)\]", re.IGNORECASE)

_FREE_USAGE_EXHAUSTED = "subscription:free-usage-exhausted"
_SPENDING_LIMIT = "personal-team-blocked:spending-limit"
_OUT_OF_CREDITS = "run out of credits"


class GrokFailure(str, Enum):
    """What to do about it, which is the only distinction worth drawing.

    Deliberately coarser than xAI's code space: the point is to pick the
    sentence a user reads and whether anything should be retried, not to
    mirror a vocabulary that is still changing.
    """

    FREE_QUOTA_EXHAUSTED = "free_quota_exhausted"
    SPENDING_BLOCKED = "spending_blocked"
    RATE_LIMITED = "rate_limited"
    AUTH_EXPIRED = "auth_expired"
    CLIENT_REJECTED = "client_rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GrokError:
    failure: GrokFailure
    message: str
    status_code: int | None = None
    # The machine code, from wherever it was actually carried.
    code: str | None = None
    retry_after_seconds: int | None = None

    @property
    def should_refresh_credentials(self) -> bool:
        """Whether re-authenticating could plausibly help.

        False for a 403 even though it reads like an authorization problem:
        refreshing there can race xAI's failed-grant counter and wipe a
        working credential, turning a temporary refusal into a broken
        connection.
        """
        return self.failure is GrokFailure.AUTH_EXPIRED

    @property
    def is_retryable(self) -> bool:
        return self.failure is GrokFailure.RATE_LIMITED


def classify(
    status_code: int | None,
    body: Any,
    retry_after: str | int | None = None,
) -> GrokError:
    """Turn a refusal into something with a next step attached."""
    payload = body if isinstance(body, dict) else {}
    message = str(payload.get("error") or "").strip()
    code = _machine_code(payload, message)
    retry_seconds = _retry_after_seconds(retry_after)

    if code == _FREE_USAGE_EXHAUSTED:
        return GrokError(
            GrokFailure.FREE_QUOTA_EXHAUSTED,
            message or "This xAI account has used its free allowance for now.",
            status_code,
            code,
            retry_seconds,
        )

    if code == _SPENDING_LIMIT or status_code == 402:
        # 402 needs no message filter -- it is only ever a spend block.
        return GrokError(
            GrokFailure.SPENDING_BLOCKED,
            message or "This xAI account has hit its spending limit.",
            status_code,
            code,
            retry_seconds,
        )

    if status_code == 403:
        if _OUT_OF_CREDITS in message.lower():
            return GrokError(
                GrokFailure.SPENDING_BLOCKED, message, status_code, code, retry_seconds
            )
        # Not an auth failure, however much it reads like one. Re-authing
        # here can destroy the stored credential.
        return GrokError(
            GrokFailure.CLIENT_REJECTED,
            message or "xAI refused this request.",
            status_code,
            code,
            retry_seconds,
        )

    if status_code == 429:
        return GrokError(
            GrokFailure.RATE_LIMITED,
            message or "xAI is rate limiting this account.",
            status_code,
            code,
            retry_seconds,
        )

    if status_code == 401:
        return GrokError(
            GrokFailure.AUTH_EXPIRED,
            message or "This xAI connection needs to be signed in again.",
            status_code,
            code,
            retry_seconds,
        )

    if status_code == 426:
        # The client-version gate. Nothing a user can do, and retrying is
        # pointless -- it needs a build that matches the published floor.
        return GrokError(
            GrokFailure.CLIENT_REJECTED,
            message or "xAI requires a newer client than this build sends.",
            status_code,
            code,
            retry_seconds,
        )

    return GrokError(
        GrokFailure.UNKNOWN,
        message or "xAI returned an error.",
        status_code,
        code,
        retry_seconds,
    )


def _machine_code(payload: dict[str, Any], message: str) -> str | None:
    """The machine-readable code, from either place it can hide.

    ``code`` is preferred but only when it actually looks like a code: xAI
    puts sentences there too, and treating one as an identifier is how a
    classifier starts matching on prose.
    """
    inline = _INLINE_CODE.search(message)
    if inline:
        return inline.group(1).lower()

    raw = payload.get("code")
    if isinstance(raw, str) and _looks_like_a_code(raw):
        return raw.strip().lower()
    return None


def _looks_like_a_code(value: str) -> bool:
    candidate = value.strip()
    if not candidate or " " in candidate:
        # A space means prose. Every real code here is kebab or
        # namespace:kebab.
        return False
    return bool(re.fullmatch(r"[a-z0-9_.:-]+", candidate, re.IGNORECASE))


def _retry_after_seconds(retry_after: str | int | None) -> int | None:
    if retry_after is None:
        return None
    try:
        seconds = int(retry_after)
    except (TypeError, ValueError):
        # An HTTP-date is legal here. Resolving it needs a clock the caller
        # owns, and guessing is worse than saying nothing.
        return None
    if seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)
