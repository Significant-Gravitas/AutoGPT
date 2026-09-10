"""What went wrong with the provider behind a turn, said precisely.

Every provider failure currently reaches the client as one opaque
``baseline_error`` carrying a stringified exception. An expired ChatGPT
login, a monthly usage limit, a retired model and a policy refusal are all
the same event to the UI, so it can only offer the same useless advice --
"Press Try Again" -- to three failures where trying again cannot work.

This module names the difference once, at the boundary where the provider
exception is still intact, and hands the caller something a UI can act on:
whether retrying is pointless, whether reconnecting would help, and when
the limit actually lifts.

``resets_at`` is deliberately ``None`` unless the provider reported a real
timestamp. An invented "try again in an hour" that turns out to be wrong is
worse than saying nothing, because the user plans around it.
"""

import logging
import re
import time
from enum import Enum
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel, Field

from backend.copilot.rate_limit import UserPaywalledError
from backend.integrations.codex.transport import (
    CodexCredentialBusyError,
    CodexCredentialIntegrityError,
    CodexInvocationTimeoutError,
    CodexTransportOverloadedError,
)

logger = logging.getLogger(__name__)


class ProviderFailureKind(str, Enum):
    """Why a provider refused, in terms that imply what to do about it."""

    # The credential was understood and rejected -- reconnecting fixes it.
    AUTH_EXPIRED = "auth_expired"
    # The credential is malformed or unusable; reconnecting fixes it.
    INVALID_CREDENTIAL = "invalid_credential"
    # A quota is spent. Waiting fixes it; retrying now does not.
    USAGE_LIMIT = "usage_limit"
    # The model is gone or not available to this account.
    MODEL_UNAVAILABLE = "model_unavailable"
    # The provider refused on policy grounds. Retrying is pointless.
    POLICY_DENIED = "policy_denied"
    # The user's plan does not include this route.
    #
    # Shares its value with ``ExecutionFailureReason.ENTITLEMENT_REQUIRED``
    # on purpose: the same denial can terminate a graph run or a chat turn,
    # and two names for it would split every downstream count in half.
    ENTITLEMENT_REQUIRED = "entitlement_required"
    # Anything that might genuinely succeed on a second attempt.
    TRANSIENT = "transient"


# Retrying the same turn is only honest for a failure that could resolve
# itself. Everything else needs the user to do something first.
_RETRYABLE = frozenset({ProviderFailureKind.TRANSIENT})

# The kinds a reconnect would actually clear.
_RECONNECTABLE = frozenset(
    {
        ProviderFailureKind.AUTH_EXPIRED,
        ProviderFailureKind.INVALID_CREDENTIAL,
    }
)


class ProviderFailure(BaseModel):
    """One provider refusal, carried whole from the runtime to the client."""

    kind: ProviderFailureKind
    message: str = Field(description="What to tell the user, already humanized")
    auth_provider: str | None = Field(
        default=None, description="Which connection failed: platform or codex"
    )
    credential_id: str | None = None
    resets_at: int | None = Field(
        default=None,
        description=(
            "Unix seconds, only when the provider reported one. Never guessed."
        ),
    )

    @property
    def retryable(self) -> bool:
        return self.kind in _RETRYABLE

    @property
    def reconnect_fixes_it(self) -> bool:
        return self.kind in _RECONNECTABLE

    def as_part(self) -> dict[str, Any]:
        """The payload carried on the stream, including derived advice.

        ``retryable`` is computed here rather than left to the client so a
        second implementation of "is this worth retrying" cannot drift from
        this one.
        """
        return {
            "kind": self.kind.value,
            "message": self.message,
            "authProvider": self.auth_provider,
            "credentialId": self.credential_id,
            "resetsAt": self.resets_at,
            "retryable": self.retryable,
            "reconnectFixesIt": self.reconnect_fixes_it,
        }


# Codex failures arrive carrying an internal code as their message
# ("codex_credential_busy"), which is a log line, not something to show a
# person. These say the same thing in the user's terms, and say what to do.
_CODEX_MESSAGES: dict[type[BaseException], str] = {
    # Deliberately does not say "only one chat at a time". Measured against a
    # live ChatGPT connection, concurrent chats do NOT contend: three
    # simultaneous turns all completed in ~8s, and a short turn fired six
    # seconds into a 42s one answered in 4s, with no busy error in either
    # case. The five-second timeout is on the *credential* lease, which is
    # contended when something needs to write the credential -- a token
    # refresh -- not by ordinary turns sharing it.
    #
    # So this is rarer than "you have two chats open", and wording it that
    # way would send the user to close a chat that is not the problem.
    CodexCredentialBusyError: (
        "This ChatGPT connection is briefly unavailable. " "Try again in a moment."
    ),
    CodexCredentialIntegrityError: (
        "This ChatGPT connection can't be used. Reconnect the account in "
        "Settings, then send this again."
    ),
    CodexTransportOverloadedError: (
        "ChatGPT is busy right now. Try again in a moment."
    ),
    CodexInvocationTimeoutError: ("ChatGPT took too long to respond. Try again."),
}


def _humanize(exc: BaseException) -> str | None:
    """A user-facing sentence for a failure whose own message is a code."""
    return _CODEX_MESSAGES.get(type(exc))


def classify(
    exc: BaseException,
    *,
    auth_provider: str | None = None,
    credential_id: str | None = None,
    message: str | None = None,
) -> ProviderFailure | None:
    """Name the failure, or return ``None`` if it is not the provider's.

    ``None`` means "not recognisably a provider refusal" and leaves the
    caller's existing behaviour untouched. That matters: a bug in our own
    code dressed up as a provider failure would send the user to reconnect
    an account that was never broken.
    """
    kind = _kind_of(exc)
    if kind is None:
        return None
    return ProviderFailure(
        kind=kind,
        message=message or _humanize(exc) or str(exc) or type(exc).__name__,
        auth_provider=auth_provider,
        credential_id=credential_id,
        resets_at=_resets_at(exc),
    )


# HTTP status the compat gateway should answer with for each kind, so the
# CLI upstream of it can tell "stop asking" from "try again". A blanket 502
# reads as a server fault and invites a retry that cannot succeed.
_STATUS_BY_KIND: dict[ProviderFailureKind, int] = {
    ProviderFailureKind.AUTH_EXPIRED: 401,
    ProviderFailureKind.INVALID_CREDENTIAL: 401,
    ProviderFailureKind.ENTITLEMENT_REQUIRED: 402,
    ProviderFailureKind.POLICY_DENIED: 403,
    ProviderFailureKind.MODEL_UNAVAILABLE: 404,
    ProviderFailureKind.USAGE_LIMIT: 429,
    ProviderFailureKind.TRANSIENT: 503,
}


def status_for(failure: "ProviderFailure") -> int:
    """The status a gateway should return for this failure."""
    return _STATUS_BY_KIND.get(failure.kind, 502)


def _kind_of(exc: BaseException) -> ProviderFailureKind | None:
    if isinstance(exc, UserPaywalledError):
        return ProviderFailureKind.ENTITLEMENT_REQUIRED
    # Codex runs behind a CLI, so its failures arrive as transport
    # exceptions rather than HTTP errors. Only the ones whose meaning is
    # unambiguous are named; a bare CodexTransportError could be anything.
    if isinstance(exc, CodexCredentialIntegrityError):
        return ProviderFailureKind.INVALID_CREDENTIAL
    if isinstance(
        exc,
        (
            CodexInvocationTimeoutError,
            CodexTransportOverloadedError,
            CodexCredentialBusyError,
        ),
    ):
        return ProviderFailureKind.TRANSIENT
    if isinstance(exc, AuthenticationError):
        return ProviderFailureKind.AUTH_EXPIRED
    if isinstance(exc, PermissionDeniedError):
        return ProviderFailureKind.POLICY_DENIED
    if isinstance(exc, RateLimitError):
        return ProviderFailureKind.USAGE_LIMIT
    if isinstance(exc, NotFoundError):
        return ProviderFailureKind.MODEL_UNAVAILABLE
    if isinstance(exc, (APIConnectionError, APITimeoutError, InternalServerError)):
        return ProviderFailureKind.TRANSIENT
    if isinstance(exc, APIStatusError):
        # Any other 5xx is worth another attempt; a 4xx we have not named
        # above is a request we built wrong, which is ours, not the
        # provider's, and must not be dressed up as a provider refusal.
        return ProviderFailureKind.TRANSIENT if exc.status_code >= 500 else None
    return None


def _resets_at(exc: BaseException) -> int | None:
    """A reset time only if the provider actually sent one.

    Read from the standard ``x-ratelimit-reset-*`` response headers. Anything
    unparseable is dropped rather than approximated -- a wrong time is worse
    than no time, because the user schedules around it.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("x-ratelimit-reset-requests") or headers.get(
        "x-ratelimit-reset-tokens"
    )
    if not raw:
        return None
    try:
        return _to_unix_seconds(str(raw))
    except (ValueError, TypeError):
        logger.debug("Unparseable rate-limit reset header: %r", raw)
        return None


def _to_unix_seconds(raw: str) -> int | None:
    """Absolute epoch seconds, or ``None`` for a shape we do not understand.

    OpenAI sends a duration like ``60s`` / ``1m30s`` / ``6m0s`` / ``2h``;
    some compatible servers send an absolute epoch instead. A duration is
    still the provider reporting a reset -- "60 seconds from now" is a fact,
    not a guess -- so it is anchored to the current clock. What is refused is
    inventing a reset the provider never mentioned at all.
    """
    value = raw.strip().lower()
    if not value:
        return None
    if value.isdigit():
        seconds = int(value)
        # Below this, it is a duration in seconds rather than an epoch:
        # 10**9 seconds is 2001, so no real reset epoch is ever smaller.
        return seconds if seconds > 10**9 else int(time.time()) + seconds
    total = _duration_seconds(value)
    return int(time.time() + total) if total is not None else None


_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)")


def _duration_seconds(value: str) -> float | None:
    """Seconds in a compound duration, or ``None`` if it is not one.

    Whole-string match, so a value carrying anything we did not parse is
    rejected rather than silently half-read.
    """
    parts = _DURATION_PART.findall(value)
    if not parts:
        return None
    if "".join(number + unit for number, unit in parts) != value:
        return None
    return sum(float(number) * _DURATION_UNITS[unit] for number, unit in parts)
