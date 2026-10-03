"""Why an OAuth refresh failed, and what that means for the stored credential.

A refresh failure used to reach the logs as a bare exception, so nothing said
whether the provider was down for a minute or had revoked the grant for good,
and the next turn replayed the same dead refresh token. This module reads the
provider's answer (HTTP status plus the RFC 6749 §5.2 ``error`` code), reports
it with the provider and credential id, and records a definitive refusal on
the credential so later turns ask the user to reconnect instead.

Nothing here ever logs or stores a token: only the status, the error code and
the exception's class name leave this module.
"""

import json
import logging
import re
import time
from typing import Any

import sentry_sdk
from pydantic import BaseModel, ValidationError

from backend.data.model import OAuth2Credentials
from backend.integrations.providers import provider_key
from backend.util.request import HTTPClientError

# Codes a provider returns when the refresh token itself is dead, so retrying
# with it can never succeed. GitHub answers ``bad_refresh_token`` (with HTTP
# 200) where the RFC says ``invalid_grant``.
DEFINITIVE_REFRESH_ERRORS = frozenset({"invalid_grant", "bad_refresh_token"})

# RFC 6749 §5.2 codes plus GitHub's variant. Used to recognise the code in an
# error that only carries it as text (e.g. google-auth's RefreshError).
_KNOWN_ERROR_CODES = DEFINITIVE_REFRESH_ERRORS | {
    "invalid_request",
    "invalid_client",
    "unauthorized_client",
    "unsupported_grant_type",
    "invalid_scope",
    "invalid_token",
}
_KNOWN_ERROR_CODE_RE = re.compile(
    r"\b(" + "|".join(sorted(_KNOWN_ERROR_CODES, key=len, reverse=True)) + r")\b"
)
# An error code is a short identifier; anything else in the ``error`` field
# is free text we do not want in a tag.
_ERROR_CODE_SHAPE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

RECONNECT_REQUIRED_KEY = "reconnect_required"


class OAuthTokenRequestError(Exception):
    """A provider's token endpoint refused the request.

    Raised by handlers whose transport does not surface the HTTP status on its
    own (google-auth), or that report errors in a 200 body (GitHub). The
    message never includes the provider's body, only the status and code.
    """

    def __init__(
        self,
        provider: str,
        *,
        status_code: int | None,
        error_code: str | None,
    ):
        super().__init__(
            f"{provider} token request failed "
            f"(HTTP {status_code if status_code is not None else 'unknown'}, "
            f"error={error_code or 'unknown'})"
        )
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code


class RefreshFailure(BaseModel):
    status_code: int | None
    error_code: str | None

    @property
    def definitive(self) -> bool:
        """The refresh token is dead; retrying it cannot succeed."""
        return self.error_code in DEFINITIVE_REFRESH_ERRORS


def describe_refresh_failure(exc: BaseException) -> RefreshFailure:
    """The provider's HTTP status and error code, from *exc* or its causes."""
    status_code: int | None = None
    error_code: str | None = None
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if status_code is None:
            status_code = _status_of(current)
        if error_code is None:
            error_code = _error_code_of(current)
        if status_code is not None and error_code is not None:
            break
        current = current.__cause__ or current.__context__
    return RefreshFailure(status_code=status_code, error_code=error_code)


def _status_of(exc: BaseException) -> int | None:
    for value in (getattr(exc, "status_code", None), getattr(exc, "status", None)):
        if isinstance(value, int):
            return value
    return None


def _error_code_of(exc: BaseException) -> str | None:
    code = getattr(exc, "error_code", None)
    if isinstance(code, str) and code:
        return code
    body = getattr(exc, "body", None)
    if isinstance(body, (bytes, str)):
        if code := _error_code_from_payload(body):
            return code
    # google-auth: RefreshError(message, response_data)
    for arg in getattr(exc, "args", ()):
        if isinstance(arg, dict) and (code := _error_code_from_mapping(arg)):
            return code
    match = _KNOWN_ERROR_CODE_RE.search(str(exc))
    return match.group(1) if match else None


def _error_code_from_payload(body: bytes | str) -> str | None:
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return None
    return _error_code_from_mapping(payload) if isinstance(payload, dict) else None


def _error_code_from_mapping(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, str) and _ERROR_CODE_SHAPE.match(error):
        return error
    return None


def report_refresh_failure(
    logger: logging.Logger,
    *,
    provider: str,
    credential_id: str,
    failure: RefreshFailure,
    exc: BaseException,
) -> None:
    """Log why a refresh failed, with a Sentry breadcrumb and tags.

    A definitive refusal is logged at ERROR, so it becomes a Sentry event
    tagged with the provider and code; a transient one is a WARNING that only
    leaves a breadcrumb for whatever error follows it. No exc_info: the
    exception chain can carry the provider's raw body.
    """
    fields = {
        "provider": provider,
        "credential_id": credential_id,
        "http_status": failure.status_code,
        "oauth_error": failure.error_code,
        "definitive": failure.definitive,
        "exception_type": type(exc).__name__,
    }
    sentry_sdk.add_breadcrumb(
        category="oauth.refresh",
        message=f"{provider} OAuth refresh failed",
        level="error" if failure.definitive else "warning",
        data=fields,
    )
    with sentry_sdk.new_scope() as scope:
        scope.set_tag("oauth_refresh_provider", provider)
        scope.set_tag("oauth_refresh_status", str(failure.status_code))
        scope.set_tag("oauth_refresh_error", failure.error_code or "unknown")
        logger.log(
            logging.ERROR if failure.definitive else logging.WARNING,
            "OAuth refresh failed for %s credential #%s: HTTP %s, error=%s " "(%s)%s",
            provider,
            credential_id,
            failure.status_code if failure.status_code is not None else "unknown",
            failure.error_code or "unknown",
            type(exc).__name__,
            "; marking it as needing reconnect" if failure.definitive else "",
            extra={"json_fields": fields},
        )


# -- The reconnect marker, stored in the credential's metadata -- #


class ReconnectRequired(BaseModel):
    error_code: str
    status_code: int | None = None
    marked_at: int = 0

    def reason(self, provider_name: str) -> str:
        """Why, in words a user can act on; never carries a secret."""
        status = f", HTTP {self.status_code}" if self.status_code is not None else ""
        return (
            f"{provider_name} refused to refresh the saved sign-in "
            f"({self.error_code}{status})"
        )


def reconnect_required(credentials: object) -> ReconnectRequired | None:
    """The reconnect marker on *credentials*, if a refresh was refused for good."""
    if not isinstance(credentials, OAuth2Credentials):
        return None
    raw = (credentials.metadata or {}).get(RECONNECT_REQUIRED_KEY)
    if raw is None:
        return None
    try:
        return ReconnectRequired.model_validate(raw)
    except ValidationError:
        return None


def mark_reconnect_required(
    credentials: OAuth2Credentials, failure: RefreshFailure
) -> tuple[OAuth2Credentials, ReconnectRequired]:
    """A copy of *credentials* carrying the reconnect marker, and the marker."""
    marker = ReconnectRequired(
        error_code=failure.error_code or "unknown",
        status_code=failure.status_code,
        marked_at=int(time.time()),
    )
    marked = credentials.model_copy(deep=True)
    marked.metadata = {
        **(credentials.metadata or {}),
        RECONNECT_REQUIRED_KEY: marker.model_dump(),
    }
    return marked, marker


def without_reconnect_required(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (metadata or {}).items() if k != RECONNECT_REQUIRED_KEY}


def provider_display_name(provider: str) -> str:
    return provider_key(provider).replace("_", " ").title()


class CredentialsNeedReconnectError(HTTPClientError):
    """The stored credential's refresh token was refused for good.

    An ``HTTPClientError`` so callers that already turn a provider's refusal
    into a "reconnect" card (run_block) do so for this too, without replaying
    the dead refresh token first.
    """

    def __init__(self, provider: str, credential_id: str, marker: ReconnectRequired):
        self.provider = provider
        self.credential_id = credential_id
        self.marker = marker
        super().__init__(
            f"{marker.reason(provider_display_name(provider))}; "
            "reconnect the account to continue",
            marker.status_code or 400,
        )
