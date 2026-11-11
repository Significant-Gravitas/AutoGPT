"""Why a Gemini Live connect or session failed, as far as the vendor said.

Google documents no error or close code for a refused Live session, so the
relay reads what each failure carries:

- a connection the server refused outright: ``websockets``' ``InvalidStatus``,
  which the SDK lets through, with the HTTP status;
- a socket the server closed, during setup or mid-session: the SDK's
  ``APIError``, with the close code and reason, or ``websockets``'
  ``ConnectionClosed`` from a send;
- an error the server sent as JSON: the SDK's ``APIError``, with its HTTP
  code and API status.

From the HTTP status and the API status alone, never the free text of a
message, it gives a bounded :data:`Reason`. A drop that follows the session's
closing notice (GoAway) is a reset the server warned of.

:meth:`VendorFailure.describe` is what gets logged. It carries no API key,
token or credential in a URL, and nothing quoted from the request, which is
how the vendor echoes content it refused: nothing the person said.
"""

import re
from typing import Literal

from google.genai import errors
from pydantic import BaseModel, ConfigDict
from websockets.exceptions import ConnectionClosed, InvalidStatus

Reason = Literal[
    "quota", "payment", "forbidden", "unavailable", "server_reset", "other"
]

# Where in a call it failed: its first session's connect, a replacement's, or a
# session already running.
Stage = Literal["connect", "reconnect", "session"]

# A close reason is at most 123 bytes on the wire; anything longer is cut.
MAX_REASON_CHARS = 160
REDACTED = "[redacted]"
# An API status is a name such as RESOURCE_EXHAUSTED; anything else is left
# out.
_API_STATUS = re.compile(r"[A-Z][A-Z_]{1,39}")
# What never reaches a log, each with what stands in for it.
_SECRETS: tuple[tuple[re.Pattern[str], str], ...] = (
    # A Google API key.
    (re.compile(r"AIza[0-9A-Za-z_\-]{10,}"), REDACTED),
    # A credential in a URL's user info, or in its query.
    (re.compile(r"(://)[^/\s@]+@"), rf"\1{REDACTED}@"),
    (
        re.compile(r"(?i)\b(key|api_key|access_token|token|sig|signature)=[^&\s]+"),
        rf"\1={REDACTED}",
    ),
    # A bearer token.
    (re.compile(r"(?i)\b(bearer)\s+\S+"), rf"\1 {REDACTED}"),
    # Anything quoted.
    (re.compile(r"\"[^\"]*\"|'[^']*'|“[^”]*”"), REDACTED),
)


class VendorFailure(BaseModel):
    """What a failed Live connect or session said about why."""

    model_config = ConfigDict(frozen=True)

    reason: Reason = "other"
    # The exception's class, which is all a failure the vendor explained
    # nothing about has.
    error_type: str
    http_status: int | None = None
    api_status: str | None = None
    close_code: int | None = None
    close_reason: str | None = None

    def describe(self) -> str:
        """The failure in one line, for a log."""
        said = [self.error_type]
        if self.http_status is not None:
            said.append(f"HTTP {self.http_status}")
        if self.api_status is not None:
            said.append(self.api_status)
        if self.close_code is not None:
            said.append(f"close {self.close_code}")
        if self.close_reason:
            said.append(f"reason: {self.close_reason}")
        return f"{self.reason} ({', '.join(said)})"

    def summary(self) -> str:
        """The failure in a few words, for the call's record."""
        return f"{self.reason} ({self.error_type})"


def vendor_failure(
    error: BaseException, *, after_notice: bool = False
) -> VendorFailure:
    """*error*, from a Live connect or a running session, read for why it
    failed. *after_notice*: the session had sent its closing notice before it
    dropped."""
    failure = _read(error)
    return failure.model_copy(
        update={"reason": _reason_of(failure, after_notice=after_notice)}
    )


def _read(error: BaseException) -> VendorFailure:
    error_type = type(error).__name__
    match error:
        case InvalidStatus():
            return VendorFailure(
                error_type=error_type, http_status=error.response.status_code
            )
        case errors.APIError():
            return _read_api_error(error, error_type)
        case ConnectionClosed():
            close = error.rcvd
            if close is None:
                return VendorFailure(error_type=error_type)
            return VendorFailure(
                error_type=error_type,
                close_code=close.code,
                close_reason=scrubbed(close.reason),
            )
        case _:
            return VendorFailure(error_type=error_type)


def _read_api_error(error: errors.APIError, error_type: str) -> VendorFailure:
    """The SDK's error: an HTTP code with the API status a JSON error named,
    or, for a socket the server closed, the close code and reason."""
    code = error.code if isinstance(error.code, int) else None
    status = error.status if isinstance(error.status, str) else None
    api_status = status if status and _API_STATUS.fullmatch(status) else None
    if code is not None and 100 <= code <= 599:
        return VendorFailure(
            error_type=error_type, http_status=code, api_status=api_status
        )
    reason = error.details if isinstance(error.details, str) else None
    return VendorFailure(
        error_type=error_type,
        api_status=api_status,
        close_code=code,
        close_reason=scrubbed(reason),
    )


def _reason_of(failure: VendorFailure, *, after_notice: bool) -> Reason:
    status, api_status = failure.http_status, failure.api_status
    if status == 429 or api_status == "RESOURCE_EXHAUSTED":
        return "quota"
    if status == 402:
        return "payment"
    if status in (401, 403) or api_status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        return "forbidden"
    if status == 503 or api_status == "UNAVAILABLE":
        return "unavailable"
    if after_notice:
        return "server_reset"
    return "other"


def scrubbed(text: str | None) -> str | None:
    """*text* with no key, token or URL credential, and nothing quoted, cut
    to ``MAX_REASON_CHARS``; None for nothing left."""
    if not text:
        return None
    for pattern, stand_in in _SECRETS:
        text = pattern.sub(stand_in, text)
    text = " ".join(text.split())[:MAX_REASON_CHARS]
    return text or None
