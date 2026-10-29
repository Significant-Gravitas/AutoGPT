"""Thin async client for the Treg gateway (https://treg.to).

Treg relays each call to the provider that serves it and reports what it
charged in response headers. This module sends one request, reads those
headers and turns failures into readable errors; it never retries, because a
repeated call to a paid provider can be billed again.
"""

import asyncio
import json
import re
from typing import Any, Literal

from aiohttp import ClientError, ClientTimeout

from backend.sdk import APIKeyCredentials, BaseModel, Requests
from backend.util.request import Response

TREG_URL = "https://treg.to"

# A routed call may wait up to 60 s on an asynchronous provider, and a page
# extraction can take a while; stop well before aiohttp's 300 s default so a
# hung call doesn't hold an executor worker.
REQUEST_TIMEOUT = ClientTimeout(total=150)
REQUEST_DEADLINE_SECONDS = 150

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]

# Headers the caller may not set on a call: the credential is ours to send.
RESERVED_HEADERS = frozenset({"x-treg-token", "authorization", "cookie", "host"})

# A catalog tool id, e.g. "treg.people.search".
CATALOG_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# A catalog id ("treg.people.search"), a team tool and its path
# ("intercom/conversations"), or an upstream URL behind the prefix
# ("https://api.stripe.com/v1/charges"). The query goes in its own input.
_TOOL_PATTERN = re.compile(
    r"^(?:https://[^\s?#\\]+|[A-Za-z0-9][A-Za-z0-9._-]*(?:/[^\s?#\\]*)?)$"
)


class TregResponse(BaseModel):
    """One Treg answer: the provider's status and body plus Treg's metadata."""

    status: int
    body: Any
    cost_usd: float
    call_id: str | None
    served_by: str | None
    # Set when the tool started an asynchronous task (a generation): how to
    # poll it. The charge is then a reservation, settled when the task ends.
    async_task: dict[str, Any] | None
    cached: bool
    # The provider reported a failure inside a 2xx answer.
    upstream_failed: bool

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class TregError(ValueError):
    """A failed Treg call. ValueError puts it on the block's error output."""


def validate_tool(tool: str) -> str:
    """Check a tool reference before it is put in a URL path.

    Rejects anything that could step out of ``/call/`` to another Treg route
    with the user's token, such as ``../orgs/...`` or an encoded slash.
    """
    tool = tool.strip()
    lowered = tool.lower()
    segments = tool.removeprefix("https://").split("/")
    if (
        not _TOOL_PATTERN.match(tool)
        or any(segment in (".", "..") for segment in segments)
        or "%2e" in lowered
        or "%2f" in lowered
        or "%5c" in lowered
    ):
        raise TregError(
            f"'{tool}' is not a Treg tool. Use a catalog id such as "
            "'treg.people.email.find', a team tool and its path such as "
            "'intercom/conversations', or an https:// upstream URL."
        )
    return tool


async def send(
    credentials: APIKeyCredentials,
    method: HttpMethod,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> TregResponse:
    """Send one request to Treg and return the answer, whatever its status.

    Network failures and timeouts raise TregError. HTTP errors are returned,
    so the caller can record any charge before reporting them.
    """
    requests = Requests(
        trusted_origins=[TREG_URL],
        raise_for_status=False,
        # One attempt: a retried call can bill again (a routed miss, a paid
        # lookup whose answer was lost on the way back).
        retry_max_attempts=1,
        extra_headers={"X-Treg-Token": credentials.api_key.get_secret_value()},
    )
    request_headers = {
        k: v for k, v in (headers or {}).items() if k.lower() not in RESERVED_HEADERS
    }
    try:
        async with asyncio.timeout(REQUEST_DEADLINE_SECONDS):
            response = await requests.request(
                method,
                f"{TREG_URL}/{path.lstrip('/')}",
                params=_query(params),
                json=body,
                headers=request_headers or None,
                # Requests re-sends extra_headers on every hop and only strips
                # Authorization/Cookie, so a redirect would leak X-Treg-Token.
                allow_redirects=False,
                timeout=REQUEST_TIMEOUT,
            )
    except TimeoutError as e:
        raise TregError(
            "Treg did not answer within "
            f"{REQUEST_DEADLINE_SECONDS} seconds. The provider may still have "
            "answered and charged; check your activity at https://treg.to before "
            "running it again."
        ) from e
    except ClientError as e:
        raise TregError(f"Could not reach Treg: {type(e).__name__} {e}") from e
    return _parse(response)


def _query(params: dict[str, Any] | None) -> list[tuple[str, str]] | None:
    """Flatten query values: lists repeat the key, booleans are lowercase."""
    if not params:
        return None
    pairs: list[tuple[str, str]] = []
    for key, value in params.items():
        for item in value if isinstance(value, list) else [value]:
            if item is None:
                continue
            if isinstance(item, bool):
                item = str(item).lower()
            elif isinstance(item, (dict, list)):
                item = json.dumps(item)
            pairs.append((key, str(item)))
    return pairs or None


def _parse(response: Response) -> TregResponse:
    headers = response.headers
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text()
    async_header = headers.get("X-Treg-Async")
    async_task = None
    if async_header:
        try:
            async_task = json.loads(async_header)
        except ValueError:
            async_task = {"descriptor": async_header}
    return TregResponse(
        status=response.status,
        body=body,
        # A call still pending reports only what it reserved; that is what
        # Treg will settle against the team's balance if the task finishes.
        cost_usd=micro_to_usd(
            headers.get("X-Treg-Cost-Micro") or headers.get("X-Treg-Reserved-Micro")
        ),
        call_id=headers.get("X-Treg-Call-Id"),
        served_by=headers.get("X-Treg-Served-By"),
        async_task=async_task,
        cached=headers.get("X-Treg-Cache", "").lower() in ("hit", "stale"),
        upstream_failed=bool(headers.get("X-Treg-Upstream-Failed")),
    )


def usd_header(value: float) -> str:
    """A USD amount in plain decimal: "5e-05" may not parse as a spending cap."""
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def micro_to_usd(value: Any) -> float:
    """Treg reports money as integer micro-USD; anything unreadable is zero."""
    try:
        return max(0, int(value)) / 1_000_000
    except (TypeError, ValueError):
        return 0.0


def error_message(response: TregResponse) -> str:
    """Explain a failed call: who refused it, why, and what to do next."""
    detail = response.body
    if isinstance(detail, dict) and "detail" in detail:
        detail = detail["detail"]
    code = message = None
    if isinstance(detail, dict):
        code = detail.get("error") or detail.get("code")
        message = detail.get("message") or detail.get("detail")
        topup = detail.get("topup_url")
        if topup and message:
            message = f"{message} Top up at {topup}."
    elif isinstance(detail, str):
        message = detail
    if not isinstance(message, str) or not message:
        message = _snippet(response.body)
    reason = f"{code}: {message}" if code else message
    if response.status == 401:
        reason += ". Check your Treg API key, or make a new one at https://treg.to."
    call = f" (call id {response.call_id})" if response.call_id else ""
    return f"Treg call failed with HTTP {response.status}{call}: {reason}"


def _snippet(body: Any) -> str:
    text = body if isinstance(body, str) else json.dumps(body, default=str)
    return text[:500] or "no details"
