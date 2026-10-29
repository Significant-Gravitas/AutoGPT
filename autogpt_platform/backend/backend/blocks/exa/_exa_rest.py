"""
Raw HTTP access to the Exa endpoints these blocks call directly.

Monitors live on the main API (``https://api.exa.ai/monitors``) and the Websets
webhook and event endpoints on the Websets API (``https://api.exa.ai/websets/v0``).
Both authenticate with the ``x-api-key`` header and answer failures with
``{"error": ..., "tag": ..., "requestId": ...}``.
"""

from typing import Any
from urllib.parse import quote

from backend.sdk import APIKeyCredentials, Requests
from backend.util.json import loads

EXA_API_BASE = "https://api.exa.ai"
WEBSETS_API_BASE = "https://api.exa.ai/websets/v0"

QueryParams = list[tuple[str, str]]


class ExaApiError(ValueError):
    """An Exa endpoint answered with a non-2xx status."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class ExaRestMixin:
    """Gives a block one mockable entry point for its Exa HTTP calls.

    The block test harness replaces ``call_exa`` on the instance through
    ``test_mock``, so every request a block makes goes through here.
    """

    @staticmethod
    async def call_exa(
        credentials: APIKeyCredentials,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
        params: QueryParams | None = None,
    ) -> dict[str, Any]:
        return await exa_request(credentials, method, url, body=body, params=params)


async def exa_request(
    credentials: APIKeyCredentials,
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    params: QueryParams | None = None,
) -> dict[str, Any]:
    """Send one request to Exa and return the decoded JSON body.

    Raises ``ExaApiError`` carrying Exa's own error message on any non-2xx
    response, so the block run fails with something the user can act on.
    """
    headers = {"x-api-key": credentials.api_key.get_secret_value()}
    response = await Requests(raise_for_status=False).request(
        method, url, headers=headers, json=body, params=params or None
    )
    if not response.ok:
        raise ExaApiError(
            f"Exa API error ({response.status}): {error_message(response.text())}",
            response.status,
        )
    if not response.content:
        return {}
    return response.json()


def error_message(raw_body: str) -> str:
    """Pull the human-readable message (and tag) out of an Exa error body."""
    fallback = raw_body.strip() or "no error details returned"
    parsed = loads(raw_body, fallback=None) if raw_body.strip() else None
    if not isinstance(parsed, dict):
        return fallback
    error = parsed.get("error")
    # The Agent API nests its error: {"error": {"type", "code", "message"}}.
    if isinstance(error, dict):
        message = error.get("message") or fallback
        tag = error.get("code") or error.get("type")
    else:
        message = error or parsed.get("message") or fallback
        tag = parsed.get("tag")
    return f"{message} [{tag}]" if tag else str(message)


def add_param(params: QueryParams, name: str, value: Any) -> None:
    """Append ``name=value`` to query params, skipping unset values."""
    if value is None or value == "":
        return
    if isinstance(value, bool):
        params.append((name, "true" if value else "false"))
        return
    params.append((name, str(value)))


def path_segment(value: str) -> str:
    """Quote one URL path segment, so an ID can't change the route it is sent to."""
    return quote(value, safe="")
