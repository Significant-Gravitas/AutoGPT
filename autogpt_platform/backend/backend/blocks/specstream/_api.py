"""Async client for the SpecStream REST API.

Reading a request and pricing it spend nothing. Only ``authorize`` reserves
money, and only for the exact quote and maximum it is given.
"""

from typing import Any
from urllib.parse import quote as quote_path

from pydantic import ValidationError

from backend.sdk import APIKeyCredentials, BaseModel, Field, Requests
from backend.util.request import Response

from ._models import Execution, Quote, Reading, RowsPage, RunStatus

SPECSTREAM_API_URL = "https://api.specstream.ai"
ROWS_PER_PAGE = 100


class ErrorBody(BaseModel):
    code: str = ""
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class Envelope(BaseModel):
    data: dict[str, Any] | None = None
    error: ErrorBody | None = None


class SpecStreamError(ValueError):
    """An error envelope from the API, rendered for a person."""

    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        super().__init__(f"SpecStream {code or status}: {message}")


class SpecStreamClient:
    """Thin wrapper over the SpecStream API v1."""

    def __init__(self, credentials: APIKeyCredentials):
        # Sent with each request, and redirects are off, so the bearer token
        # never goes to another address.
        self._auth_header = credentials.auth_header()
        self.requests = Requests(raise_for_status=False, retry_max_attempts=5)

    async def compile(self, text: str) -> Reading:
        data = await self._request("POST", "/v1/parse", {"q": text})
        return Reading.model_validate(data)

    async def quote(self, ast_hash: str, requested_units: int) -> Quote:
        data = await self._request(
            "POST",
            "/v1/capacity/quotes",
            {
                "ast_hash": ast_hash,
                "operation": "find",
                "requested_units": requested_units,
            },
        )
        return Quote.model_validate(data)

    async def get_quote(self, quote_id: str) -> Quote:
        data = await self._request(
            "GET", f"/v1/capacity/quotes/{quote_path(quote_id, safe='')}"
        )
        return Quote.model_validate(data)

    async def authorize(
        self, quote_id: str, quote_hash: str, confirm_max_cents: int, key: str
    ) -> Execution:
        data = await self._request(
            "POST",
            "/v1/capacity/executions",
            {
                "quote_id": quote_id,
                "quote_hash": quote_hash,
                "confirm_max_cents": confirm_max_cents,
                "idempotency_key": key,
            },
        )
        return Execution.model_validate(data)

    async def get_run(self, find_id: str) -> RunStatus:
        data = await self._request("GET", f"/v1/finds/{quote_path(find_id, safe='')}")
        return RunStatus.model_validate(data)

    async def list_rows(
        self, find_id: str, cursor: str | None = None, limit: int = ROWS_PER_PAGE
    ) -> RowsPage:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = await self._request(
            "GET", f"/v1/finds/{quote_path(find_id, safe='')}/rows", params=params
        )
        return RowsPage.model_validate(data)

    async def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self.requests.request(
            method,
            f"{SPECSTREAM_API_URL}{path}",
            headers={"Authorization": self._auth_header, "Accept": "application/json"},
            params=params,
            json=body,
            allow_redirects=False,
        )
        return unwrap(response)


def unwrap(response: Response) -> dict[str, Any]:
    """Return the ``data`` object of a success, or raise the API's own error."""
    text = response.text()
    try:
        envelope = Envelope.model_validate_json(text) if text else Envelope()
    except ValidationError:
        raise SpecStreamError(
            response.status, "invalid_response", f"not JSON: {text[:200]}"
        ) from None
    if not response.ok:
        raise _error_from(response.status, envelope.error)
    if envelope.error is not None or envelope.data is None:
        raise SpecStreamError(
            response.status, "invalid_response", "no data in the response"
        )
    return envelope.data


def _error_from(status: int, error: ErrorBody | None) -> SpecStreamError:
    if error is None:
        return SpecStreamError(status, "http_error", f"HTTP {status}")
    message = (error.message or f"HTTP {status}").rstrip()
    code = str(error.details.get("commercial_code") or error.code or "http_error")
    if status == 401:
        message = f"{message.rstrip('.')}. Check the SpecStream API key."
    elif status == 403:
        message = (
            f"{message.rstrip('.')}. Check the spending limit of the SpecStream "
            "account and the scopes of the API key."
        )
    return SpecStreamError(status, code, message)
