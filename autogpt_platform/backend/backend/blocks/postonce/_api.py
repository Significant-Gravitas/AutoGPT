"""Thin async client for the PostOnce public API (https://docs.postonce.to)."""

from typing import Any

from pydantic import TypeAdapter, ValidationError

from backend.sdk import APIKeyCredentials, BaseModel, Requests
from backend.util.request import Response

POSTONCE_API_URL = "https://postonce.to"
API_V1 = f"{POSTONCE_API_URL}/api/public/v1"

# Reads are retried on 429/5xx a few times; writes are sent once, because a
# repeated create would post twice unless the caller passed an idempotency key.
READ_RETRY_ATTEMPTS = 3
RETRY_MAX_WAIT_SECONDS = 10.0

# Every PostOnce success body is a JSON object such as {"data": ...}.
_OBJECT_BODY = TypeAdapter(dict[str, Any])


class PostOnceAPIError(ValueError):
    """A non-2xx response from the PostOnce API, with its HTTP status."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class PostOnceUnknownOutcomeError(PostOnceAPIError):
    """A write got a 2xx status but an unreadable body: it may have succeeded."""


class PostOnceClient:
    def __init__(self, credentials: APIKeyCredentials):
        self.reads = _requests(credentials, READ_RETRY_ATTEMPTS)
        self.writes = _requests(credentials, 1)

    async def list_accounts(self) -> list[dict[str, Any]]:
        body = await self._call("GET", "/accounts")
        return list(body.get("data") or [])

    async def create_post(
        self, payload: dict[str, Any], idempotency_key: str = ""
    ) -> dict[str, Any]:
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        body = await self._call("POST", "/posts", json_body=payload, headers=headers)
        return body.get("data") or {}

    async def get_post(self, post_id: str) -> dict[str, Any]:
        body = await self._call("GET", f"/posts/{post_id}")
        return body.get("data") or {}

    async def cancel_post(self, post_id: str) -> dict[str, Any]:
        body = await self._call("POST", f"/posts/{post_id}/cancel")
        return body.get("data") or {}

    async def get_post_analytics(self, post_id: str) -> dict[str, Any]:
        body = await self._call("GET", f"/posts/{post_id}/analytics")
        return body.get("data") or {}

    async def _call(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        requests = self.reads if method == "GET" else self.writes
        response = await requests.request(
            method,
            f"{API_V1}{path}",
            json=json_body,
            headers=headers,
            allow_redirects=False,
        )
        if not response.ok:
            raise PostOnceAPIError(_error_message(response), response.status)
        try:
            return _OBJECT_BODY.validate_json(response.content)
        except ValidationError as error:
            if method != "GET":
                raise PostOnceUnknownOutcomeError(
                    f"PostOnce accepted the {method} {path} request (HTTP "
                    f"{response.status}) but returned an unexpected body, so the "
                    "change may already have been made. Check it in PostOnce "
                    "before retrying.",
                    response.status,
                ) from error
            raise PostOnceAPIError(
                f"PostOnce returned an unexpected success body: {error}",
                response.status,
            ) from error


def _requests(credentials: APIKeyCredentials, attempts: int) -> Requests:
    return Requests(
        trusted_origins=[POSTONCE_API_URL],
        raise_for_status=False,
        extra_headers={
            "Authorization": f"Bearer {credentials.api_key.get_secret_value()}"
        },
        retry_max_attempts=attempts,
        retry_max_wait=RETRY_MAX_WAIT_SECONDS,
    )


class _PostOnceError(BaseModel):
    code: str | None = None
    message: str | None = None
    hint: str | None = None


class _PostOnceErrorBody(BaseModel):
    # Every error is {"error": {"code", "message", "hint"?}}.
    error: _PostOnceError | None = None


def _error_message(response: Response) -> str:
    """Build a readable message from PostOnce's error envelope."""
    try:
        error = _PostOnceErrorBody.model_validate_json(response.content).error
    except ValidationError:
        error = None
    if error is None or not error.message:
        return f"PostOnce returned HTTP {response.status}: {response.text()[:500]}"
    message = f"PostOnce returned HTTP {response.status} ({error.code}): "
    message += error.message
    return f"{message} {error.hint}" if error.hint else message
