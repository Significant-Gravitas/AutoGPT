"""
Regression tests for the v2 external API contract.

Each test here pins a defect that shipped once and was invisible to the rest of
the suite: a schema that only fails when FastAPI lazily builds it, a route that
only fails once a sibling route is registered, and an auth path only the MCP
transport exercises.
"""

from datetime import datetime, timezone
from typing import Any

import pytest
import pytest_mock
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from prisma.enums import APIKeyPermission
from starlette.routing import Match

from backend.api.external.middleware import resolve_auth_info
from backend.data.auth.base import APIAuthorizationInfo

TEST_USER_ID = "test-user-id"

_api_key_auth = APIAuthorizationInfo(
    user_id=TEST_USER_ID,
    scopes=list(APIKeyPermission),
    type="api_key",
    created_at=datetime.now(tz=timezone.utc),
)


def test_openapi_schema_builds():
    """Every response model must be resolvable at runtime.

    `/docs`, `/redoc` and `/openapi.json` all 500 when one is not, and FastAPI
    builds response models lazily, so importing the app is not enough.
    """
    from backend.api.external.v2.app import v2_app

    schema = v2_app.openapi()

    assert schema["paths"], "v2 OpenAPI schema has no paths"
    assert "/search" in schema["paths"]


def test_static_routes_are_not_shadowed_by_path_params():
    """Starlette matches in registration order.

    A static path registered after a sibling `/{id}` route is unreachable: the
    parameterised route matches the literal segment first.
    """
    from backend.api.external.v2.routes import v2_router

    for method, path, expected_endpoint in [
        ("GET", "/runs/reviews", "list_reviews"),
        ("GET", "/runs", "list_runs"),
    ]:
        endpoint = _first_matching_endpoint(v2_router, method, path)
        assert (
            endpoint == expected_endpoint
        ), f"{method} {path} resolves to {endpoint}, not {expected_endpoint}"


async def test_api_key_sent_as_bearer_resolves_to_its_user(
    mocker: pytest_mock.MockFixture,
):
    """MCP clients can only send credentials in the `Authorization` header.

    Resolving those to `None` silently drops the request into the anonymous
    per-IP rate-limit bucket (5 req/min), which an MCP handshake exhausts.
    """
    mocker.patch(
        "backend.api.external.middleware.validate_api_key",
        return_value=_api_key_auth,
    )
    validate_access_token = mocker.patch(
        "backend.api.external.middleware.validate_access_token"
    )

    auth = await resolve_auth_info(
        api_key=None,
        bearer=HTTPAuthorizationCredentials(scheme="Bearer", credentials="agpt_test"),
    )

    assert auth is not None and auth.user_id == TEST_USER_ID
    validate_access_token.assert_not_called()


async def test_invalid_bearer_still_raises_401(mocker: pytest_mock.MockFixture):
    from backend.data.auth.oauth import InvalidTokenError

    mocker.patch("backend.api.external.middleware.validate_api_key", return_value=None)
    mocker.patch(
        "backend.api.external.middleware.validate_access_token",
        side_effect=InvalidTokenError("nope"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await resolve_auth_info(
            api_key=None,
            bearer=HTTPAuthorizationCredentials(scheme="Bearer", credentials="junk"),
        )

    assert exc_info.value.status_code == 401


async def test_bearer_api_key_gets_the_authenticated_rate_limit(
    mocker: pytest_mock.MockFixture,
):
    """The consequence of the above, at the layer a client actually feels."""
    from backend.api.external.v2 import global_rate_limit

    mocker.patch(
        "backend.api.external.middleware.validate_api_key",
        return_value=_api_key_auth,
    )
    authenticated = mocker.patch.object(
        global_rate_limit._authenticated_limiter, "check"
    )
    anonymous = mocker.patch.object(global_rate_limit._anonymous_limiter, "check")

    await _call_rate_limit_middleware(headers=[(b"authorization", b"Bearer agpt_test")])

    authenticated.assert_awaited_once_with(TEST_USER_ID)
    anonymous.assert_not_awaited()


def _first_matching_endpoint(router: Any, method: str, path: str) -> str | None:
    """Name of the endpoint Starlette would dispatch `method path` to."""
    scope = {"type": "http", "method": method, "path": path, "headers": []}
    for route in router.routes:
        match, _ = route.matches(scope)
        if match == Match.FULL:
            return route.endpoint.__name__
    return None


async def _call_rate_limit_middleware(headers: list[tuple[bytes, bytes]]) -> None:
    from backend.api.external.v2.global_rate_limit import GlobalRateLimitMiddleware

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/runs",
        "headers": headers,
        "client": ("1.2.3.4", 0),
    }
    await GlobalRateLimitMiddleware(app)(scope, receive, send)
