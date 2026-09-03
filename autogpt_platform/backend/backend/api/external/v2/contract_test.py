"""
Regression tests for the v2 external API contract.

Each test here pins a defect that shipped once and was invisible to the rest of
the suite: a schema that only fails when FastAPI lazily builds it, a route that
only fails once a sibling route is registered, and an auth path only the MCP
transport exercises.
"""

import json
from datetime import datetime, timezone
from typing import Any, Optional, get_origin, get_type_hints
from unittest import mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi import HTTPException
from fastapi.dependencies.utils import get_flat_params
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials
from prisma.enums import APIKeyPermission
from starlette.routing import Match

from backend.api.external.middleware import require_auth, resolve_auth_info
from backend.api.external.v2.errors import add_v2_exception_handlers
from backend.api.external.v2.pagination import Page, PageRequest, encode_page_cursor
from backend.data.auth.base import APIAuthorizationInfo
from backend.util.exceptions import NotAuthorizedError, NotFoundError

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


async def test_reviews_are_reachable_over_http():
    """`GET /runs/reviews` end-to-end, not just through route matching.

    Before the ordering fix this answered `404 Run #reviews not found` from
    `get_run`.
    """
    from backend.util.models import Pagination

    from .runs import runs_router

    app = fastapi.FastAPI()
    app.include_router(runs_router, prefix="/runs")
    app.dependency_overrides[require_auth] = lambda: _api_key_auth
    add_v2_exception_handlers(app)

    async def no_reviews(**kwargs) -> tuple[list, Pagination]:
        return [], Pagination(
            total_items=0, total_pages=0, current_page=1, page_size=20
        )

    async def no_such_run(**kwargs) -> None:
        return None

    # `get_run` is stubbed too: when the ordering regresses the request lands
    # there, and this keeps the failure a legible 404 instead of a DB error.
    with (
        mock.patch("backend.data.human_review.get_reviews", new=no_reviews),
        mock.patch("backend.data.execution.get_graph_execution", new=no_such_run),
    ):
        response = fastapi.testclient.TestClient(app).get("/runs/reviews")

    assert response.status_code == 200, response.text
    assert response.json()["items"] == []


async def test_a_string_without_the_key_prefix_is_not_an_api_key():
    """The prefix check short-circuits before any database lookup."""
    from backend.data.auth.api_key import validate_api_key

    assert await validate_api_key("not-an-agpt-key") is None


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


# ---------------------------------------------------------------------------
# Contract shape: one envelope, one error body, deterministic status codes
# ---------------------------------------------------------------------------

# Collection endpoints whose name does not start with `list_`.
_EXTRA_COLLECTION_ENDPOINTS = {"search", "get_folder_tree"}

# Non-2xx responses every operation must document with the error envelope.
_DOCUMENTED_ERROR_STATUSES = {"400", "401", "403", "404", "422", "429", "500"}


def test_every_collection_endpoint_returns_the_page_envelope():
    """`{"items": [...], "next_cursor": ...}` is the only list shape in v2."""
    offenders = []
    for route in _v2_routes():
        returns = get_type_hints(route.endpoint).get("return")
        is_collection = (
            route.name.startswith("list_") or route.name in _EXTRA_COLLECTION_ENDPOINTS
        )
        if is_collection and not _is_page(returns):
            offenders.append(f"{route.name} returns {returns}, not Page[...]")
        if get_origin(returns) is list:
            offenders.append(f"{route.name} returns a bare {returns}")

    assert not offenders, "\n".join(offenders)


def test_every_collection_endpoint_takes_limit_and_cursor():
    for route in _v2_routes():
        if not _is_page(get_type_hints(route.endpoint).get("return")):
            continue
        params = {p.alias for p in get_flat_params(route.dependant)}
        assert {"limit", "cursor"} <= params, f"{route.name} query params: {params}"


def test_status_codes_follow_one_rule_per_method():
    """200 read/update, 204 delete. POST is disambiguated by the table below."""
    allowed = {
        "GET": {200},
        "POST": {200, 201, 202},
        "PUT": {200, 204},
        "PATCH": {200},
        "DELETE": {204},
    }
    offenders = [
        f"{sorted(r.methods)[0]} {r.path} -> {r.status_code or 200} ({r.name})"
        for r in _v2_routes()
        if (r.status_code or 200) not in allowed[sorted(r.methods)[0]]
    ]
    assert not offenders, "\n".join(offenders)


def test_every_post_declares_whether_it_creates_or_enqueues():
    """201 when a resource comes into being, 202 when work is queued, else 200.

    Listed one by one so a new POST has to make the choice, not inherit a default.
    """
    expected = {
        "create_credential": 201,
        "create_folder": 201,
        "create_graph": 201,
        "create_graph_schedule": 201,
        "create_preset": 201,
        "create_submission": 201,
        "setup_trigger": 201,
        "fork_library_agent": 201,
        "add_agent_to_library": 201,
        "enable_sharing": 201,
        "upload_file": 201,
        "upload_submission_media": 201,
        "execute_agent": 202,
        "execute_preset": 202,
        "stop_run": 202,
        "submit_reviews": 202,
        "move_folder": 200,
    }
    actual = {r.name: r.status_code or 200 for r in _v2_routes() if "POST" in r.methods}
    assert actual == expected


def test_every_operation_documents_the_error_envelope():
    from backend.api.external.v2.app import v2_app
    from backend.api.external.v2.errors import ErrorResponse

    expected_ref = f"#/components/schemas/{ErrorResponse.__name__}"
    offenders = []
    for path, operations in v2_app.openapi()["paths"].items():
        for method, operation in operations.items():
            for code, response in operation["responses"].items():
                if code not in _DOCUMENTED_ERROR_STATUSES:
                    continue
                schema = response.get("content", {}).get("application/json", {})
                if schema.get("schema", {}).get("$ref") != expected_ref:
                    offenders.append(f"{method.upper()} {path} {code}: {schema}")

    assert not offenders, "\n".join(offenders)
    assert _DOCUMENTED_ERROR_STATUSES <= set(
        next(iter(v2_app.openapi()["paths"]["/runs"].values()))["responses"]
    )


@pytest.mark.parametrize(
    "exception,expected_status,expected_code",
    [
        (NotFoundError("gone"), 404, "not_found"),
        (NotAuthorizedError("nope"), 403, "forbidden"),
        (ValueError("bad"), 400, "bad_request"),
        (RuntimeError("boom"), 500, "internal_error"),
        (HTTPException(status_code=409, detail="taken"), 409, "conflict"),
    ],
)
def test_error_bodies_all_have_the_envelope(
    exception: Exception, expected_status: int, expected_code: str
):
    response = _error_client(exception).get("/boom")

    assert response.status_code == expected_status
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["code"] == expected_code


def test_validation_errors_have_the_envelope():
    response = _error_client(None).get("/typed?number=not-a-number")

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"]["errors"]


async def test_rate_limit_body_has_the_envelope(mocker: pytest_mock.MockFixture):
    """The limiter is ASGI middleware, outside the handlers — easy to miss."""
    from backend.api.external.v2 import global_rate_limit

    mocker.patch("backend.api.external.middleware.validate_api_key", return_value=None)
    mocker.patch.object(
        global_rate_limit._anonymous_limiter,
        "check",
        side_effect=HTTPException(status_code=429, detail="Rate limit exceeded"),
    )

    sent: list[dict] = []
    await _call_rate_limit_middleware(headers=[], capture=sent)

    start = next(m for m in sent if m["type"] == "http.response.start")
    body = json.loads(
        b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    )
    assert start["status"] == 429
    assert body["error"]["code"] == "rate_limit_exceeded"
    assert body["error"]["message"] == "Rate limit exceeded"


def test_cursors_round_trip_and_reject_garbage():
    assert PageRequest(limit=10, cursor=None).page == 1
    assert PageRequest(limit=10, cursor=encode_page_cursor(4)).page == 4

    with pytest.raises(HTTPException) as exc_info:
        PageRequest(limit=10, cursor="not-a-cursor").page
    assert exc_info.value.status_code == 400


def test_next_cursor_is_null_on_the_last_page():
    request = PageRequest(limit=10, cursor=None)

    assert request.paged(["a"] * 10, total_count=25).next_cursor is not None
    assert request.paged(["a"] * 10, total_count=10).next_cursor is None
    assert request.paged([], total_count=0).next_cursor is None


def _is_page(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, Page)


def _v2_routes() -> list[APIRoute]:
    from backend.api.external.v2.routes import v2_router

    return [r for r in v2_router.routes if isinstance(r, APIRoute)]


def _error_client(exception: Optional[Exception]) -> fastapi.testclient.TestClient:
    from backend.api.external.v2.errors import add_v2_exception_handlers

    app = fastapi.FastAPI()

    @app.get("/boom")
    async def boom() -> None:
        assert exception is not None
        raise exception

    @app.get("/typed")
    async def typed(number: int) -> int:
        return number

    add_v2_exception_handlers(app)
    return fastapi.testclient.TestClient(app, raise_server_exceptions=False)


def _first_matching_endpoint(router: Any, method: str, path: str) -> str | None:
    """Name of the endpoint Starlette would dispatch `method path` to."""
    scope = {"type": "http", "method": method, "path": path, "headers": []}
    for route in router.routes:
        match, _ = route.matches(scope)
        if match == Match.FULL:
            return route.endpoint.__name__
    return None


async def _call_rate_limit_middleware(
    headers: list[tuple[bytes, bytes]], capture: Optional[list[dict]] = None
) -> None:
    from backend.api.external.v2.global_rate_limit import GlobalRateLimitMiddleware

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if capture is not None:
            capture.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/runs",
        "headers": headers,
        "client": ("1.2.3.4", 0),
    }
    await GlobalRateLimitMiddleware(app)(scope, receive, send)
