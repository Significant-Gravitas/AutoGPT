"""
Regression tests for the v2 external API contract.

Each test here pins a defect that shipped once and was invisible to the rest of
the suite: a schema that only fails when FastAPI lazily builds it, a route that
only fails once a sibling route is registered, and an auth path only the MCP
transport exercises.
"""

import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, get_origin, get_type_hints
from unittest import mock

import fastapi
import fastapi.testclient
import pydantic
import pytest
import pytest_mock
from fastapi import HTTPException
from fastapi.dependencies.utils import get_flat_params
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials
from prisma.enums import APIKeyPermission, ReviewStatus
from starlette.routing import Match

from backend.api.external.middleware import resolve_auth_info
from backend.api.external.v2.errors import add_v2_exception_handlers
from backend.api.external.v2.pagination import (
    MAX_PAGE,
    Page,
    PageRequest,
    encode_page_cursor,
    encode_token_cursor,
    single_page_request,
)
from backend.api.external.v2.tenancy import TenantContext, require_auth
from backend.api.features.executions.review.model import PendingHumanReviewModel
from backend.data.auth.base import APIAuthorizationInfo
from backend.util.exceptions import NotAuthorizedError, NotFoundError

TEST_USER_ID = "test-user-id"

_api_key_auth = APIAuthorizationInfo(
    user_id=TEST_USER_ID,
    scopes=list(APIKeyPermission),
    type="api_key",
    created_at=datetime.now(tz=timezone.utc),
)

_tenant = TenantContext(
    user_id=TEST_USER_ID,
    scopes=list(APIKeyPermission),
    type="api_key",
    organization_id="test-org-id",
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
    app.dependency_overrides[require_auth] = lambda: _tenant
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


async def test_submitting_reviews_resumes_the_run(mocker: pytest_mock.MockFixture):
    """One approval and one rejection, after which nothing is left pending.

    The resume is the point of the endpoint: the graph sits parked until the
    last review lands, so a failure here strands the run rather than erroring.
    """
    approved = _review("node-a", ReviewStatus.APPROVED)
    rejected = _review("node-b", ReviewStatus.REJECTED)
    _mock_review_db(
        mocker,
        pending=[approved, rejected],
        processed={"node-a": approved, "node-b": rejected},
        still_pending=False,
    )
    add_graph_execution = _mock_resume_path(mocker)

    result = await _submit(
        {"node_exec_id": "node-a", "approved": True},
        {"node_exec_id": "node-b", "approved": False, "message": "no"},
    )

    assert (result.run_id, result.approved_count, result.rejected_count) == (
        "run-1",
        1,
        1,
    )
    add_graph_execution.assert_awaited_once()
    assert add_graph_execution.await_args.kwargs["graph_exec_id"] == "run-1"


async def test_reviews_still_pending_leave_the_run_parked(
    mocker: pytest_mock.MockFixture,
):
    approved = _review("node-a", ReviewStatus.APPROVED)
    _mock_review_db(
        mocker,
        pending=[approved],
        processed={"node-a": approved},
        still_pending=True,
    )
    add_graph_execution = _mock_resume_path(mocker)

    result = await _submit({"node_exec_id": "node-a", "approved": True})

    assert result.approved_count == 1
    add_graph_execution.assert_not_awaited()


async def test_a_review_id_from_another_run_is_rejected(
    mocker: pytest_mock.MockFixture,
):
    _mock_review_db(
        mocker,
        pending=[_review("node-a", ReviewStatus.WAITING)],
        processed={},
        still_pending=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _submit({"node_exec_id": "node-from-another-run", "approved": True})

    assert exc_info.value.status_code == 400
    assert "node-from-another-run" in str(exc_info.value.detail)


async def test_a_failed_resume_does_not_fail_the_submission(
    mocker: pytest_mock.MockFixture,
):
    """The decisions are already persisted; a resume failure must not undo them."""
    approved = _review("node-a", ReviewStatus.APPROVED)
    _mock_review_db(
        mocker,
        pending=[approved],
        processed={"node-a": approved},
        still_pending=False,
    )
    _mock_resume_path(mocker).side_effect = RuntimeError("executor unreachable")

    result = await _submit({"node_exec_id": "node-a", "approved": True})

    assert result.approved_count == 1


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


def test_every_page_response_carries_all_three_fields():
    """`total_count` is present on every list, even where its value is null."""
    from backend.api.external.v2.app import v2_app

    schemas = v2_app.openapi()["components"]["schemas"]
    page_schemas = {k: v for k, v in schemas.items() if k.startswith("Page_")}

    assert page_schemas
    for name, schema in page_schemas.items():
        assert set(schema["required"]) == {
            "items",
            "next_cursor",
            "total_count",
        }, f"{name} requires {schema['required']}"


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
        "create_schedule": 201,
        "create_preset": 201,
        "create_submission": 201,
        "setup_trigger": 201,
        "fork_library_agent": 201,
        "add_agent_to_library": 201,
        "enable_sharing": 201,
        "upload_file": 201,
        "upload_submission_media": 201,
        "execute_agent": 202,
        "run_preset": 202,
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


def test_field_validator_errors_have_the_envelope():
    """A `@field_validator` leaves a live exception in the error's `ctx`.

    Serializing that without a fallback raises a `ValueError` subclass, which
    the 400 handler then catches — turning a 422 into a 400 whose message is a
    serializer complaint.
    """
    response = _error_client(None).post("/validated", json={"host": ""})

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert "Invalid hostname" in json.dumps(body["error"]["details"])


def test_a_paywalled_request_is_not_logged_as_a_server_warning(caplog):
    """402 means the caller lacks credits, which is not a server event."""
    from backend.copilot.rate_limit import UserPaywalledError

    with caplog.at_level(logging.WARNING, logger="backend.api.external.v2.errors"):
        response = _error_client(UserPaywalledError("no credits")).get("/boom")

    assert response.status_code == 402
    assert not caplog.records


def test_http_exception_headers_are_forwarded():
    response = _error_client(
        HTTPException(
            status_code=429, detail="slow down", headers={"Retry-After": "30"}
        )
    ).get("/boom")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "30"


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
    assert PageRequest(limit=10, cursor=encode_token_cursor("t")).token == "t"

    for bad in ["not-a-cursor", _forge({"v": 99, "k": "p", "p": 2}), _forge({"p": 2})]:
        with pytest.raises(HTTPException) as exc_info:
            PageRequest(limit=10, cursor=bad)
        assert exc_info.value.status_code == 400, bad


def test_a_cursor_from_another_endpoint_is_rejected():
    """A wrong-kind cursor must not silently reset the caller to page 1."""
    page_cursor = PageRequest(limit=10, cursor=encode_page_cursor(3))
    token_cursor = PageRequest(limit=10, cursor=encode_token_cursor("2026-01-01"))

    for request, read in [(page_cursor, "token"), (token_cursor, "page")]:
        with pytest.raises(HTTPException) as exc_info:
            getattr(request, read)
        assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        page_cursor.uncounted(["a"])
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        page_cursor.keyset(["a"], next_token=None)
    assert exc_info.value.status_code == 400


def test_a_single_page_endpoint_rejects_a_cursor_before_it_does_any_work():
    """The guard runs in dependency resolution, not the route body.

    `/credits/invoices` calls Stripe; a cursor must not buy an outbound request
    the endpoint then throws away.
    """
    request = PageRequest(limit=10, cursor=encode_page_cursor(2))

    with pytest.raises(HTTPException) as exc_info:
        single_page_request(request)
    assert exc_info.value.status_code == 400
    assert single_page_request(PageRequest(limit=10, cursor=None)).limit == 10


def test_the_documented_cursor_is_one_the_encoder_emits():
    """A reader copying the API guide's example must not get a 400."""
    documented = "eyJ2IjoxLCJrIjoicCIsInAiOjJ9"

    assert encode_page_cursor(2) == documented
    assert PageRequest(limit=10, cursor=documented).page == 2


def test_an_absurd_page_is_rejected_rather_than_passed_to_the_database():
    """An offset that deep is a forged cursor; unbounded it becomes a 500."""
    with pytest.raises(HTTPException) as exc_info:
        PageRequest(limit=100, cursor=encode_page_cursor(MAX_PAGE + 1)).page
    assert exc_info.value.status_code == 400

    assert PageRequest(limit=100, cursor=encode_page_cursor(MAX_PAGE)).page == MAX_PAGE


def test_next_cursor_is_null_on_the_last_page():
    request = PageRequest(limit=10, cursor=None)

    assert request.paged(["a"] * 10, total_count=25).next_cursor is not None
    assert request.paged(["a"] * 10, total_count=10).next_cursor is None
    assert request.paged([], total_count=0).next_cursor is None


def test_total_count_reports_the_whole_result_set():
    request = PageRequest(limit=10, cursor=None)

    assert request.paged(["a"] * 10, total_count=25).total_count == 25
    assert request.slice(["a"] * 25).total_count == 25
    assert request.keyset(["a"], next_token=None).total_count is None
    assert request.uncounted(["a"]).total_count is None


def _forge(payload: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


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

    @app.post("/validated")
    async def validated(body: _ValidatedBody) -> str:
        return body.host

    add_v2_exception_handlers(app)
    return fastapi.testclient.TestClient(app, raise_server_exceptions=False)


async def _submit(*decisions: dict):
    """Call the endpoint directly: a TestClient portal would put the resume
    path's async clients on a different event loop than the test."""
    from .models import AgentRunReviewsSubmitRequest
    from .runs import submit_reviews

    return await submit_reviews(
        request=AgentRunReviewsSubmitRequest.model_validate({"reviews": decisions}),
        run_id="run-1",
        auth=_tenant,
    )


def _review(node_exec_id: str, status: ReviewStatus) -> PendingHumanReviewModel:
    return PendingHumanReviewModel(
        node_exec_id=node_exec_id,
        user_id=TEST_USER_ID,
        graph_exec_id="run-1",
        graph_id="graph-1",
        graph_version=1,
        payload={},
        editable=True,
        status=status,
        created_at=datetime.now(tz=timezone.utc),
    )


def _mock_review_db(
    mocker: pytest_mock.MockFixture,
    pending: list[PendingHumanReviewModel],
    processed: dict[str, PendingHumanReviewModel],
    still_pending: bool,
) -> None:
    from backend.data import execution as execution_db
    from backend.data import human_review

    # The run carries the tenancy the reviews are checked against.
    mocker.patch.object(
        execution_db,
        "get_graph_execution",
        new=mock.AsyncMock(
            return_value=mock.Mock(organization_id=_tenant.organization_id)
        ),
    )
    mocker.patch.object(
        human_review,
        "get_pending_reviews_for_execution",
        new=mock.AsyncMock(return_value=pending),
    )
    mocker.patch.object(
        human_review,
        "process_all_reviews_for_execution",
        new=mock.AsyncMock(return_value=processed),
    )
    mocker.patch.object(
        human_review,
        "has_pending_reviews_for_graph_exec",
        new=mock.AsyncMock(return_value=still_pending),
    )


def _mock_resume_path(mocker: pytest_mock.MockFixture) -> mock.AsyncMock:
    """Stub everything the resume needs; returns the enqueue mock to assert on."""
    from backend.data.graph import GraphSettings
    from backend.executor import utils as execution_utils

    from . import runs

    mocker.patch.object(
        runs,
        "get_user_by_id",
        new=mock.AsyncMock(return_value=mock.Mock(timezone="Europe/Amsterdam")),
    )
    mocker.patch.object(
        runs, "get_graph_settings", new=mock.AsyncMock(return_value=GraphSettings())
    )
    mocker.patch.object(
        runs,
        "get_or_create_workspace",
        new=mock.AsyncMock(return_value=mock.Mock(id="workspace-1")),
    )
    enqueue = mock.AsyncMock()
    mocker.patch.object(execution_utils, "add_graph_execution", new=enqueue)
    return enqueue


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


class _ValidatedBody(pydantic.BaseModel):
    host: str

    @pydantic.field_validator("host")
    @classmethod
    def _reject_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("Invalid hostname: expected a domain like api.example.com")
        return value
