"""
Regression tests for the v2 external API contract.

Each test here pins a defect that shipped once and was invisible to the rest of
the suite: a schema that only fails when FastAPI lazily builds it, a route that
only fails once a sibling route is registered, and an auth path only the MCP
transport exercises.
"""

from datetime import datetime, timezone
from typing import Any
from unittest import mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from prisma.enums import APIKeyPermission, ReviewStatus
from starlette.routing import Match

from backend.api.external.middleware import require_auth, resolve_auth_info
from backend.api.features.executions.review.model import PendingHumanReviewModel
from backend.api.utils.exceptions import add_exception_handlers
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
    add_exception_handlers(app)

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
    assert response.json()["reviews"] == []


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


async def _submit(*decisions: dict):
    """Call the endpoint directly: a TestClient portal would put the resume
    path's async clients on a different event loop than the test."""
    from .models import AgentRunReviewsSubmitRequest
    from .runs import submit_reviews

    return await submit_reviews(
        request=AgentRunReviewsSubmitRequest.model_validate({"reviews": decisions}),
        run_id="run-1",
        auth=_api_key_auth,
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
    from backend.data import human_review

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
