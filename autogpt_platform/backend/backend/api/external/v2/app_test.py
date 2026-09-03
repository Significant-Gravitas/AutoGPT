"""
Tests for v2 API error handling behavior.

The v2 app registers its own exception handlers (since mounted sub-apps don't
inherit handlers from the parent app). These tests verify that exceptions from
the DB/service layer are correctly mapped to HTTP status codes.

We construct a lightweight test app rather than importing the full v2_app,
because the latter eagerly loads the MCP server, block registry, and other
heavy dependencies that are irrelevant for error handling tests.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from prisma.enums import APIKeyPermission
from pytest_snapshot.plugin import Snapshot

from backend.api.external.middleware import require_auth
from backend.api.utils.exceptions import add_exception_handlers
from backend.data.auth.base import APIAuthorizationInfo
from backend.util.exceptions import DatabaseError, NotFoundError

from .library.agents import agents_router
from .marketplace import marketplace_router

TEST_USER_ID = "test-user-id"

_mock_auth = APIAuthorizationInfo(
    user_id=TEST_USER_ID,
    scopes=list(APIKeyPermission),
    type="api_key",
    created_at=datetime.now(tz=timezone.utc),
)

# ---------------------------------------------------------------------------
# Build a lightweight test app with the shared exception handlers
# but only the routers we need for testing.
# ---------------------------------------------------------------------------

app = fastapi.FastAPI()
app.include_router(agents_router, prefix="/library")
app.include_router(marketplace_router, prefix="/marketplace")
add_exception_handlers(app)


@pytest.fixture(autouse=True)
def _override_auth():
    """Bypass API key / OAuth auth for all tests in this module."""

    async def fake_auth() -> APIAuthorizationInfo:
        return _mock_auth

    app.dependency_overrides[require_auth] = fake_auth
    yield
    app.dependency_overrides.clear()


client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)


# ============================================================================
# NotFoundError → 404
# ============================================================================


def test_not_found_error_returns_404(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """NotFoundError raised by the DB layer should become a 404 response."""
    mocker.patch(
        "backend.api.features.library.db.get_library_agent",
        new_callable=AsyncMock,
        side_effect=NotFoundError("Agent #nonexistent not found"),
    )

    response = client.get("/library/agents/nonexistent")

    assert response.status_code == 404
    body = response.json()
    assert body["detail"] == "Agent #nonexistent not found"
    assert "message" in body
    assert body["hint"] == "Adjust the request and retry."

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(body, indent=2, sort_keys=True),
        "v2_not_found_error_404",
    )


def test_not_found_error_on_delete_returns_404(
    mocker: pytest_mock.MockFixture,
) -> None:
    """NotFoundError on DELETE should return 404, not 204 or 500."""
    mocker.patch(
        "backend.api.features.library.db.delete_library_agent",
        new_callable=AsyncMock,
        side_effect=NotFoundError("Agent #gone not found"),
    )

    response = client.delete("/library/agents/gone")

    assert response.status_code == 404
    assert response.json()["detail"] == "Agent #gone not found"
    assert "message" in response.json()


def test_not_found_error_on_marketplace_returns_404(
    mocker: pytest_mock.MockFixture,
) -> None:
    """NotFoundError from store DB layer should become a 404."""
    mocker.patch(
        "backend.api.features.store.db.get_store_agent_by_version_id",
        new_callable=AsyncMock,
        side_effect=NotFoundError("Store listing not found"),
    )

    response = client.get("/marketplace/agents/by-version/nonexistent")

    assert response.status_code == 404
    assert response.json()["detail"] == "Store listing not found"
    assert "message" in response.json()


# ============================================================================
# ValueError → 400
# ============================================================================


def test_value_error_returns_400(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """ValueError raised by the service layer should become a 400 response."""
    mocker.patch(
        "backend.api.features.library.db.update_library_agent",
        new_callable=AsyncMock,
        side_effect=ValueError("Invalid graph version: -1"),
    )

    response = client.patch(
        "/library/agents/some-id",
        json={"graph_version": -1},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "Invalid graph version: -1"
    assert "message" in body
    assert body["hint"] == "Adjust the request and retry."

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(body, indent=2, sort_keys=True),
        "v2_value_error_400",
    )


# ============================================================================
# NotFoundError is a ValueError subclass — verify specificity wins
# ============================================================================


def test_not_found_error_takes_precedence_over_value_error(
    mocker: pytest_mock.MockFixture,
) -> None:
    """
    NotFoundError(ValueError) should match the NotFoundError handler (404),
    not the ValueError handler (400).
    """
    mocker.patch(
        "backend.api.features.library.db.get_library_agent",
        new_callable=AsyncMock,
        side_effect=NotFoundError("Specific not found"),
    )

    response = client.get("/library/agents/test-id")

    # Must be 404, not 400
    assert response.status_code == 404


# ============================================================================
# Unhandled Exception → 500
# ============================================================================


def test_unhandled_exception_returns_500(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """
    Unexpected exceptions should return a generic 500 without leaking
    internal details.
    """
    mocker.patch(
        "backend.api.features.library.db.get_library_agent",
        new_callable=AsyncMock,
        side_effect=DatabaseError("connection refused"),
    )

    response = client.get("/library/agents/some-id")

    assert response.status_code == 500
    body = response.json()
    assert "message" in body
    assert "detail" in body
    assert body["hint"] == "Check server logs and dependent services."

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(body, indent=2, sort_keys=True),
        "v2_unhandled_exception_500",
    )


def test_runtime_error_returns_500(
    mocker: pytest_mock.MockFixture,
) -> None:
    """RuntimeError (not ValueError) should hit the catch-all 500 handler."""
    mocker.patch(
        "backend.api.features.library.db.delete_library_agent",
        new_callable=AsyncMock,
        side_effect=RuntimeError("something broke"),
    )

    response = client.delete("/library/agents/some-id")

    assert response.status_code == 500
    assert "detail" in response.json()
    assert response.json()["hint"] == "Check server logs and dependent services."


# ============================================================================
# Response format consistency
# ============================================================================


def test_all_error_responses_have_consistent_format(
    mocker: pytest_mock.MockFixture,
) -> None:
    """All error responses should use {"message": ..., "detail": ..., "hint": ...} format."""
    cases = [
        (NotFoundError("not found"), 404),
        (ValueError("bad value"), 400),
        (RuntimeError("boom"), 500),
    ]

    for exc, expected_status in cases:
        mocker.patch(
            "backend.api.features.library.db.get_library_agent",
            new_callable=AsyncMock,
            side_effect=exc,
        )

        response = client.get("/library/agents/test-id")

        assert response.status_code == expected_status, (
            f"Expected {expected_status} for {type(exc).__name__}, "
            f"got {response.status_code}"
        )
        body = response.json()
        assert (
            "message" in body
        ), f"Missing 'message' key for {type(exc).__name__}: {body}"
        assert (
            "detail" in body
        ), f"Missing 'detail' key for {type(exc).__name__}: {body}"
        assert "hint" in body, f"Missing 'hint' key for {type(exc).__name__}: {body}"
