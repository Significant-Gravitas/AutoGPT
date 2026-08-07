"""Tests for the briefings API routes.

Pattern mirrors backend/api/features/executions/review/review_routes_test.py:
the full app from backend.api.rest_api, the session-scoped `server` fixture
(real Prisma DB), and `mock_jwt_user` for auth overrides. Unlike the review
tests, these seed real rows via backend.data.briefing.create_briefing rather
than mocking the data layer, since the behavior under test includes how the
route handles content that fails to validate against BriefingContent.
"""

import datetime
from collections.abc import AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from prisma.models import UserBriefing

from backend.api.rest_api import app
from backend.data import briefing as briefing_db

from .routes import router


@pytest_asyncio.fixture(loop_scope="session")
async def client(server, mock_jwt_user) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create async HTTP client with auth overrides"""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        yield http_client

    app.dependency_overrides.pop(get_jwt_payload, None)


@pytest_asyncio.fixture(autouse=True)
async def _clean_briefings(server, test_user_id: str):
    """Ensure no leftover briefings for the shared test user before/after each test."""
    await UserBriefing.prisma().delete_many(where={"userId": test_user_id})
    yield
    await UserBriefing.prisma().delete_many(where={"userId": test_user_id})


def _valid_content() -> dict:
    return {
        "generated_at": "2026-08-07T09:00:00+00:00",
        "timezone": "UTC",
        "zero_expert_fallback": False,
        "run_items": [
            {
                "expert_id": "expert-1",
                "expert_name": "Maria",
                "expert_avatar_url": None,
                "agent_name": "SEO Blog Writer",
                "graph_id": "graph-1",
                "execution_id": "exec-1",
                "library_agent_id": "library-agent-1",
                "status": "COMPLETED",
                "summary": "Published 3 posts",
                "link": "/library/agents/library-agent-1",
            }
        ],
        "decision_items": [
            {
                "node_exec_id": "node-exec-1",
                "graph_exec_id": "exec-1",
                "title": "Approve email send",
                "expert_id": "expert-1",
                "expert_name": "Maria",
                "expert_avatar_url": None,
                "link": "/review/exec-1",
            }
        ],
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_briefing_returns_null_when_none_exists(
    client: httpx.AsyncClient,
) -> None:
    """No briefing rows for the user -> 200 with a null body."""
    response = await client.get("/api/briefings/latest")

    assert response.status_code == 200
    assert response.json() is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_briefing_returns_typed_content(
    client: httpx.AsyncClient,
    setup_test_user: str,
) -> None:
    """A seeded briefing comes back as typed content matching what was stored."""
    content = _valid_content()
    record = await briefing_db.create_briefing(
        setup_test_user, datetime.date(2026, 8, 7), content
    )

    response = await client.get("/api/briefings/latest")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == record.id
    assert data["briefing_date"] == "2026-08-07"
    assert data["content"]["timezone"] == "UTC"
    assert data["content"]["zero_expert_fallback"] is False
    assert data["content"]["run_items"][0]["expert_name"] == "Maria"
    assert data["content"]["decision_items"][0]["title"] == "Approve email send"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_briefing_returns_only_the_latest_by_date(
    client: httpx.AsyncClient,
    setup_test_user: str,
) -> None:
    """When multiple briefings exist, the one with the latest briefing_date wins."""
    await briefing_db.create_briefing(
        setup_test_user, datetime.date(2026, 8, 5), _valid_content()
    )
    latest = await briefing_db.create_briefing(
        setup_test_user, datetime.date(2026, 8, 7), _valid_content()
    )

    response = await client.get("/api/briefings/latest")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == latest.id
    assert data["briefing_date"] == "2026-08-07"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_briefing_returns_null_on_invalid_stored_content(
    client: httpx.AsyncClient,
    setup_test_user: str,
) -> None:
    """Content that no longer validates against BriefingContent is treated as
    if there were no briefing, rather than 500ing the request."""
    await briefing_db.create_briefing(
        setup_test_user, datetime.date(2026, 8, 7), {"unexpected": "shape"}
    )

    response = await client.get("/api/briefings/latest")

    assert response.status_code == 200
    assert response.json() is None


def test_get_latest_briefing_requires_auth() -> None:
    """GET /briefings/latest should return 401 when no valid JWT is provided."""
    import fastapi
    import fastapi.testclient

    unauthenticated_app = fastapi.FastAPI()
    unauthenticated_app.include_router(router)
    unauthenticated_client = fastapi.testclient.TestClient(unauthenticated_app)

    response = unauthenticated_client.get("/briefings/latest")

    assert response.status_code == 401
