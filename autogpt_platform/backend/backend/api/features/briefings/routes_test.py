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


async def _delete_briefings(user_id: str) -> None:
    """Delete the user's briefings, retrying once on a stale event loop.

    Fire-and-forget background tasks in earlier tests can leave the Prisma
    pool holding a connection bound to a since-closed function-scoped loop;
    the first session-loop call after that raises ``RuntimeError: Event loop
    is closed`` and the pool re-establishes itself on the retry. Same
    workaround as ``backend/conftest.py::_create_user_with_loop_retry``.
    """
    try:
        await UserBriefing.prisma().delete_many(where={"userId": user_id})
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise
        await UserBriefing.prisma().delete_many(where={"userId": user_id})


@pytest_asyncio.fixture(loop_scope="session", autouse=True)
async def _clean_briefings(server, test_user_id: str):
    """Ensure no leftover briefings for the shared test user before/after each test.

    loop_scope must match the session-scoped Prisma client — a
    function-scoped loop here hits "Event loop is closed" on the first
    delete_many once another test has already bound the client's HTTP
    session to a since-closed loop."""
    await _delete_briefings(test_user_id)
    yield
    await _delete_briefings(test_user_id)


def _today() -> datetime.date:
    """UTC today — the route's own reference point for the recency bound."""
    return datetime.datetime.now(datetime.timezone.utc).date()


def _days_ago(days: int) -> datetime.date:
    return _today() - datetime.timedelta(days=days)


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
    today = _today()
    record = await briefing_db.create_briefing(setup_test_user, today, content)

    response = await client.get("/api/briefings/latest")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == record.id
    assert data["briefing_date"] == today.isoformat()
    assert data["delivered_at"] is None
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
    await briefing_db.create_briefing(setup_test_user, _days_ago(1), _valid_content())
    latest = await briefing_db.create_briefing(
        setup_test_user, _today(), _valid_content()
    )

    response = await client.get("/api/briefings/latest")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == latest.id
    assert data["briefing_date"] == _today().isoformat()


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_briefing_returns_null_on_invalid_stored_content(
    client: httpx.AsyncClient,
    setup_test_user: str,
) -> None:
    """Content that no longer validates against BriefingContent is treated as
    if there were no briefing, rather than 500ing the request."""
    await briefing_db.create_briefing(
        setup_test_user, _today(), {"unexpected": "shape"}
    )

    response = await client.get("/api/briefings/latest")

    assert response.status_code == 200
    assert response.json() is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_briefing_falls_back_to_the_newest_readable_briefing(
    client: httpx.AsyncClient,
    setup_test_user: str,
) -> None:
    """One unreadable row on the newest date must not hide older briefings."""
    readable = await briefing_db.create_briefing(
        setup_test_user, _days_ago(1), _valid_content()
    )
    await briefing_db.create_briefing(
        setup_test_user, _today(), {"unexpected": "shape"}
    )

    response = await client.get("/api/briefings/latest")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == readable.id
    assert data["briefing_date"] == _days_ago(1).isoformat()


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_briefing_ignores_stale_briefings(
    client: httpx.AsyncClient,
    setup_test_user: str,
) -> None:
    """A briefing older than yesterday is history, not "this morning" — the
    home card must not present weeks-old runs under a year-less date label."""
    await briefing_db.create_briefing(setup_test_user, _days_ago(21), _valid_content())

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
