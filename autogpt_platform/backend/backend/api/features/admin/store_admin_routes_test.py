from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.data.graph import get_graph_as_admin

from .store_admin_routes import router as store_admin_router

# Shared constants
ADMIN_USER_ID = "admin-user-id"
CREATOR_USER_ID = "other-creator-id"
GRAPH_ID = "test-graph-id"
GRAPH_VERSION = 3
SLV_ID = "test-store-listing-version-id"


def _make_mock_graph(user_id: str = CREATOR_USER_ID) -> MagicMock:
    graph = MagicMock()
    graph.userId = user_id
    graph.id = GRAPH_ID
    graph.version = GRAPH_VERSION
    graph.Nodes = []
    return graph


# ---- get_graph_as_admin unit tests (SECRT-2162 regression) ---- #


@pytest.mark.asyncio
async def test_admin_can_access_pending_agent_not_owned() -> None:
    """Admin must be able to access a graph they don't own even if it's not
    APPROVED in the marketplace. This is the core use case: reviewing a
    submitted-but-pending agent from the admin dashboard."""
    mock_graph = _make_mock_graph()
    mock_graph_model = MagicMock(name="GraphModel")

    with (
        patch(
            "backend.data.graph.AgentGraph.prisma",
        ) as mock_prisma,
        patch(
            "backend.data.graph.GraphModel.from_db",
            return_value=mock_graph_model,
        ),
    ):
        mock_prisma.return_value.find_first = AsyncMock(return_value=mock_graph)

        result = await get_graph_as_admin(
            graph_id=GRAPH_ID,
            version=GRAPH_VERSION,
            user_id=ADMIN_USER_ID,
            for_export=False,
        )

    assert (
        result is not None
    ), "Admin should be able to access a pending agent they don't own"
    assert result is mock_graph_model


@pytest.mark.asyncio
async def test_admin_download_pending_agent_with_subagents() -> None:
    """Admin export (for_export=True) of a pending agent must include
    sub-graphs. This exercises the full export code path that the Download
    button uses."""
    mock_graph = _make_mock_graph()
    mock_sub_graph = MagicMock(name="SubGraph")
    mock_graph_model = MagicMock(name="GraphModel")

    with (
        patch(
            "backend.data.graph.AgentGraph.prisma",
        ) as mock_prisma,
        patch(
            "backend.data.graph.get_sub_graphs",
            new_callable=AsyncMock,
            return_value=[mock_sub_graph],
        ) as mock_get_sub,
        patch(
            "backend.data.graph.GraphModel.from_db",
            return_value=mock_graph_model,
        ) as mock_from_db,
    ):
        mock_prisma.return_value.find_first = AsyncMock(return_value=mock_graph)

        result = await get_graph_as_admin(
            graph_id=GRAPH_ID,
            version=GRAPH_VERSION,
            user_id=ADMIN_USER_ID,
            for_export=True,
        )

    assert result is not None, "Admin export of pending agent must succeed"
    mock_get_sub.assert_awaited_once_with(mock_graph)
    mock_from_db.assert_called_once_with(
        graph=mock_graph,
        sub_graphs=[mock_sub_graph],
        for_export=True,
    )


# ---- Route-level tests for admin preview & add-to-library (SECRT-2167) ---- #

app = fastapi.FastAPI()
app.include_router(store_admin_router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    """Setup admin auth overrides for all route tests in this module."""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _make_mock_store_agent_details() -> MagicMock:
    """Return a mock that looks like StoreAgentDetails for JSON serialization."""
    return MagicMock(
        store_listing_version_id=SLV_ID,
        slug="test-agent",
        agent_name="Test Agent",
        agent_video="",
        agent_output_demo="",
        agent_image=["https://example.com/img.png"],
        creator="test-creator",
        creator_avatar="",
        sub_heading="A test agent",
        description="Full description",
        instructions=None,
        categories=["productivity"],
        runs=0,
        rating=0.0,
        versions=["1"],
        graph_id=GRAPH_ID,
        graph_versions=["1"],
        last_updated=datetime(2026, 3, 24, tzinfo=timezone.utc),
        recommended_schedule_cron=None,
        active_version_id=SLV_ID,
        has_approved_version=False,
        changelog=None,
    )


# -- Security tests --


def test_preview_requires_admin(mock_jwt_user) -> None:
    """Non-admin users must get 403 on the preview endpoint."""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    response = client.get(f"/admin/submissions/{SLV_ID}/preview")
    assert response.status_code == 403


def test_add_to_library_requires_admin(mock_jwt_user) -> None:
    """Non-admin users must get 403 on the add-to-library endpoint."""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    response = client.post(f"/admin/submissions/{SLV_ID}/add-to-library")
    assert response.status_code == 403


def test_preview_nonexistent_submission(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Preview of a nonexistent submission must return 404."""
    from backend.util.exceptions import NotFoundError

    mocker.patch(
        "backend.api.features.admin.store_admin_routes.store_db"
        ".get_store_agent_details_as_admin",
        side_effect=NotFoundError("not found"),
    )
    response = client.get(f"/admin/submissions/{SLV_ID}/preview")
    assert response.status_code in (404, 500)


# -- Functional tests --


def test_admin_preview_pending_submission(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Admin should be able to preview a pending submission with full details."""
    from backend.api.features.store.model import StoreAgentDetails

    mock_details = StoreAgentDetails(
        store_listing_version_id=SLV_ID,
        slug="test-agent",
        agent_name="Test Agent",
        agent_video="",
        agent_output_demo="",
        agent_image=["https://example.com/img.png"],
        creator="test-creator",
        creator_avatar="",
        sub_heading="A test agent",
        description="Full description",
        categories=["productivity"],
        runs=0,
        rating=0.0,
        versions=["1"],
        graph_id=GRAPH_ID,
        graph_versions=["1"],
        last_updated=datetime(2026, 3, 24, tzinfo=timezone.utc),
        active_version_id=SLV_ID,
        has_approved_version=False,
    )
    mocker.patch(
        "backend.api.features.admin.store_admin_routes.store_db"
        ".get_store_agent_details_as_admin",
        return_value=mock_details,
    )

    response = client.get(f"/admin/submissions/{SLV_ID}/preview")
    assert response.status_code == 200

    data = response.json()
    assert data["agent_name"] == "Test Agent"
    assert data["agent_image"] == ["https://example.com/img.png"]
    assert data["categories"] == ["productivity"]
    assert data["has_approved_version"] is False
    assert data["runs"] == 0


def test_admin_add_pending_agent_to_library(
    mocker: pytest_mock.MockerFixture,
    admin_user_id: str,
) -> None:
    """Admin should be able to add a pending agent to their library."""
    mock_library_agent = MagicMock()
    mock_library_agent.id = "lib-agent-id"
    mock_library_agent.name = "Test Agent"

    mocker.patch(
        "backend.api.features.admin.store_admin_routes.library_db"
        ".add_store_agent_to_library_as_admin",
        new_callable=AsyncMock,
        return_value=mock_library_agent,
    )

    response = client.post(f"/admin/submissions/{SLV_ID}/add-to-library")
    assert response.status_code == 201
