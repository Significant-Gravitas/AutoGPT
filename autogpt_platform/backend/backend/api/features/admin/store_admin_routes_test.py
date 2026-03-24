"""Tests for admin store routes and the bypass logic they depend on.

Tests are organized by what they protect:
- SECRT-2162: get_graph_as_admin bypasses ownership/marketplace checks
- SECRT-2167 security: admin endpoints reject non-admin users
- SECRT-2167 bypass: preview queries StoreListingVersion (not StoreAgent view),
  and add-to-library uses get_graph_as_admin (not get_graph)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.responses
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.data.graph import get_graph_as_admin
from backend.util.exceptions import NotFoundError

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


# ---- SECRT-2162: get_graph_as_admin bypasses ownership checks ---- #


@pytest.mark.asyncio
async def test_admin_can_access_pending_agent_not_owned() -> None:
    """get_graph_as_admin must return a graph even when the admin doesn't own
    it and it's not APPROVED in the marketplace."""
    mock_graph = _make_mock_graph()
    mock_graph_model = MagicMock(name="GraphModel")

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_prisma,
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

    assert result is mock_graph_model


@pytest.mark.asyncio
async def test_admin_download_pending_agent_with_subagents() -> None:
    """get_graph_as_admin with for_export=True must call get_sub_graphs
    and pass sub_graphs to GraphModel.from_db."""
    mock_graph = _make_mock_graph()
    mock_sub_graph = MagicMock(name="SubGraph")
    mock_graph_model = MagicMock(name="GraphModel")

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_prisma,
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

    assert result is mock_graph_model
    mock_get_sub.assert_awaited_once_with(mock_graph)
    mock_from_db.assert_called_once_with(
        graph=mock_graph,
        sub_graphs=[mock_sub_graph],
        for_export=True,
    )


# ---- SECRT-2167 security: admin endpoints reject non-admin users ---- #

app = fastapi.FastAPI()
app.include_router(store_admin_router)


@app.exception_handler(NotFoundError)
async def _not_found_handler(
    request: fastapi.Request, exc: NotFoundError
) -> fastapi.responses.JSONResponse:
    return fastapi.responses.JSONResponse(status_code=404, content={"detail": str(exc)})


client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    """Setup admin auth overrides for all route tests in this module."""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


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
    """Preview of a nonexistent submission returns 404."""
    mocker.patch(
        "backend.api.features.admin.store_admin_routes.store_db"
        ".get_store_agent_details_as_admin",
        side_effect=NotFoundError("not found"),
    )
    response = client.get(f"/admin/submissions/{SLV_ID}/preview")
    assert response.status_code == 404


# ---- SECRT-2167 bypass: verify the right data sources are used ---- #


@pytest.mark.asyncio
async def test_preview_queries_store_listing_version_not_store_agent() -> None:
    """get_store_agent_details_as_admin must query StoreListingVersion
    directly (not the APPROVED-only StoreAgent view). This is THE test that
    prevents the bypass from being accidentally reverted."""
    from backend.api.features.store.db import get_store_agent_details_as_admin

    mock_slv = MagicMock()
    mock_slv.id = SLV_ID
    mock_slv.name = "Test Agent"
    mock_slv.subHeading = "Short desc"
    mock_slv.description = "Long desc"
    mock_slv.videoUrl = None
    mock_slv.agentOutputDemoUrl = None
    mock_slv.imageUrls = ["https://example.com/img.png"]
    mock_slv.instructions = None
    mock_slv.categories = ["productivity"]
    mock_slv.version = 1
    mock_slv.agentGraphId = GRAPH_ID
    mock_slv.agentGraphVersion = GRAPH_VERSION
    mock_slv.updatedAt = datetime(2026, 3, 24, tzinfo=timezone.utc)
    mock_slv.recommendedScheduleCron = "0 9 * * *"

    mock_listing = MagicMock()
    mock_listing.id = "listing-id"
    mock_listing.slug = "test-agent"
    mock_listing.activeVersionId = SLV_ID
    mock_listing.hasApprovedVersion = False
    mock_listing.CreatorProfile = MagicMock(username="creator", avatarUrl="")
    mock_slv.StoreListing = mock_listing

    with (
        patch(
            "backend.api.features.store.db.prisma.models" ".StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch(
            "backend.api.features.store.db.prisma.models.StoreAgent.prisma",
        ) as mock_store_agent_prisma,
    ):
        mock_slv_prisma.return_value.find_unique = AsyncMock(return_value=mock_slv)

        result = await get_store_agent_details_as_admin(SLV_ID)

    # Verify it queried StoreListingVersion (not the APPROVED-only StoreAgent)
    mock_slv_prisma.return_value.find_unique.assert_awaited_once()
    await_args = mock_slv_prisma.return_value.find_unique.await_args
    assert await_args is not None
    assert await_args.kwargs["where"] == {"id": SLV_ID}

    # Verify the APPROVED-only StoreAgent view was NOT touched
    mock_store_agent_prisma.assert_not_called()

    # Verify the result has the right data
    assert result.agent_name == "Test Agent"
    assert result.agent_image == ["https://example.com/img.png"]
    assert result.has_approved_version is False
    assert result.runs == 0
    assert result.rating == 0.0


@pytest.mark.asyncio
async def test_resolve_graph_admin_uses_get_graph_as_admin() -> None:
    """resolve_graph_for_library(admin=True) must call get_graph_as_admin,
    not get_graph. This is THE test that prevents the add-to-library bypass
    from being accidentally reverted."""
    from backend.api.features.library._add_to_library import resolve_graph_for_library

    mock_slv = MagicMock()
    mock_slv.AgentGraph = MagicMock(id=GRAPH_ID, version=GRAPH_VERSION)
    mock_graph_model = MagicMock(name="GraphModel")

    with (
        patch(
            "backend.api.features.library._add_to_library.prisma.models"
            ".StoreListingVersion.prisma",
        ) as mock_prisma,
        patch(
            "backend.api.features.library._add_to_library.graph_db"
            ".get_graph_as_admin",
            new_callable=AsyncMock,
            return_value=mock_graph_model,
        ) as mock_admin,
        patch(
            "backend.api.features.library._add_to_library.graph_db.get_graph",
            new_callable=AsyncMock,
        ) as mock_regular,
    ):
        mock_prisma.return_value.find_unique = AsyncMock(return_value=mock_slv)

        result = await resolve_graph_for_library(SLV_ID, ADMIN_USER_ID, admin=True)

    assert result is mock_graph_model
    mock_admin.assert_awaited_once_with(
        graph_id=GRAPH_ID, version=GRAPH_VERSION, user_id=ADMIN_USER_ID
    )
    mock_regular.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_graph_regular_uses_get_graph() -> None:
    """resolve_graph_for_library(admin=False) must call get_graph,
    not get_graph_as_admin. Ensures the non-admin path is preserved."""
    from backend.api.features.library._add_to_library import resolve_graph_for_library

    mock_slv = MagicMock()
    mock_slv.AgentGraph = MagicMock(id=GRAPH_ID, version=GRAPH_VERSION)
    mock_graph_model = MagicMock(name="GraphModel")

    with (
        patch(
            "backend.api.features.library._add_to_library.prisma.models"
            ".StoreListingVersion.prisma",
        ) as mock_prisma,
        patch(
            "backend.api.features.library._add_to_library.graph_db"
            ".get_graph_as_admin",
            new_callable=AsyncMock,
        ) as mock_admin,
        patch(
            "backend.api.features.library._add_to_library.graph_db.get_graph",
            new_callable=AsyncMock,
            return_value=mock_graph_model,
        ) as mock_regular,
    ):
        mock_prisma.return_value.find_unique = AsyncMock(return_value=mock_slv)

        result = await resolve_graph_for_library(SLV_ID, "regular-user-id", admin=False)

    assert result is mock_graph_model
    mock_regular.assert_awaited_once_with(
        graph_id=GRAPH_ID, version=GRAPH_VERSION, user_id="regular-user-id"
    )
    mock_admin.assert_not_awaited()


# ---- Library membership grants graph access (product decision) ---- #


@pytest.mark.asyncio
async def test_library_member_can_view_pending_agent_in_builder() -> None:
    """After adding a pending agent to their library, the user should be
    able to load the graph in the builder via get_graph()."""
    mock_graph = _make_mock_graph()
    mock_graph_model = MagicMock(name="GraphModel")
    mock_library_agent = MagicMock()
    mock_library_agent.AgentGraph = mock_graph

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.GraphModel.from_db",
            return_value=mock_graph_model,
        ),
    ):
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_lib_prisma.return_value.find_first = AsyncMock(
            return_value=mock_library_agent
        )

        from backend.data.graph import get_graph

        result = await get_graph(
            graph_id=GRAPH_ID,
            version=GRAPH_VERSION,
            user_id=ADMIN_USER_ID,
        )

    assert result is mock_graph_model, "Library membership should grant graph access"
