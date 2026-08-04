import datetime
import json
from unittest.mock import AsyncMock

import fastapi.testclient
import pytest
import pytest_mock
from pytest_snapshot.plugin import Snapshot

from backend.util.models import Pagination

from . import model as library_model
from .routes import router as library_router

app = fastapi.FastAPI()
app.include_router(library_router)

client = fastapi.testclient.TestClient(app)

FIXED_NOW = datetime.datetime(2023, 1, 1, 0, 0, 0)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module"""
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_library_agents_success(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    mocked_value = library_model.LibraryAgentResponse(
        agents=[
            library_model.LibraryAgent(
                id="test-agent-1",
                graph_id="test-agent-1",
                graph_version=1,
                name="Test Agent 1",
                description="Test Description 1",
                image_url=None,
                creator_name="Test Creator",
                creator_image_url="",
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "object", "properties": {}},
                credentials_input_schema={"type": "object", "properties": {}},
                has_external_trigger=False,
                has_human_in_the_loop=False,
                has_sensitive_action=False,
                status=library_model.LibraryAgentStatus.COMPLETED,
                recommended_schedule_cron=None,
                new_output=False,
                can_access_graph=True,
                is_latest_version=True,
                is_favorite=False,
                organization_id="test-org",
                team_id="test-team",
                created_at=datetime.datetime(2023, 1, 1, 0, 0, 0),
                updated_at=datetime.datetime(2023, 1, 1, 0, 0, 0),
            ),
            library_model.LibraryAgent(
                id="test-agent-2",
                graph_id="test-agent-2",
                graph_version=1,
                name="Test Agent 2",
                description="Test Description 2",
                image_url=None,
                creator_name="Test Creator",
                creator_image_url="",
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "object", "properties": {}},
                credentials_input_schema={"type": "object", "properties": {}},
                has_external_trigger=False,
                has_human_in_the_loop=False,
                has_sensitive_action=False,
                status=library_model.LibraryAgentStatus.COMPLETED,
                recommended_schedule_cron=None,
                new_output=False,
                can_access_graph=False,
                is_latest_version=True,
                is_favorite=False,
                created_at=datetime.datetime(2023, 1, 1, 0, 0, 0),
                updated_at=datetime.datetime(2023, 1, 1, 0, 0, 0),
            ),
        ],
        pagination=Pagination(
            total_items=2, total_pages=1, current_page=1, page_size=50
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.library.db.list_library_agents")
    mock_db_call.return_value = mocked_value

    response = client.get("/agents?search_term=test")
    assert response.status_code == 200

    data = library_model.LibraryAgentResponse.model_validate(response.json())
    assert len(data.agents) == 2
    assert data.agents[0].graph_id == "test-agent-1"
    assert data.agents[0].can_access_graph is True
    # Team tenancy surfaced for list badges/filters (SECRT-2488).
    assert data.agents[0].team_id == "test-team"
    assert data.agents[0].organization_id == "test-org"
    assert data.agents[1].graph_id == "test-agent-2"
    assert data.agents[1].can_access_graph is False

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "lib_agts_search")

    mock_db_call.assert_called_once_with(
        user_id=test_user_id,
        search_term="test",
        sort_by=library_model.LibraryAgentSort.UPDATED_AT,
        page=1,
        page_size=15,
        folder_id=None,
        include_root_only=False,
        is_hidden=None,
        organization_id="test-org",
    )


@pytest.mark.asyncio
async def test_get_favorite_library_agents_success(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    mocked_value = library_model.LibraryAgentResponse(
        agents=[
            library_model.LibraryAgent(
                id="test-agent-1",
                graph_id="test-agent-1",
                graph_version=1,
                name="Favorite Agent 1",
                description="Test Favorite Description 1",
                image_url=None,
                creator_name="Test Creator",
                creator_image_url="",
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "object", "properties": {}},
                credentials_input_schema={"type": "object", "properties": {}},
                has_external_trigger=False,
                has_human_in_the_loop=False,
                has_sensitive_action=False,
                status=library_model.LibraryAgentStatus.COMPLETED,
                recommended_schedule_cron=None,
                new_output=False,
                can_access_graph=True,
                is_latest_version=True,
                is_favorite=True,
                created_at=datetime.datetime(2023, 1, 1, 0, 0, 0),
                updated_at=datetime.datetime(2023, 1, 1, 0, 0, 0),
            ),
        ],
        pagination=Pagination(
            total_items=1, total_pages=1, current_page=1, page_size=15
        ),
    )
    mock_db_call = mocker.patch(
        "backend.api.features.library.db.list_favorite_library_agents"
    )
    mock_db_call.return_value = mocked_value

    response = client.get("/agents/favorites")
    assert response.status_code == 200

    data = library_model.LibraryAgentResponse.model_validate(response.json())
    assert len(data.agents) == 1
    assert data.agents[0].is_favorite is True
    assert data.agents[0].name == "Favorite Agent 1"

    mock_db_call.assert_called_once_with(
        user_id=test_user_id,
        page=1,
        page_size=15,
    )


def test_add_agent_to_library_success(
    mocker: pytest_mock.MockFixture, test_user_id: str
):
    mock_library_agent = library_model.LibraryAgent(
        id="test-library-agent-id",
        graph_id="test-agent-1",
        graph_version=1,
        name="Test Agent 1",
        description="Test Description 1",
        image_url=None,
        creator_name="Test Creator",
        creator_image_url="",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        credentials_input_schema={"type": "object", "properties": {}},
        has_external_trigger=False,
        has_human_in_the_loop=False,
        has_sensitive_action=False,
        status=library_model.LibraryAgentStatus.COMPLETED,
        new_output=False,
        can_access_graph=True,
        is_latest_version=True,
        is_favorite=False,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )

    mock_db_call = mocker.patch(
        "backend.api.features.library.db.add_store_agent_to_library"
    )
    mock_db_call.return_value = mock_library_agent
    mock_complete_onboarding = mocker.patch(
        "backend.api.features.library.routes.agents.complete_onboarding_step",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/agents", json={"store_listing_version_id": "test-version-id"}
    )
    assert response.status_code == 201

    # Verify the response contains the library agent data
    data = library_model.LibraryAgent.model_validate(response.json())
    assert data.id == "test-library-agent-id"
    assert data.graph_id == "test-agent-1"

    mock_db_call.assert_called_once_with(
        store_listing_version_id="test-version-id", user_id=test_user_id
    )
    mock_complete_onboarding.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_trigger_agents_route(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """GET /agents/{id}/triggers delegates to db.list_trigger_agents
    and returns the serialized LibraryAgent list."""
    trigger_agent = library_model.LibraryAgent(
        id="trigger-1",
        graph_id="trigger-graph",
        graph_version=1,
        name="Email Watcher",
        description="Watches the inbox",
        image_url=None,
        creator_name="",
        creator_image_url="",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        credentials_input_schema={"type": "object", "properties": {}},
        has_external_trigger=False,
        has_human_in_the_loop=False,
        has_sensitive_action=False,
        status=library_model.LibraryAgentStatus.COMPLETED,
        new_output=False,
        can_access_graph=True,
        is_latest_version=True,
        is_favorite=False,
        is_hidden=True,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )
    mock_db_call = mocker.patch("backend.api.features.library.db.list_trigger_agents")
    mock_db_call.return_value = [trigger_agent]

    response = client.get("/agents/parent-id/triggers")
    assert response.status_code == 200

    data = [library_model.LibraryAgent.model_validate(a) for a in response.json()]
    assert len(data) == 1
    assert data[0].id == "trigger-1"
    assert data[0].is_hidden is True

    mock_db_call.assert_called_once_with(
        user_id=test_user_id,
        library_agent_id="parent-id",
    )


# ---------------------------------------------------------------------------
# Fork + folder create honor the caller's active team context (SECRT-2488)
#
# mock_jwt_user's ctx has org_id="test-org", team_id="test-team". Both create
# paths must thread that tenancy to the db layer so an explicit X-Team-Id
# lands the fork / new folder in that team instead of the default team.
# ---------------------------------------------------------------------------
def _library_agent(agent_id: str = "forked-1") -> library_model.LibraryAgent:
    return library_model.LibraryAgent(
        id=agent_id,
        graph_id="g-forked",
        graph_version=1,
        name="Forked Agent",
        description="A fork",
        image_url=None,
        creator_name="",
        creator_image_url="",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        credentials_input_schema={"type": "object", "properties": {}},
        has_external_trigger=False,
        has_human_in_the_loop=False,
        has_sensitive_action=False,
        status=library_model.LibraryAgentStatus.COMPLETED,
        new_output=False,
        can_access_graph=True,
        is_latest_version=True,
        is_favorite=False,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )


@pytest.mark.asyncio
async def test_fork_library_agent_threads_ctx_tenancy(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Forking passes the caller's active org/team (X-Team-Id) to the db."""
    fork_mock = mocker.patch(
        "backend.api.features.library.db.fork_library_agent",
        new=AsyncMock(return_value=_library_agent()),
    )

    response = client.post("/agents/lib-agent-1/fork")

    assert response.status_code == 200
    assert fork_mock.await_args.kwargs["organization_id"] == "test-org"
    assert fork_mock.await_args.kwargs["team_id"] == "test-team"


@pytest.mark.asyncio
async def test_create_folder_threads_ctx_tenancy(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """Creating a folder passes the caller's active org/team to the db, and
    the response surfaces the team_id field for badges."""
    folder = library_model.LibraryFolder(
        id="folder-1",
        user_id=test_user_id,
        name="My Folder",
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
        organization_id="test-org",
        team_id="test-team",
    )
    create_mock = mocker.patch(
        "backend.api.features.library.db.create_folder",
        new=AsyncMock(return_value=folder),
    )

    response = client.post("/folders", json={"name": "My Folder"})

    assert response.status_code == 201
    assert create_mock.await_args.kwargs["organization_id"] == "test-org"
    assert create_mock.await_args.kwargs["team_id"] == "test-team"
    assert response.json()["team_id"] == "test-team"
