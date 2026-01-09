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
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
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
                owner_user_id=test_user_id,
                name="Test Agent 1",
                description="Test Description 1",
                image_url=None,
                creator_name="Test Creator",
                creator_image_url="",
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "object", "properties": {}},
                credentials_input_schema={"type": "object", "properties": {}},
                has_external_trigger=False,
                status=library_model.LibraryAgentStatus.COMPLETED,
                recommended_schedule_cron=None,
                new_output=False,
                can_access_graph=True,
                is_latest_version=True,
                is_favorite=False,
                created_at=datetime.datetime(2023, 1, 1, 0, 0, 0),
                updated_at=datetime.datetime(2023, 1, 1, 0, 0, 0),
            ),
            library_model.LibraryAgent(
                id="test-agent-2",
                graph_id="test-agent-2",
                graph_version=1,
                owner_user_id=test_user_id,
                name="Test Agent 2",
                description="Test Description 2",
                image_url=None,
                creator_name="Test Creator",
                creator_image_url="",
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "object", "properties": {}},
                credentials_input_schema={"type": "object", "properties": {}},
                has_external_trigger=False,
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
    )


def test_get_library_agents_error(mocker: pytest_mock.MockFixture, test_user_id: str):
    mock_db_call = mocker.patch("backend.api.features.library.db.list_library_agents")
    mock_db_call.side_effect = Exception("Test error")

    response = client.get("/agents?search_term=test")
    assert response.status_code == 500
    mock_db_call.assert_called_once_with(
        user_id=test_user_id,
        search_term="test",
        sort_by=library_model.LibraryAgentSort.UPDATED_AT,
        page=1,
        page_size=15,
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
                owner_user_id=test_user_id,
                name="Favorite Agent 1",
                description="Test Favorite Description 1",
                image_url=None,
                creator_name="Test Creator",
                creator_image_url="",
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "object", "properties": {}},
                credentials_input_schema={"type": "object", "properties": {}},
                has_external_trigger=False,
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


def test_get_favorite_library_agents_error(
    mocker: pytest_mock.MockFixture, test_user_id: str
):
    mock_db_call = mocker.patch(
        "backend.api.features.library.db.list_favorite_library_agents"
    )
    mock_db_call.side_effect = Exception("Test error")

    response = client.get("/agents/favorites")
    assert response.status_code == 500
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
        owner_user_id=test_user_id,
        name="Test Agent 1",
        description="Test Description 1",
        image_url=None,
        creator_name="Test Creator",
        creator_image_url="",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        credentials_input_schema={"type": "object", "properties": {}},
        has_external_trigger=False,
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


def test_add_agent_to_library_error(mocker: pytest_mock.MockFixture, test_user_id: str):
    mock_db_call = mocker.patch(
        "backend.api.features.library.db.add_store_agent_to_library"
    )
    mock_db_call.side_effect = Exception("Test error")

    response = client.post(
        "/agents", json={"store_listing_version_id": "test-version-id"}
    )
    assert response.status_code == 500
    assert "detail" in response.json()  # Verify error response structure
    mock_db_call.assert_called_once_with(
        store_listing_version_id="test-version-id", user_id=test_user_id
    )
