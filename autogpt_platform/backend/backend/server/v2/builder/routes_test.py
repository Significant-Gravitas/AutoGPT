import json

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from pytest_snapshot.plugin import Snapshot

import backend.server.v2.builder.model as builder_model
from backend.server.v2.builder.routes import router as builder_router

app = fastapi.FastAPI()
app.include_router(builder_router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_get_block_categories(mocker: pytest_mock.MockFixture, snapshot: Snapshot):
    mocked_value = [
        builder_model.BlockCategoryResponse(
            name="AI",
            total_blocks=2,
            blocks=[
                {"id": "block1", "name": "Block 1"},
                {"id": "block2", "name": "Block 2"},
            ],
        )
    ]
    mock_db_call = mocker.patch("backend.server.v2.builder.db.get_block_categories")
    mock_db_call.return_value = mocked_value
    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(data, indent=2), "builder_block_categories")
    mock_db_call.assert_called_once()


def test_get_blocks(mocker: pytest_mock.MockFixture, snapshot: Snapshot):
    mocked_value = builder_model.BlockResponse(
        blocks=[{"id": "block1", "name": "Block 1"}],
        pagination=builder_model.Pagination(
            total_items=1, total_pages=1, current_page=1, page_size=50
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.builder.db.get_blocks")
    mock_db_call.return_value = mocked_value
    response = client.get("/blocks?category=AI")
    assert response.status_code == 200
    data = response.json()
    assert "blocks" in data
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(data, indent=2), "builder_blocks")
    mock_db_call.assert_called_once_with(
        category="AI", type=None, provider=None, page=1, page_size=50
    )


def test_get_providers(mocker: pytest_mock.MockFixture, snapshot: Snapshot):
    from backend.integrations.providers import ProviderName

    mocked_value = builder_model.ProviderResponse(
        providers=[
            builder_model.Provider(
                name=ProviderName.GITHUB,
                description="GitHub integration",
                integration_count=2,
            )
        ],
        pagination=builder_model.Pagination(
            total_items=1, total_pages=1, current_page=1, page_size=50
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.builder.db.get_providers")
    mock_db_call.return_value = mocked_value
    response = client.get("/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(data, indent=2), "builder_providers")
    mock_db_call.assert_called_once_with(page=1, page_size=50)


@pytest.mark.asyncio
async def test_get_counts(mocker: pytest_mock.MockFixture):
    mock_db_call = mocker.patch("backend.server.v2.builder.db.get_counts")
    mock_db_call.return_value = builder_model.CountResponse(
        all_blocks=10,
        input_blocks=2,
        action_blocks=5,
        output_blocks=3,
        integrations=4,
        marketplace_agents=7,
        my_agents=1,
    )
    response = client.get("/counts")
    assert response.status_code == 200
    data = response.json()
    assert data["all_blocks"] == 10
    mock_db_call.assert_called_once()


@pytest.mark.asyncio
async def test_search(mocker: pytest_mock.MockFixture, snapshot: Snapshot):
    mock_blocks = builder_model.SearchBlocksResponse(
        blocks=builder_model.BlockResponse(
            blocks=[{"id": "block1", "name": "Block 1"}],
            pagination=builder_model.Pagination(
                total_items=1, total_pages=1, current_page=1, page_size=50
            ),
        ),
        total_block_count=1,
        total_integration_count=0,
    )
    mocker.patch("backend.server.v2.builder.db.search_blocks", return_value=mock_blocks)
    mocker.patch(
        "backend.server.v2.library.db.list_library_agents",
        return_value=builder_model.library_model.LibraryAgentResponse(
            agents=[], pagination=builder_model.Pagination.empty()
        ),
    )
    mocker.patch(
        "backend.server.v2.store.db.get_store_agents",
        return_value=builder_model.store_model.StoreAgentsResponse(
            agents=[], pagination=builder_model.Pagination.empty()
        ),
    )
    response = client.post("/search", json={"search_query": "test"})
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(data, indent=2), "builder_search")


@pytest.mark.asyncio
async def test_get_suggestions(mocker: pytest_mock.MockFixture, snapshot: Snapshot):
    mock_db_call = mocker.patch("backend.server.v2.builder.db.get_suggested_blocks")
    mock_db_call.return_value = [{"id": "block1", "name": "Block 1"}]
    response = client.get("/suggestions")
    assert response.status_code == 200
    data = response.json()
    assert "otto_suggestions" in data
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(data, indent=2), "builder_suggestions")
    mock_db_call.assert_called_once()
