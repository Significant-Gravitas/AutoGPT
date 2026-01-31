from types import SimpleNamespace

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_user_id, requires_user

import backend.api.features.builder.model as builder_model
import backend.api.features.builder.routes as builder_routes
from backend.api.features.builder.db import _SearchCacheEntry
from backend.api.test_helpers import assert_response_status
from backend.data.block import BlockInfo
from backend.util.models import Pagination


@pytest.fixture(scope="session")
async def server():
    class _DummyAgentServer:
        async def test_create_graph(self, *args, **kwargs):
            return SimpleNamespace(id="dummy-graph")

        async def test_delete_graph(self, *args, **kwargs):
            return {"version_counts": 1}

    class _DummyServer:
        agent_server = _DummyAgentServer()

    yield _DummyServer()


@pytest.fixture(scope="session", autouse=True)
async def graph_cleanup(server):
    yield


app = fastapi.FastAPI()
app.include_router(builder_routes.router)

client = fastapi.testclient.TestClient(app)


def _make_block_info(block_id: str, name: str) -> BlockInfo:
    return BlockInfo(
        id=block_id,
        name=name,
        inputSchema={},
        outputSchema={},
        costs=[],
        description=f"{name} description",
        categories=[],
        contributors=[],
        staticOutput=False,
        uiType="default",
    )


@pytest.fixture(autouse=True)
def setup_auth_overrides(mock_jwt_user):
    app.dependency_overrides[get_user_id] = lambda: mock_jwt_user["user_id"]
    app.dependency_overrides[requires_user] = lambda: None
    yield
    app.dependency_overrides.clear()


def test_get_suggestions_returns_expected_payload(
    mocker: pytest_mock.MockFixture,
) -> None:
    mock_recent = [
        builder_model.SearchEntry(
            search_query="alpha", filter=["blocks"], search_id="1"
        )
    ]
    mock_blocks = [_make_block_info("block-1", "Alpha Block")]

    mocker.patch(
        "backend.server.v2.builder.routes.builder_db.get_recent_searches",
        new_callable=mocker.AsyncMock,
        return_value=mock_recent,
    )
    mocker.patch(
        "backend.server.v2.builder.routes.builder_db.get_suggested_blocks",
        new_callable=mocker.AsyncMock,
        return_value=mock_blocks,
    )

    response = client.get("/suggestions")
    assert_response_status(response, 200)
    data = builder_model.SuggestionsResponse.model_validate(response.json())

    assert data.recent_searches[0].search_query == "alpha"
    assert data.top_blocks[0].id == "block-1"


def test_get_block_categories_forwards_limit(
    mocker: pytest_mock.MockFixture,
) -> None:
    category_response = [
        builder_model.BlockCategoryResponse(
            name="ai",
            total_blocks=1,
            blocks=[_make_block_info("block-1", "Alpha")],
        )
    ]
    mock_get_categories = mocker.patch(
        "backend.server.v2.builder.routes.builder_db.get_block_categories",
        return_value=category_response,
    )

    response = client.get("/categories?blocks_per_category=2")
    assert_response_status(response, 200)
    data = response.json()
    assert data[0]["name"] == "ai"
    mock_get_categories.assert_called_once_with(2)


def test_get_blocks_forwards_filters(
    mocker: pytest_mock.MockFixture,
) -> None:
    block_response = builder_model.BlockResponse(
        blocks=[_make_block_info("block-1", "Alpha")],
        pagination=Pagination(
            total_items=1,
            total_pages=1,
            current_page=1,
            page_size=50,
        ),
    )
    mock_get_blocks = mocker.patch(
        "backend.server.v2.builder.routes.builder_db.get_blocks",
        return_value=block_response,
    )

    response = client.get("/blocks?type=action&page=2&page_size=10")
    assert_response_status(response, 200)
    assert response.json()["blocks"][0]["id"] == "block-1"
    mock_get_blocks.assert_called_once_with(
        category=None,
        type="action",
        provider=None,
        page=2,
        page_size=10,
    )


def test_get_specific_blocks_filters_missing(
    mocker: pytest_mock.MockFixture,
) -> None:
    block_info = _make_block_info("block-1", "Alpha")
    mock_get_block = mocker.patch(
        "backend.server.v2.builder.routes.builder_db.get_block_by_id",
        side_effect=[block_info, None],
    )

    response = client.get(
        "/blocks/batch",
        params=[("block_ids", "block-1"), ("block_ids", "missing")],
    )
    assert_response_status(response, 200)
    assert response.json() == [block_info.model_dump()]
    assert mock_get_block.call_count == 2


def test_get_providers_forwards_pagination(
    mocker: pytest_mock.MockFixture,
) -> None:
    provider_response = builder_model.ProviderResponse(
        providers=[],
        pagination=Pagination(
            total_items=0,
            total_pages=0,
            current_page=1,
            page_size=50,
        ),
    )
    mock_get_providers = mocker.patch(
        "backend.server.v2.builder.routes.builder_db.get_providers",
        return_value=provider_response,
    )

    response = client.get("/providers?page=3&page_size=5")
    assert_response_status(response, 200)
    mock_get_providers.assert_called_once_with(page=3, page_size=5)


def test_search_applies_defaults_and_sanitizes_query(
    mocker: pytest_mock.MockFixture,
) -> None:
    cache_entry = _SearchCacheEntry(
        items=[_make_block_info("block-1", "Alpha")],
        total_items={
            "blocks": 1,
            "integrations": 0,
            "marketplace_agents": 0,
            "my_agents": 0,
        },
    )
    mock_get_results = mocker.patch(
        "backend.server.v2.builder.routes.builder_db.get_sorted_search_results",
        new_callable=mocker.AsyncMock,
        return_value=cache_entry,
    )
    mock_update_search = mocker.patch(
        "backend.server.v2.builder.routes.builder_db.update_search",
        new_callable=mocker.AsyncMock,
        return_value="search-1",
    )

    response = client.get(
        "/search",
        params={"search_query": "  alpha%  ", "page": 2, "page_size": 1},
    )
    assert_response_status(response, 200)
    data = builder_model.SearchResponse.model_validate(response.json())
    assert data.search_id == "search-1"
    assert data.pagination.current_page == 2
    assert data.pagination.total_items == 1

    mock_get_results.assert_awaited_once()
    await_args = mock_get_results.await_args
    assert await_args is not None
    kwargs = await_args.kwargs
    assert kwargs["filters"] == [
        "blocks",
        "integrations",
        "marketplace_agents",
        "my_agents",
    ]
    assert kwargs["search_query"] == "alpha\\%"
    mock_update_search.assert_awaited_once()


def test_search_forwards_custom_filters_and_creators(
    mocker: pytest_mock.MockFixture,
) -> None:
    cache_entry = _SearchCacheEntry(items=[], total_items={})
    mock_get_results = mocker.patch(
        "backend.server.v2.builder.routes.builder_db.get_sorted_search_results",
        new_callable=mocker.AsyncMock,
        return_value=cache_entry,
    )
    mocker.patch(
        "backend.server.v2.builder.routes.builder_db.update_search",
        new_callable=mocker.AsyncMock,
        return_value="search-2",
    )

    response = client.get(
        "/search",
        params=[
            ("filter", "blocks"),
            ("filter", "my_agents"),
            ("by_creator", "alpha"),
            ("by_creator", "beta"),
        ],
    )
    assert_response_status(response, 200)
    await_args = mock_get_results.await_args
    assert await_args is not None
    kwargs = await_args.kwargs
    assert kwargs["filters"] == ["blocks", "my_agents"]
    assert kwargs["by_creator"] == ["alpha", "beta"]


def test_get_counts_returns_payload(
    mocker: pytest_mock.MockFixture,
) -> None:
    counts = builder_model.CountResponse(
        all_blocks=1,
        input_blocks=1,
        action_blocks=0,
        output_blocks=0,
        integrations=0,
        marketplace_agents=0,
        my_agents=1,
    )
    mock_get_counts = mocker.patch(
        "backend.server.v2.builder.routes.builder_db.get_counts",
        new_callable=mocker.AsyncMock,
        return_value=counts,
    )

    response = client.get("/counts")
    assert_response_status(response, 200)
    assert response.json()["my_agents"] == 1
    mock_get_counts.assert_awaited_once()
