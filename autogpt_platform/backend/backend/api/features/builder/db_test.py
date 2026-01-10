import datetime
from types import SimpleNamespace

import pytest
import pytest_mock

import backend.api.features.builder.db as builder_db
import backend.api.features.builder.model as builder_model
import backend.api.features.library.model as library_model
import backend.api.features.store.model as store_model
from backend.blocks.llm import LlmModel
from backend.data.block import BlockCategory, BlockCost, BlockCostType, BlockInfo
from backend.data.block import BlockType as DataBlockType
from backend.integrations.providers import ProviderName
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


def _make_block_info(block_id: str, name: str, *, description: str) -> BlockInfo:
    return BlockInfo(
        id=block_id,
        name=name,
        description=description,
        inputSchema={},
        outputSchema={},
        costs=[BlockCost(cost_amount=1, cost_type=BlockCostType.RUN)],
        categories=[],
        contributors=[],
        staticOutput=False,
        uiType="default",
    )


def _make_input_schema(
    *,
    has_credentials: bool = False,
    providers: list[ProviderName] | None = None,
    include_llm_field: bool = False,
):
    providers = providers or []
    credentials_info_data = (
        {"token": SimpleNamespace(provider=providers)} if has_credentials else {}
    )
    credentials_fields_data = {"token": object()} if has_credentials else {}
    model_fields_data = (
        {"model": SimpleNamespace(annotation=LlmModel)}
        if include_llm_field
        else {"text": SimpleNamespace(annotation=str)}
    )

    class _InputSchema:
        _credentials_info: dict[str, SimpleNamespace] = credentials_info_data
        _credentials_fields: dict[str, object] = credentials_fields_data
        model_fields: dict[str, SimpleNamespace] = model_fields_data

        @classmethod
        def get_credentials_fields_info(cls):
            return cls._credentials_info

        @classmethod
        def get_credentials_fields(cls):
            return cls._credentials_fields

    return _InputSchema


def _make_block_factory(
    *,
    block_id: str,
    name: str,
    categories: list[BlockCategory],
    block_type: DataBlockType,
    description: str = "Block description",
    disabled: bool = False,
    has_credentials: bool = False,
    providers: list[ProviderName] | None = None,
    include_llm_field: bool = False,
):
    block_info = _make_block_info(block_id, name, description=description)
    block_info.categories = [category.dict() for category in categories]
    input_schema = _make_input_schema(
        has_credentials=has_credentials,
        providers=providers,
        include_llm_field=include_llm_field,
    )

    class _Block:
        def __init__(self):
            self.id = block_id
            self.block_type = block_type
            self.description = description
            self.categories = categories
            self.disabled = disabled
            self.input_schema = input_schema
            self.output_schema = input_schema

        def get_info(self) -> BlockInfo:
            return block_info

    return _Block


def _make_library_agent(name: str) -> library_model.LibraryAgent:
    return library_model.LibraryAgent(
        id=f"{name}-id",
        graph_id="graph-id",
        graph_version=1,
        image_url=None,
        creator_name="Creator",
        creator_image_url="",
        status=library_model.LibraryAgentStatus.COMPLETED,
        created_at=FIXED_TIME,
        updated_at=FIXED_TIME,
        name=name,
        description=f"{name} description",
        instructions="Do things",
        input_schema={},
        output_schema={},
        credentials_input_schema={},
        has_external_trigger=False,
        trigger_setup_info=None,
        new_output=False,
        can_access_graph=True,
        is_latest_version=True,
        is_favorite=False,
    )


def _make_store_agent(name: str) -> store_model.StoreAgent:
    return store_model.StoreAgent(
        slug=f"{name}-slug",
        agent_name=name,
        agent_image="image.png",
        creator="creator-a",
        creator_avatar="avatar.png",
        sub_heading=f"{name} subheading",
        description=f"{name} description",
        runs=1,
        rating=4.5,
    )


def _patch_builder_search_history(
    mocker: pytest_mock.MockFixture, **methods
) -> SimpleNamespace:
    client = SimpleNamespace(**methods)

    class _MockHistory:
        @staticmethod
        def prisma():
            return client

    mocker.patch(
        "backend.server.v2.builder.db.prisma.models.BuilderSearchHistory",
        new=_MockHistory,
        create=True,
    )
    return client


def test_get_block_categories_groups_and_limits(
    mocker: pytest_mock.MockFixture,
) -> None:
    block_one = _make_block_factory(
        block_id="block-one",
        name="Alpha Block",
        categories=[BlockCategory.AI, BlockCategory.DATA],
        block_type=DataBlockType.STANDARD,
    )
    block_two = _make_block_factory(
        block_id="block-two",
        name="Beta Block",
        categories=[BlockCategory.DATA],
        block_type=DataBlockType.STANDARD,
    )
    disabled_block = _make_block_factory(
        block_id="block-three",
        name="Disabled Block",
        categories=[BlockCategory.TEXT],
        block_type=DataBlockType.INPUT,
        disabled=True,
    )
    mocker.patch(
        "backend.server.v2.builder.db.load_all_blocks",
        return_value={
            "block-one": block_one,
            "block-two": block_two,
            "block-three": disabled_block,
        },
    )

    categories = builder_db.get_block_categories(category_blocks=1)

    assert [category.name for category in categories] == ["ai", "data"]
    ai_category = categories[0]
    data_category = categories[1]
    assert ai_category.total_blocks == 1
    assert data_category.total_blocks == 2
    assert len(ai_category.blocks) == 1 and ai_category.blocks[0].id == "block-one"
    assert len(data_category.blocks) == 1 and data_category.blocks[0].id == "block-one"


def test_get_blocks_filters_by_type_and_provider(
    mocker: pytest_mock.MockFixture,
) -> None:
    input_block = _make_block_factory(
        block_id="input-block",
        name="Input Block",
        categories=[BlockCategory.AI],
        block_type=DataBlockType.INPUT,
    )
    action_block = _make_block_factory(
        block_id="action-block",
        name="Action Block",
        categories=[BlockCategory.DATA],
        block_type=DataBlockType.STANDARD,
    )
    integration_block = _make_block_factory(
        block_id="integration-block",
        name="Integration Block",
        categories=[BlockCategory.DATA],
        block_type=DataBlockType.STANDARD,
        has_credentials=True,
        providers=[ProviderName.GITHUB],
    )
    mocker.patch(
        "backend.server.v2.builder.db.load_all_blocks",
        return_value={
            "input": input_block,
            "action": action_block,
            "integration": integration_block,
        },
    )

    input_result = builder_db.get_blocks(type="input")
    assert [block.id for block in input_result.blocks] == ["input-block"]
    assert input_result.pagination.total_items == 1

    provider_result = builder_db.get_blocks(provider=ProviderName.GITHUB)
    assert [block.id for block in provider_result.blocks] == ["integration-block"]

    with pytest.raises(ValueError):
        builder_db.get_blocks(category="ai", type="input")


def test_get_block_by_id_returns_matching_block(
    mocker: pytest_mock.MockFixture,
) -> None:
    block_factory = _make_block_factory(
        block_id="block-one",
        name="Target Block",
        categories=[BlockCategory.AI],
        block_type=DataBlockType.INPUT,
    )
    mocker.patch(
        "backend.server.v2.builder.db.load_all_blocks",
        return_value={"block-one": block_factory},
    )

    block = builder_db.get_block_by_id("block-one")
    assert block is not None
    assert block.id == "block-one"


def test_get_providers_filters_and_paginates(
    mocker: pytest_mock.MockFixture,
) -> None:
    providers = {
        ProviderName.GITHUB: builder_model.Provider(
            name=ProviderName.GITHUB,
            description="Git provider",
            integration_count=2,
        ),
        ProviderName.TWITTER: builder_model.Provider(
            name=ProviderName.TWITTER,
            description="Tweet provider",
            integration_count=1,
        ),
    }
    mocker.patch(
        "backend.server.v2.builder.db._get_all_providers",
        return_value=providers,
    )

    response = builder_db.get_providers(query="git", page=1, page_size=1)

    assert [provider.name for provider in response.providers] == [ProviderName.GITHUB]
    assert response.pagination.total_items == 2
    assert response.pagination.current_page == 1


@pytest.mark.asyncio
async def test_get_counts_merges_user_and_static_data(
    mocker: pytest_mock.MockFixture,
) -> None:
    mock_library_agent = mocker.patch.object(
        builder_db.prisma.models.LibraryAgent, "prisma"
    )
    mock_library_agent.return_value.count = mocker.AsyncMock(return_value=3)

    static_counts = {
        "all_blocks": 10,
        "input_blocks": 2,
        "action_blocks": 5,
        "output_blocks": 3,
        "integrations": 4,
        "marketplace_agents": 6,
    }
    mock_static_counts = mocker.patch(
        "backend.server.v2.builder.db._get_static_counts",
        new_callable=mocker.AsyncMock,
    )
    mock_static_counts.return_value = static_counts

    response = await builder_db.get_counts("user-123")

    assert response.my_agents == 3
    assert response.all_blocks == 10
    mock_static_counts.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_search_updates_existing_entry(
    mocker: pytest_mock.MockFixture,
) -> None:
    history_client = _patch_builder_search_history(
        mocker,
        update=mocker.AsyncMock(),
    )

    entry = builder_model.SearchEntry(
        search_id="search-1",
        search_query="query",
        filter=["blocks"],
        by_creator=["user"],
    )

    search_id = await builder_db.update_search("user-1", entry)

    assert search_id == "search-1"
    history_client.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_search_creates_new_entry(
    mocker: pytest_mock.MockFixture,
) -> None:
    history_client = _patch_builder_search_history(
        mocker,
        create=mocker.AsyncMock(return_value=SimpleNamespace(id="new-search")),
    )

    entry = builder_model.SearchEntry(search_query="something")
    search_id = await builder_db.update_search("user-3", entry)

    assert search_id == "new-search"
    history_client.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_recent_searches_returns_entries(
    mocker: pytest_mock.MockFixture,
) -> None:
    history_client = _patch_builder_search_history(
        mocker,
        find_many=mocker.AsyncMock(
            return_value=[
                SimpleNamespace(
                    searchQuery="query-1",
                    filter=["blocks"],
                    byCreator=["creator-1"],
                    id="search-1",
                )
            ]
        ),
    )

    searches = await builder_db.get_recent_searches("user-1")

    assert len(searches) == 1
    assert searches[0].search_query == "query-1"
    history_client.find_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_sorted_search_results_includes_all_sources(
    mocker: pytest_mock.MockFixture,
) -> None:
    block_result = _make_block_factory(
        block_id="alpha-block",
        name="Alpha Utility",
        categories=[BlockCategory.AI],
        block_type=DataBlockType.STANDARD,
    )
    integration_result = _make_block_factory(
        block_id="alpha-integration",
        name="Alpha Github Integration",
        categories=[BlockCategory.DATA],
        block_type=DataBlockType.STANDARD,
        has_credentials=True,
        providers=[ProviderName.GITHUB],
        include_llm_field=True,
    )
    mocker.patch(
        "backend.server.v2.builder.db.load_all_blocks",
        return_value={
            "block": block_result,
            "integration": integration_result,
        },
    )

    library_response = library_model.LibraryAgentResponse(
        agents=[_make_library_agent("Alpha Agent")],
        pagination=Pagination(
            total_items=1,
            total_pages=1,
            current_page=1,
            page_size=50,
        ),
    )
    mock_library = mocker.patch(
        "backend.server.v2.builder.db.library_db.list_library_agents",
        new_callable=mocker.AsyncMock,
    )
    mock_library.return_value = library_response

    store_response = store_model.StoreAgentsResponse(
        agents=[_make_store_agent("Alpha Marketplace")],
        pagination=Pagination(
            total_items=1,
            total_pages=1,
            current_page=1,
            page_size=50,
        ),
    )
    mock_store = mocker.patch(
        "backend.server.v2.builder.db.store_db.get_store_agents",
        new_callable=mocker.AsyncMock,
    )
    mock_store.return_value = store_response

    search_results = await builder_db.get_sorted_search_results(
        user_id="user-1",
        search_query="Alpha",
        filters=[
            "blocks",
            "integrations",
            "my_agents",
            "marketplace_agents",
        ],
        by_creator=["creator-z", "creator-a"],
    )

    assert search_results.total_items == {
        "blocks": 1,
        "integrations": 1,
        "marketplace_agents": 1,
        "my_agents": 1,
    }
    assert len(search_results.items) == 4
    assert any(isinstance(item, BlockInfo) for item in search_results.items)
    assert any(
        isinstance(item, library_model.LibraryAgent) for item in search_results.items
    )
    assert any(
        isinstance(item, store_model.StoreAgent) for item in search_results.items
    )

    mock_library.assert_awaited_once_with(
        user_id="user-1",
        search_term="Alpha",
        page=1,
        page_size=builder_db.MAX_LIBRARY_AGENT_RESULTS,
    )
    mock_store.assert_awaited_once_with(
        creators=["creator-a", "creator-z"],
        search_query="Alpha",
        page=1,
        page_size=builder_db.MAX_MARKETPLACE_AGENT_RESULTS,
    )


FIXED_TIME = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
