import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Sequence

import prisma

import backend.api.features.library.db as library_db
import backend.api.features.library.model as library_model
import backend.api.features.store.db as store_db
import backend.api.features.store.model as store_model
import backend.data.block
from backend.blocks import load_all_blocks
from backend.blocks.llm import LlmModel
from backend.data.block import AnyBlockSchema, BlockCategory, BlockInfo, BlockSchema
from backend.data.db import query_raw_with_schema
from backend.integrations.providers import ProviderName
from backend.util.cache import cached
from backend.util.models import Pagination

from .model import (
    BlockCategoryResponse,
    BlockResponse,
    BlockType,
    CountResponse,
    FilterType,
    Provider,
    ProviderResponse,
    SearchEntry,
)

logger = logging.getLogger(__name__)
llm_models = [name.name.lower().replace("_", " ") for name in LlmModel]

MAX_LIBRARY_AGENT_RESULTS = 100
MAX_MARKETPLACE_AGENT_RESULTS = 100
MIN_SCORE_FOR_FILTERED_RESULTS = 10.0

SearchResultItem = BlockInfo | library_model.LibraryAgent | store_model.StoreAgent


@dataclass
class _ScoredItem:
    item: SearchResultItem
    filter_type: FilterType
    score: float
    sort_key: str


@dataclass
class _SearchCacheEntry:
    items: list[SearchResultItem]
    total_items: dict[FilterType, int]


def get_block_categories(category_blocks: int = 3) -> list[BlockCategoryResponse]:
    categories: dict[BlockCategory, BlockCategoryResponse] = {}

    for block_type in load_all_blocks().values():
        block: AnyBlockSchema = block_type()
        # Skip disabled blocks
        if block.disabled:
            continue
        # Skip blocks that don't have categories (all should have at least one)
        if not block.categories:
            continue

        # Add block to the categories
        for category in block.categories:
            if category not in categories:
                categories[category] = BlockCategoryResponse(
                    name=category.name.lower(),
                    total_blocks=0,
                    blocks=[],
                )

            categories[category].total_blocks += 1

            # Append if the category has less than the specified number of blocks
            if len(categories[category].blocks) < category_blocks:
                categories[category].blocks.append(block.get_info())

    # Sort categories by name
    return sorted(categories.values(), key=lambda x: x.name)


def get_blocks(
    *,
    category: str | None = None,
    type: BlockType | None = None,
    provider: ProviderName | None = None,
    page: int = 1,
    page_size: int = 50,
) -> BlockResponse:
    """
    Get blocks based on either category, type or provider.
    Providing nothing fetches all block types.
    """
    # Only one of category, type, or provider can be specified
    if (category and type) or (category and provider) or (type and provider):
        raise ValueError("Only one of category, type, or provider can be specified")

    blocks: list[AnyBlockSchema] = []
    skip = (page - 1) * page_size
    take = page_size
    total = 0

    for block_type in load_all_blocks().values():
        block: AnyBlockSchema = block_type()
        # Skip disabled blocks
        if block.disabled:
            continue
        # Skip blocks that don't match the category
        if category and category not in {c.name.lower() for c in block.categories}:
            continue
        # Skip blocks that don't match the type
        if (
            (type == "input" and block.block_type.value != "Input")
            or (type == "output" and block.block_type.value != "Output")
            or (type == "action" and block.block_type.value in ("Input", "Output"))
        ):
            continue
        # Skip blocks that don't match the provider
        if provider:
            credentials_info = block.input_schema.get_credentials_fields_info().values()
            if not any(provider in info.provider for info in credentials_info):
                continue

        total += 1
        if skip > 0:
            skip -= 1
            continue
        if take > 0:
            take -= 1
            blocks.append(block)

    return BlockResponse(
        blocks=[b.get_info() for b in blocks],
        pagination=Pagination(
            total_items=total,
            total_pages=(total + page_size - 1) // page_size,
            current_page=page,
            page_size=page_size,
        ),
    )


def get_block_by_id(block_id: str) -> BlockInfo | None:
    """
    Get a specific block by its ID.
    """
    for block_type in load_all_blocks().values():
        block: AnyBlockSchema = block_type()
        if block.id == block_id:
            return block.get_info()
    return None


async def update_search(user_id: str, search: SearchEntry) -> str:
    """
    Upsert a search request for the user and return the search ID.
    """
    if search.search_id:
        # Update existing search
        await prisma.models.BuilderSearchHistory.prisma().update(
            where={
                "id": search.search_id,
            },
            data={
                "searchQuery": search.search_query or "",
                "filter": search.filter or [],  # type: ignore
                "byCreator": search.by_creator or [],
            },
        )
        return search.search_id
    else:
        # Create new search
        new_search = await prisma.models.BuilderSearchHistory.prisma().create(
            data={
                "userId": user_id,
                "searchQuery": search.search_query or "",
                "filter": search.filter or [],  # type: ignore
                "byCreator": search.by_creator or [],
            }
        )
        return new_search.id


async def get_recent_searches(user_id: str, limit: int = 5) -> list[SearchEntry]:
    """
    Get the user's most recent search requests.
    """
    searches = await prisma.models.BuilderSearchHistory.prisma().find_many(
        where={
            "userId": user_id,
        },
        order={
            "updatedAt": "desc",
        },
        take=limit,
    )
    return [
        SearchEntry(
            search_query=s.searchQuery,
            filter=s.filter,  # type: ignore
            by_creator=s.byCreator,
            search_id=s.id,
        )
        for s in searches
    ]


async def get_sorted_search_results(
    *,
    user_id: str,
    search_query: str | None,
    filters: Sequence[FilterType],
    by_creator: Sequence[str] | None = None,
) -> _SearchCacheEntry:
    normalized_filters: tuple[FilterType, ...] = tuple(sorted(set(filters or [])))
    normalized_creators: tuple[str, ...] = tuple(sorted(set(by_creator or [])))
    return await _build_cached_search_results(
        user_id=user_id,
        search_query=search_query or "",
        filters=normalized_filters,
        by_creator=normalized_creators,
    )


@cached(ttl_seconds=300, shared_cache=True)
async def _build_cached_search_results(
    user_id: str,
    search_query: str,
    filters: tuple[FilterType, ...],
    by_creator: tuple[str, ...],
) -> _SearchCacheEntry:
    normalized_query = (search_query or "").strip().lower()

    include_blocks = "blocks" in filters
    include_integrations = "integrations" in filters
    include_library_agents = "my_agents" in filters
    include_marketplace_agents = "marketplace_agents" in filters

    scored_items: list[_ScoredItem] = []
    total_items: dict[FilterType, int] = {
        "blocks": 0,
        "integrations": 0,
        "marketplace_agents": 0,
        "my_agents": 0,
    }

    block_results, block_total, integration_total = _collect_block_results(
        normalized_query=normalized_query,
        include_blocks=include_blocks,
        include_integrations=include_integrations,
    )
    scored_items.extend(block_results)
    total_items["blocks"] = block_total
    total_items["integrations"] = integration_total

    if include_library_agents:
        library_response = await library_db.list_library_agents(
            user_id=user_id,
            search_term=search_query or None,
            page=1,
            page_size=MAX_LIBRARY_AGENT_RESULTS,
        )
        total_items["my_agents"] = library_response.pagination.total_items
        scored_items.extend(
            _build_library_items(
                agents=library_response.agents,
                normalized_query=normalized_query,
            )
        )

    if include_marketplace_agents:
        marketplace_response = await store_db.get_store_agents(
            creators=list(by_creator) or None,
            search_query=search_query or None,
            page=1,
            page_size=MAX_MARKETPLACE_AGENT_RESULTS,
        )
        total_items["marketplace_agents"] = marketplace_response.pagination.total_items
        scored_items.extend(
            _build_marketplace_items(
                agents=marketplace_response.agents,
                normalized_query=normalized_query,
            )
        )

    sorted_items = sorted(
        scored_items,
        key=lambda entry: (-entry.score, entry.sort_key, entry.filter_type),
    )

    return _SearchCacheEntry(
        items=[entry.item for entry in sorted_items],
        total_items=total_items,
    )


def _collect_block_results(
    *,
    normalized_query: str,
    include_blocks: bool,
    include_integrations: bool,
) -> tuple[list[_ScoredItem], int, int]:
    results: list[_ScoredItem] = []
    block_count = 0
    integration_count = 0

    if not include_blocks and not include_integrations:
        return results, block_count, integration_count

    for block_type in load_all_blocks().values():
        block: AnyBlockSchema = block_type()
        if block.disabled:
            continue

        block_info = block.get_info()
        credentials = list(block.input_schema.get_credentials_fields().values())
        is_integration = len(credentials) > 0

        if is_integration and not include_integrations:
            continue
        if not is_integration and not include_blocks:
            continue

        score = _score_block(block, block_info, normalized_query)
        if not _should_include_item(score, normalized_query):
            continue

        filter_type: FilterType = "integrations" if is_integration else "blocks"
        if is_integration:
            integration_count += 1
        else:
            block_count += 1

        results.append(
            _ScoredItem(
                item=block_info,
                filter_type=filter_type,
                score=score,
                sort_key=_get_item_name(block_info),
            )
        )

    return results, block_count, integration_count


def _build_library_items(
    *,
    agents: list[library_model.LibraryAgent],
    normalized_query: str,
) -> list[_ScoredItem]:
    results: list[_ScoredItem] = []

    for agent in agents:
        score = _score_library_agent(agent, normalized_query)
        if not _should_include_item(score, normalized_query):
            continue

        results.append(
            _ScoredItem(
                item=agent,
                filter_type="my_agents",
                score=score,
                sort_key=_get_item_name(agent),
            )
        )

    return results


def _build_marketplace_items(
    *,
    agents: list[store_model.StoreAgent],
    normalized_query: str,
) -> list[_ScoredItem]:
    results: list[_ScoredItem] = []

    for agent in agents:
        score = _score_store_agent(agent, normalized_query)
        if not _should_include_item(score, normalized_query):
            continue

        results.append(
            _ScoredItem(
                item=agent,
                filter_type="marketplace_agents",
                score=score,
                sort_key=_get_item_name(agent),
            )
        )

    return results


def get_providers(
    query: str = "",
    page: int = 1,
    page_size: int = 50,
) -> ProviderResponse:
    providers = []
    query = query.lower()

    skip = (page - 1) * page_size
    take = page_size

    all_providers = _get_all_providers()

    for provider in all_providers.values():
        if (
            query not in provider.name.value.lower()
            and query not in provider.description.lower()
        ):
            continue
        if skip > 0:
            skip -= 1
            continue
        if take > 0:
            take -= 1
            providers.append(provider)

    total = len(all_providers)

    return ProviderResponse(
        providers=providers,
        pagination=Pagination(
            total_items=total,
            total_pages=(total + page_size - 1) // page_size,
            current_page=page,
            page_size=page_size,
        ),
    )


async def get_counts(user_id: str) -> CountResponse:
    my_agents = await prisma.models.LibraryAgent.prisma().count(
        where={
            "userId": user_id,
            "isDeleted": False,
            "isArchived": False,
        }
    )
    counts = await _get_static_counts()
    return CountResponse(
        my_agents=my_agents,
        **counts,
    )


@cached(ttl_seconds=3600)
async def _get_static_counts():
    """
    Get counts of blocks, integrations, and marketplace agents.
    This is cached to avoid unnecessary database queries and calculations.
    """
    all_blocks = 0
    input_blocks = 0
    action_blocks = 0
    output_blocks = 0
    integrations = 0

    for block_type in load_all_blocks().values():
        block: AnyBlockSchema = block_type()
        if block.disabled:
            continue

        all_blocks += 1

        if block.block_type.value == "Input":
            input_blocks += 1
        elif block.block_type.value == "Output":
            output_blocks += 1
        else:
            action_blocks += 1

        credentials = list(block.input_schema.get_credentials_fields().values())
        if len(credentials) > 0:
            integrations += 1

    marketplace_agents = await prisma.models.StoreAgent.prisma().count()

    return {
        "all_blocks": all_blocks,
        "input_blocks": input_blocks,
        "action_blocks": action_blocks,
        "output_blocks": output_blocks,
        "integrations": integrations,
        "marketplace_agents": marketplace_agents,
    }


def _matches_llm_model(schema_cls: type[BlockSchema], query: str) -> bool:
    for field in schema_cls.model_fields.values():
        if field.annotation == LlmModel:
            # Check if query matches any value in llm_models
            if any(query in name for name in llm_models):
                return True
    return False


def _score_block(
    block: AnyBlockSchema,
    block_info: BlockInfo,
    normalized_query: str,
) -> float:
    if not normalized_query:
        return 0.0

    name = block_info.name.lower()
    description = block_info.description.lower()
    score = _score_primary_fields(name, description, normalized_query)

    category_text = " ".join(
        category.get("category", "").lower() for category in block_info.categories
    )
    score += _score_additional_field(category_text, normalized_query, 12, 6)

    credentials_info = block.input_schema.get_credentials_fields_info().values()
    provider_names = [
        provider.value.lower()
        for info in credentials_info
        for provider in info.provider
    ]
    provider_text = " ".join(provider_names)
    score += _score_additional_field(provider_text, normalized_query, 15, 6)

    if _matches_llm_model(block.input_schema, normalized_query):
        score += 20

    return score


def _score_library_agent(
    agent: library_model.LibraryAgent,
    normalized_query: str,
) -> float:
    if not normalized_query:
        return 0.0

    name = agent.name.lower()
    description = (agent.description or "").lower()
    instructions = (agent.instructions or "").lower()

    score = _score_primary_fields(name, description, normalized_query)
    score += _score_additional_field(instructions, normalized_query, 15, 6)
    score += _score_additional_field(
        agent.creator_name.lower(), normalized_query, 10, 5
    )

    return score


def _score_store_agent(
    agent: store_model.StoreAgent,
    normalized_query: str,
) -> float:
    if not normalized_query:
        return 0.0

    name = agent.agent_name.lower()
    description = agent.description.lower()
    sub_heading = agent.sub_heading.lower()

    score = _score_primary_fields(name, description, normalized_query)
    score += _score_additional_field(sub_heading, normalized_query, 12, 6)
    score += _score_additional_field(agent.creator.lower(), normalized_query, 10, 5)

    return score


def _score_primary_fields(name: str, description: str, query: str) -> float:
    score = 0.0
    if name == query:
        score += 120
    elif name.startswith(query):
        score += 90
    elif query in name:
        score += 60

    score += SequenceMatcher(None, name, query).ratio() * 50
    if description:
        if query in description:
            score += 30
        score += SequenceMatcher(None, description, query).ratio() * 25
    return score


def _score_additional_field(
    value: str,
    query: str,
    contains_weight: float,
    similarity_weight: float,
) -> float:
    if not value or not query:
        return 0.0

    score = 0.0
    if query in value:
        score += contains_weight
    score += SequenceMatcher(None, value, query).ratio() * similarity_weight
    return score


def _should_include_item(score: float, normalized_query: str) -> bool:
    if not normalized_query:
        return True
    return score >= MIN_SCORE_FOR_FILTERED_RESULTS


def _get_item_name(item: SearchResultItem) -> str:
    if isinstance(item, BlockInfo):
        return item.name.lower()
    if isinstance(item, library_model.LibraryAgent):
        return item.name.lower()
    return item.agent_name.lower()


@cached(ttl_seconds=3600)
def _get_all_providers() -> dict[ProviderName, Provider]:
    providers: dict[ProviderName, Provider] = {}

    for block_type in load_all_blocks().values():
        block: AnyBlockSchema = block_type()
        if block.disabled:
            continue

        credentials_info = block.input_schema.get_credentials_fields_info().values()
        for info in credentials_info:
            for provider in info.provider:  # provider is a ProviderName enum member
                if provider in providers:
                    providers[provider].integration_count += 1
                else:
                    providers[provider] = Provider(
                        name=provider, description="", integration_count=1
                    )
    return providers


@cached(ttl_seconds=3600)
async def get_suggested_blocks(count: int = 5) -> list[BlockInfo]:
    suggested_blocks = []
    # Sum the number of executions for each block type
    # Prisma cannot group by nested relations, so we do a raw query
    # Calculate the cutoff timestamp
    timestamp_threshold = datetime.now(timezone.utc) - timedelta(days=30)

    results = await query_raw_with_schema(
        """
        SELECT
            agent_node."agentBlockId" AS block_id,
            COUNT(execution.id) AS execution_count
        FROM {schema_prefix}"AgentNodeExecution" execution
        JOIN {schema_prefix}"AgentNode" agent_node ON execution."agentNodeId" = agent_node.id
        WHERE execution."endedTime" >= $1::timestamp
        GROUP BY agent_node."agentBlockId"
        ORDER BY execution_count DESC;
        """,
        timestamp_threshold,
    )

    # Get the top blocks based on execution count
    # But ignore Input and Output blocks
    blocks: list[tuple[BlockInfo, int]] = []

    for block_type in load_all_blocks().values():
        block: AnyBlockSchema = block_type()
        if block.disabled or block.block_type in (
            backend.data.block.BlockType.INPUT,
            backend.data.block.BlockType.OUTPUT,
            backend.data.block.BlockType.AGENT,
        ):
            continue
        # Find the execution count for this block
        execution_count = next(
            (row["execution_count"] for row in results if row["block_id"] == block.id),
            0,
        )
        blocks.append((block.get_info(), execution_count))
    # Sort blocks by execution count
    blocks.sort(key=lambda x: x[1], reverse=True)

    suggested_blocks = [block[0] for block in blocks]

    # Return the top blocks
    return suggested_blocks[:count]
