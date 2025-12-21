import logging
from typing import Annotated, Sequence

import fastapi
from autogpt_libs.auth.dependencies import get_user_id, requires_user

from backend.integrations.providers import ProviderName
from backend.util.models import Pagination

from . import db as builder_db
from . import model as builder_model

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(
    dependencies=[fastapi.Security(requires_user)],
)


# Taken from backend/server/v2/store/db.py
def sanitize_query(query: str | None) -> str | None:
    if query is None:
        return query
    query = query.strip()[:100]
    return (
        query.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace(";", "\\;")
        .replace("--", "\\--")
        .replace("/*", "\\/*")
        .replace("*/", "\\*/")
    )


@router.get(
    "/suggestions",
    summary="Get Builder suggestions",
    response_model=builder_model.SuggestionsResponse,
)
async def get_suggestions(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
) -> builder_model.SuggestionsResponse:
    """
    Get all suggestions for the Blocks Menu.
    """
    return builder_model.SuggestionsResponse(
        otto_suggestions=[
            "What blocks do I need to get started?",
            "Help me create a list",
            "Help me feed my data to Google Maps",
        ],
        recent_searches=await builder_db.get_recent_searches(user_id),
        providers=[
            ProviderName.TWITTER,
            ProviderName.GITHUB,
            ProviderName.NOTION,
            ProviderName.GOOGLE,
            ProviderName.DISCORD,
            ProviderName.GOOGLE_MAPS,
        ],
        top_blocks=await builder_db.get_suggested_blocks(),
    )


@router.get(
    "/categories",
    summary="Get Builder block categories",
    response_model=Sequence[builder_model.BlockCategoryResponse],
)
async def get_block_categories(
    blocks_per_category: Annotated[int, fastapi.Query()] = 3,
) -> Sequence[builder_model.BlockCategoryResponse]:
    """
    Get all block categories with a specified number of blocks per category.
    """
    return builder_db.get_block_categories(blocks_per_category)


@router.get(
    "/blocks",
    summary="Get Builder blocks",
    response_model=builder_model.BlockResponse,
)
async def get_blocks(
    category: Annotated[str | None, fastapi.Query()] = None,
    type: Annotated[builder_model.BlockType | None, fastapi.Query()] = None,
    provider: Annotated[ProviderName | None, fastapi.Query()] = None,
    page: Annotated[int, fastapi.Query()] = 1,
    page_size: Annotated[int, fastapi.Query()] = 50,
) -> builder_model.BlockResponse:
    """
    Get blocks based on either category, type, or provider.
    """
    return builder_db.get_blocks(
        category=category,
        type=type,
        provider=provider,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/blocks/batch",
    summary="Get specific blocks",
    response_model=list[builder_model.BlockInfo],
)
async def get_specific_blocks(
    block_ids: Annotated[list[str], fastapi.Query()],
) -> list[builder_model.BlockInfo]:
    """
    Get specific blocks by their IDs.
    """
    blocks = []
    for block_id in block_ids:
        block = builder_db.get_block_by_id(block_id)
        if block:
            blocks.append(block)
    return blocks


@router.get(
    "/providers",
    summary="Get Builder integration providers",
    response_model=builder_model.ProviderResponse,
)
async def get_providers(
    page: Annotated[int, fastapi.Query()] = 1,
    page_size: Annotated[int, fastapi.Query()] = 50,
) -> builder_model.ProviderResponse:
    """
    Get all integration providers with their block counts.
    """
    return builder_db.get_providers(
        page=page,
        page_size=page_size,
    )


@router.get(
    "/search",
    summary="Builder search",
    tags=["store", "private"],
    response_model=builder_model.SearchResponse,
)
async def search(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    search_query: Annotated[str | None, fastapi.Query()] = None,
    filter: Annotated[list[builder_model.FilterType] | None, fastapi.Query()] = None,
    search_id: Annotated[str | None, fastapi.Query()] = None,
    by_creator: Annotated[list[str] | None, fastapi.Query()] = None,
    page: Annotated[int, fastapi.Query()] = 1,
    page_size: Annotated[int, fastapi.Query()] = 50,
) -> builder_model.SearchResponse:
    """
    Search for blocks (including integrations), marketplace agents, and user library agents.
    """
    # If no filters are provided, then we will return all types
    if not filter:
        filter = [
            "blocks",
            "integrations",
            "marketplace_agents",
            "my_agents",
        ]
    search_query = sanitize_query(search_query)

    # Get all possible results
    cached_results = await builder_db.get_sorted_search_results(
        user_id=user_id,
        search_query=search_query,
        filters=filter,
        by_creator=by_creator,
    )

    # Paginate results
    total_combined_items = len(cached_results.items)
    pagination = Pagination(
        total_items=total_combined_items,
        total_pages=(total_combined_items + page_size - 1) // page_size,
        current_page=page,
        page_size=page_size,
    )

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_items = cached_results.items[start_idx:end_idx]

    # Update the search entry by id
    search_id = await builder_db.update_search(
        user_id,
        builder_model.SearchEntry(
            search_query=search_query,
            filter=filter,
            by_creator=by_creator,
            search_id=search_id,
        ),
    )

    return builder_model.SearchResponse(
        items=paginated_items,
        search_id=search_id,
        total_items=cached_results.total_items,
        pagination=pagination,
    )


@router.get(
    "/counts",
    summary="Get Builder item counts",
    response_model=builder_model.CountResponse,
)
async def get_counts(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
) -> builder_model.CountResponse:
    """
    Get item counts for the menu categories in the Blocks Menu.
    """
    return await builder_db.get_counts(user_id)
