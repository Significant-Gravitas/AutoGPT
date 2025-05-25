import logging
from typing import Annotated, Sequence

import fastapi
from autogpt_libs.auth.depends import auth_middleware, get_user_id

import backend.server.model as server_model
import backend.server.v2.builder.db as builder_db
import backend.server.v2.builder.model as builder_model
import backend.server.v2.library.db as library_db
import backend.server.v2.library.model as library_model
import backend.server.v2.store.db as store_db
import backend.server.v2.store.model as store_model
from backend.integrations.providers import ProviderName

logger = logging.getLogger(__name__)

router = fastapi.APIRouter()


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
    dependencies=[fastapi.Depends(auth_middleware)],
)
async def get_suggestions(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
) -> builder_model.SuggestionsResponse:
    # todo kcze temp response
    return builder_model.SuggestionsResponse(
        otto_suggestions=[
            "What blocks do I need to get started?",
            "Help me create a list",
            "Help me feed my data to Google Maps",
        ],
        recent_searches=[
            "image generation",
            "deepfake",
            "competitor analysis",
        ],
        providers=[
            ProviderName.TWITTER,
            ProviderName.GITHUB,
            ProviderName.HUBSPOT,
            ProviderName.EXA,
            ProviderName.JINA,
            ProviderName.GOOGLE_MAPS,
        ],
        top_blocks=builder_db.get_blocks(page_size=5).blocks,
    )


@router.get(
    "/categories",
    dependencies=[fastapi.Depends(auth_middleware)],
)
async def get_block_categories(
    category_blocks: Annotated[int, fastapi.Query()] = 3,
) -> Sequence[builder_model.BlockCategoryResponse]:
    return builder_db.get_block_categories(category_blocks)


@router.get(
    "/blocks",
    dependencies=[fastapi.Depends(auth_middleware)],
)
async def get_blocks(
    category: Annotated[str | None, fastapi.Query()] = None,
    type: Annotated[builder_model.BlockType | None, fastapi.Query()] = None,
    provider: Annotated[ProviderName | None, fastapi.Query()] = None,
    page: Annotated[int, fastapi.Query()] = 1,
    page_size: Annotated[int, fastapi.Query()] = 50,
) -> builder_model.BlockResponse:
    return builder_db.get_blocks(
        category=category,
        type=type,
        provider=provider,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/providers",
    dependencies=[fastapi.Depends(auth_middleware)],
)
async def get_providers(
    page: Annotated[int, fastapi.Query()] = 1,
    page_size: Annotated[int, fastapi.Query()] = 50,
) -> builder_model.ProviderResponse:
    return builder_db.get_providers(
        page=page,
        page_size=page_size,
    )


@router.post(
    "/search",
    tags=["store", "private"],
    dependencies=[fastapi.Depends(auth_middleware)],
)
async def search(
    options: builder_model.SearchRequest,
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
) -> builder_model.SearchResponse:
    # If no filters are provided, then we will return all types
    if not options.filter:
        options.filter = [
            "blocks",
            "integrations",
            "providers",
            "marketplace_agents",
            "my_agents",
        ]
    options.search_query = sanitize_query(options.search_query)
    options.page = options.page or 1
    options.page_size = options.page_size or 50

    # Blocks&Integrations
    blocks = builder_model.SearchBlocksResponse(
        blocks=builder_model.BlockResponse(
            blocks=[],
            pagination=server_model.Pagination.empty(),
        ),
        total_block_count=0,
        total_integration_count=0,
    )
    if "blocks" in options.filter or "integrations" in options.filter:
        blocks = builder_db.search_blocks(
            include_blocks="blocks" in options.filter,
            include_integrations="integrations" in options.filter,
            query=options.search_query or "",
            page=options.page,
            page_size=options.page_size,
        )

    # Providers
    providers = builder_model.ProviderResponse(
        providers=[],
        pagination=server_model.Pagination.empty(),
    )
    if "providers" in options.filter:
        providers = builder_db.get_providers(
            query=options.search_query or "",
            page=options.page,
            page_size=options.page_size,
        )

    # Library Agents
    my_agents = library_model.LibraryAgentResponse(
        agents=[],
        pagination=server_model.Pagination.empty(),
    )
    if "my_agents" in options.filter:
        my_agents = await library_db.list_library_agents(
            user_id=user_id,
            search_term=options.search_query,
            page=options.page,
            page_size=options.page_size,
        )

    # Marketplace Agents
    marketplace_agents = store_model.StoreAgentsResponse(
        agents=[],
        pagination=server_model.Pagination.empty(),
    )
    if "marketplace_agents" in options.filter:
        marketplace_agents = await store_db.get_store_agents(
            creators=options.by_creator,
            search_query=options.search_query,
            page=options.page,
            page_size=options.page_size,
        )

    more_pages = False
    if blocks.blocks.pagination.current_page < blocks.blocks.pagination.total_pages:
        more_pages = True
    if my_agents.pagination.current_page < my_agents.pagination.total_pages:
        more_pages = True
    if (
        marketplace_agents.pagination.current_page
        < marketplace_agents.pagination.total_pages
    ):
        more_pages = True

    # todo kcze sort results

    return builder_model.SearchResponse(
        items=blocks.blocks.blocks
        + providers.providers
        + my_agents.agents
        + marketplace_agents.agents,
        total_items={
            "blocks": blocks.total_block_count,
            "integrations": blocks.total_integration_count,
            "marketplace_agents": marketplace_agents.pagination.total_items,
            "my_agents": my_agents.pagination.total_items,
        },
        page=options.page,
        more_pages=more_pages,
    )
