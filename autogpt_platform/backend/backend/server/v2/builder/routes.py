import logging
import typing

import fastapi
import fastapi.responses
from autogpt_libs.auth.depends import auth_middleware, get_user_id

import backend.server.model as server_model
import backend.server.v2.builder.db as builder_db
import backend.server.v2.builder.model as builder_model
import backend.server.v2.library.db as library_db
import backend.server.v2.library.model as library_model
import backend.server.v2.store.db as store_db
import backend.server.v2.store.model as store_model

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


@router.post(
    "/blocks",
    tags=["store", "private"],
    dependencies=[fastapi.Depends(auth_middleware)],
)
async def get_blocks(
    options: builder_model.SearchOptions,
    user_id: typing.Annotated[str, fastapi.Depends(get_user_id)],
) -> builder_model.BlockSearchResponse:
    # If no filters are provided, then we will return all types
    if not options.filter:
        options.filter = [
            "all_blocks",
            "integrations",
            "marketplace_agents",
            "my_agents",
            "providers",
        ]
    options.search_query = sanitize_query(options.search_query)
    options.page = options.page or 1
    options.page_size = options.page_size or 50

    # Blocks&Integrations
    blocks = builder_model.BlockResponse(
        blocks=[],
        total_block_count=0,
        total_integration_count=0,
        pagination=server_model.Pagination.empty(),
    )
    if (
        "all_blocks" in options.filter
        or "input_blocks" in options.filter
        or "action_blocks" in options.filter
        or "output_blocks" in options.filter
        or "integrations" in options.filter
    ):
        blocks = builder_db.get_blocks(
            filter=options.filter,
            query=options.search_query or "",
            page=options.page,
            page_size=options.page_size,
        )

    # Providers
    providers = builder_model.ProviderResponse(
        providers=[],
        pagination=server_model.Pagination.empty(),
    )
    # if "providers" in options.filter:
    #     providers = builder_db.get_providers(
    #         query=options.search_query or "",
    #         page=options.page,
    #         page_size=options.page_size,
    #     )

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
    if blocks.pagination.current_page < blocks.pagination.total_pages:
        more_pages = True
    if my_agents.pagination.current_page < my_agents.pagination.total_pages:
        more_pages = True
    if (
        marketplace_agents.pagination.current_page
        < marketplace_agents.pagination.total_pages
    ):
        more_pages = True

    # todo kcze sort results

    return builder_model.BlockSearchResponse(
        items=blocks.blocks
        + providers.providers
        + my_agents.agents
        + marketplace_agents.agents,
        total_items={
            "blocks": blocks.total_block_count,
            "integrations": blocks.total_integration_count,
            "providers": providers.pagination.total_items,
            "marketplace_agents": marketplace_agents.pagination.total_items,
            "my_agents": my_agents.pagination.total_items,
        },
        page=options.page,
        more_pages=more_pages,
    )
