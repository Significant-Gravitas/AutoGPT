import logging
from datetime import datetime, timedelta, timezone

import prisma

import backend.data.block
from backend.blocks import load_all_blocks
from backend.blocks.llm import LlmModel
from backend.data.block import Block, BlockCategory, BlockInfo, BlockSchema
from backend.integrations.providers import ProviderName
from backend.server.v2.builder.model import (
    BlockCategoryResponse,
    BlockResponse,
    BlockType,
    CountResponse,
    Provider,
    ProviderResponse,
    SearchBlocksResponse,
)
from backend.util.cache import cached
from backend.util.models import Pagination

logger = logging.getLogger(__name__)
llm_models = [name.name.lower().replace("_", " ") for name in LlmModel]
_static_counts_cache: dict | None = None
_suggested_blocks: list[BlockInfo] | None = None


def get_block_categories(category_blocks: int = 3) -> list[BlockCategoryResponse]:
    categories: dict[BlockCategory, BlockCategoryResponse] = {}

    for block_type in load_all_blocks().values():
        block: Block[BlockSchema, BlockSchema] = block_type()
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

    blocks: list[Block[BlockSchema, BlockSchema]] = []
    skip = (page - 1) * page_size
    take = page_size
    total = 0

    for block_type in load_all_blocks().values():
        block: Block[BlockSchema, BlockSchema] = block_type()
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
        block: Block[BlockSchema, BlockSchema] = block_type()
        if block.id == block_id:
            return block.get_info()
    return None


def search_blocks(
    include_blocks: bool = True,
    include_integrations: bool = True,
    query: str = "",
    page: int = 1,
    page_size: int = 50,
) -> SearchBlocksResponse:
    """
    Get blocks based on the filter and query.
    `providers` only applies for `integrations` filter.
    """
    blocks: list[Block[BlockSchema, BlockSchema]] = []
    query = query.lower()

    total = 0
    skip = (page - 1) * page_size
    take = page_size
    block_count = 0
    integration_count = 0

    for block_type in load_all_blocks().values():
        block: Block[BlockSchema, BlockSchema] = block_type()
        # Skip disabled blocks
        if block.disabled:
            continue
        # Skip blocks that don't match the query
        if (
            query not in block.name.lower()
            and query not in block.description.lower()
            and not _matches_llm_model(block.input_schema, query)
        ):
            continue
        keep = False
        credentials = list(block.input_schema.get_credentials_fields().values())
        if include_integrations and len(credentials) > 0:
            keep = True
            integration_count += 1
        if include_blocks and len(credentials) == 0:
            keep = True
            block_count += 1

        if not keep:
            continue

        total += 1
        if skip > 0:
            skip -= 1
            continue
        if take > 0:
            take -= 1
            blocks.append(block)

    return SearchBlocksResponse(
        blocks=BlockResponse(
            blocks=[b.get_info() for b in blocks],
            pagination=Pagination(
                total_items=total,
                total_pages=(total + page_size - 1) // page_size,
                current_page=page,
                page_size=page_size,
            ),
        ),
        total_block_count=block_count,
        total_integration_count=integration_count,
    )


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


async def _get_static_counts():
    """
    Get counts of blocks, integrations, and marketplace agents.
    This is cached to avoid unnecessary database queries and calculations.
    Can't use functools.cache here because the function is async.
    """
    global _static_counts_cache
    if _static_counts_cache is not None:
        return _static_counts_cache

    all_blocks = 0
    input_blocks = 0
    action_blocks = 0
    output_blocks = 0
    integrations = 0

    for block_type in load_all_blocks().values():
        block: Block[BlockSchema, BlockSchema] = block_type()
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

    _static_counts_cache = {
        "all_blocks": all_blocks,
        "input_blocks": input_blocks,
        "action_blocks": action_blocks,
        "output_blocks": output_blocks,
        "integrations": integrations,
        "marketplace_agents": marketplace_agents,
    }

    return _static_counts_cache


def _matches_llm_model(schema_cls: type[BlockSchema], query: str) -> bool:
    for field in schema_cls.model_fields.values():
        if field.annotation == LlmModel:
            # Check if query matches any value in llm_models
            if any(query in name for name in llm_models):
                return True
    return False


@cached(ttl_seconds=3600)
def _get_all_providers() -> dict[ProviderName, Provider]:
    providers: dict[ProviderName, Provider] = {}

    for block_type in load_all_blocks().values():
        block: Block[BlockSchema, BlockSchema] = block_type()
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


async def get_suggested_blocks(count: int = 5) -> list[BlockInfo]:
    global _suggested_blocks

    if _suggested_blocks is not None and len(_suggested_blocks) >= count:
        return _suggested_blocks[:count]

    _suggested_blocks = []
    # Sum the number of executions for each block type
    # Prisma cannot group by nested relations, so we do a raw query
    # Calculate the cutoff timestamp
    timestamp_threshold = datetime.now(timezone.utc) - timedelta(days=30)

    results = await prisma.get_client().query_raw(
        """
        SELECT
            agent_node."agentBlockId" AS block_id,
            COUNT(execution.id) AS execution_count
        FROM "AgentNodeExecution" execution
        JOIN "AgentNode" agent_node ON execution."agentNodeId" = agent_node.id
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
        block: Block[BlockSchema, BlockSchema] = block_type()
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

    _suggested_blocks = [block[0] for block in blocks]

    # Return the top blocks
    return _suggested_blocks[:count]
