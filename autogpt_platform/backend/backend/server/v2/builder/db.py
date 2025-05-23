import functools

import backend.server.model as server_model
from backend.blocks import load_all_blocks
from backend.data.block import Block, BlockSchema, BlockType
from backend.data.credit import get_block_costs
from backend.integrations.providers import ProviderName
from backend.server.v2.builder.model import (
    BlockResponse,
    FilterType,
    Provider,
    ProviderResponse,
)


def get_blocks(
    filter: list[FilterType],
    query: str = "",
    providers: list[ProviderName] | None = None,
    page: int = 1,
    page_size: int = 50,
) -> BlockResponse:
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
        if query not in block.name.lower() or query not in block.description.lower():
            continue
        keep = False
        credentials = list(block.input_schema.get_credentials_fields().values())
        # Skip blocks that don't match the filter
        if (
            ("all_blocks" in filter)
            or ("input_blocks" in filter and block.block_type == BlockType.INPUT)
            or ("output_block" in filter and block.block_type == BlockType.OUTPUT)
        ):
            block_count += 1
            keep = True
        elif (
            "action_blocks" in filter
            and block.block_type != BlockType.INPUT
            and block.block_type != BlockType.OUTPUT
        ):
            block_count += 1
            keep = True
        elif "integrations" in filter and len(credentials) > 0:
            # Only keep if provider is in the list
            if providers:
                if any(c.provider in providers for c in credentials):
                    keep = True
                    integration_count += 1
            else:
                keep = True
                integration_count += 1

        if not keep:
            continue

        total += 1
        if skip > 0:
            skip -= 1
            continue
        if take > 0:
            take -= 1
            blocks.append(block)

    costs = get_block_costs()

    return BlockResponse(
        blocks=[{**b.to_dict(), "costs": costs.get(b.id, [])} for b in blocks],
        total_block_count=block_count,
        total_integration_count=integration_count,
        pagination=server_model.Pagination(
            total_items=total,
            total_pages=total // page_size + (1 if total % page_size > 0 else 0),
            current_page=page,
            page_size=page_size,
        ),
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
        pagination=server_model.Pagination(
            total_items=total,
            total_pages=total // page_size + (1 if total % page_size > 0 else 0),
            current_page=page,
            page_size=page_size,
        ),
    )


@functools.cache
def _get_all_providers() -> dict[ProviderName, Provider]:
    providers = {}
    for block_type in load_all_blocks().values():
        block: Block[BlockSchema, BlockSchema] = block_type()
        # Skip disabled blocks
        if block.disabled:
            continue
        credentials = list(block.input_schema.get_credentials_fields().values())
        for c in credentials:
            if c.provider in providers:
                providers[c.provider].integration_count += 1
            else:
                providers[c.provider] = Provider(
                    name=c.provider, description="", integration_count=1
                )
    return providers
