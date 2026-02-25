import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator

from prisma.models import AgentBlock
from prisma.types import AgentBlockCreateInput

from backend.util import json

if TYPE_CHECKING:
    from backend.blocks._base import AnyBlockSchema

logger = logging.getLogger(__name__)


BlockInput = dict[str, Any]  # Input: 1 input pin <- 1 data.
BlockOutputEntry = tuple[str, Any]  # Output data should be a tuple of (name, value).
BlockOutput = AsyncGenerator[BlockOutputEntry, None]  # Output: 1 output pin -> N data.
CompletedBlockOutput = dict[str, list[Any]]  # Completed stream, collected as a dict.


async def initialize_blocks() -> None:
    from backend.blocks import get_blocks
    from backend.sdk.cost_integration import sync_all_provider_costs
    from backend.util.retry import func_retry

    sync_all_provider_costs()

    @func_retry
    async def sync_block_to_db(block: "AnyBlockSchema") -> None:
        existing_block = await AgentBlock.prisma().find_first(
            where={"OR": [{"id": block.id}, {"name": block.name}]}
        )
        if not existing_block:
            await AgentBlock.prisma().create(
                data=AgentBlockCreateInput(
                    id=block.id,
                    name=block.name,
                    inputSchema=json.dumps(block.input_schema.jsonschema()),
                    outputSchema=json.dumps(block.output_schema.jsonschema()),
                )
            )
            return

        input_schema = json.dumps(block.input_schema.jsonschema())
        output_schema = json.dumps(block.output_schema.jsonschema())
        if (
            block.id != existing_block.id
            or block.name != existing_block.name
            or input_schema != existing_block.inputSchema
            or output_schema != existing_block.outputSchema
        ):
            await AgentBlock.prisma().update(
                where={"id": existing_block.id},
                data={
                    "id": block.id,
                    "name": block.name,
                    "inputSchema": input_schema,
                    "outputSchema": output_schema,
                },
            )

    failed_blocks: list[str] = []
    for cls in get_blocks().values():
        block = cls()
        try:
            await sync_block_to_db(block)
        except Exception as e:
            logger.warning(
                f"Failed to sync block {block.name} to database: {e}. "
                "Block is still available in memory.",
                exc_info=True,
            )
            failed_blocks.append(block.name)

    if failed_blocks:
        logger.error(
            f"Failed to sync {len(failed_blocks)} block(s) to database: "
            f"{', '.join(failed_blocks)}. These blocks are still available in memory."
        )
