"""Block facts the registry and the copilot tools share.

Lives below both ``backend.copilot.tools`` and the registry sources so that
importing the registry never imports the tools package (which imports the
registry tools, which import the registry: a cycle otherwise).
"""

from functools import cache

from backend.blocks._base import AnyBlockSchema, BlockSchemaInput, BlockType
from backend.integrations.providers import ProviderName

# Block types that only work within graphs and cannot run standalone in CoPilot.
COPILOT_EXCLUDED_BLOCK_TYPES = {
    BlockType.INPUT,  # Graph interface definition - data enters via chat, not graph inputs
    BlockType.OUTPUT,  # Graph interface definition - data exits via chat, not graph outputs
    BlockType.WEBHOOK,  # Wait for external events - would hang forever in CoPilot
    BlockType.WEBHOOK_MANUAL,  # Same as WEBHOOK
    BlockType.NOTE,  # Visual annotation only - no runtime behavior
    BlockType.HUMAN_IN_THE_LOOP,  # Pauses for human approval - CoPilot IS human-in-the-loop
    BlockType.AGENT,  # AgentExecutorBlock requires execution_context - use run_agent tool
    BlockType.MCP_TOOL,  # MCP servers are capabilities of their own with discovery + auth
}

# Specific block IDs excluded from CoPilot (STANDARD type but still require graph context)
COPILOT_EXCLUDED_BLOCK_IDS = {
    # OrchestratorBlock - dynamically discovers downstream blocks via graph topology;
    # usable in agent graphs (guide hardcodes its ID) but cannot run standalone.
    "3b191d9f-356f-482d-8238-ba04b6d18381",
    # AutoPilotBlock - has dedicated run_sub_session tool with async start +
    # poll lifecycle. Running it as a capability would block the parent stream
    # for the sub-Otto's entire runtime (15-45+ min typical).
    "c069dc6b-c3ed-4c12-b6e5-d47361e64ce6",
}


def is_graph_only_block(block: AnyBlockSchema) -> bool:
    return (
        block.block_type in COPILOT_EXCLUDED_BLOCK_TYPES
        or block.id in COPILOT_EXCLUDED_BLOCK_IDS
    )


def get_block_provider(block: AnyBlockSchema) -> str | None:
    """Sole integration provider slug for a block, or None when the block
    uses zero or multiple providers."""
    try:
        return _get_input_schema_provider(block.input_schema)
    except Exception:
        return None


@cache
def _get_input_schema_provider(input_schema: type[BlockSchemaInput]) -> str | None:
    infos = input_schema.get_credentials_fields_info()
    providers = {
        ProviderName(provider).value
        for info in infos.values()
        for provider in info.provider
    }
    if len(providers) != 1:
        return None
    return next(iter(providers))
