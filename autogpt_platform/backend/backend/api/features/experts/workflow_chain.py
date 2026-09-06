from collections import Counter
from collections.abc import Sequence

import prisma

from backend.api.features.experts.models import (
    ExpertWorkflowChainItem,
    ExpertWorkflowChainKind,
)
from backend.blocks import get_block
from backend.blocks._base import AnyBlockSchema, BlockType
from backend.integrations.providers import ProviderName

CHAIN_LENGTH = 3
_PROVIDER_PRIORITY = 100

_KIND_BY_BLOCK_TYPE: dict[BlockType, ExpertWorkflowChainKind] = {
    BlockType.INPUT: "input",
    BlockType.OUTPUT: "output",
    BlockType.WEBHOOK: "trigger",
    BlockType.WEBHOOK_MANUAL: "trigger",
    BlockType.AGENT: "agent",
    BlockType.AI: "ai",
    BlockType.MCP_TOOL: "mcp",
    BlockType.HUMAN_IN_THE_LOOP: "human",
}

# Chain reads left to right like the graph: what starts it, what it uses,
# what it produces.
_DISPLAY_ORDER: dict[ExpertWorkflowChainKind, int] = {
    "trigger": 0,
    "input": 0,
    "output": 2,
}

ChainKey = tuple[ExpertWorkflowChainKind, str | None]


def build_workflow_chain(
    nodes: Sequence[prisma.models.AgentNode],
) -> list[ExpertWorkflowChainItem]:
    counts: Counter[ChainKey] = Counter()
    for node in nodes:
        key = _classify_node(node)
        if key is not None:
            counts[key] += 1

    ranked = sorted(counts.items(), key=_rank)[:CHAIN_LENGTH]
    ordered = sorted(ranked, key=lambda item: _DISPLAY_ORDER.get(item[0][0], 1))
    return [
        ExpertWorkflowChainItem(kind=kind, provider=provider)
        for (kind, provider), _ in ordered
    ]


def _rank(item: tuple[ChainKey, int]) -> tuple[int, str, str]:
    (kind, provider), count = item
    score = count + (_PROVIDER_PRIORITY if provider else 0)
    return (-score, kind, provider or "")


def _classify_node(node: prisma.models.AgentNode) -> ChainKey | None:
    block = get_block(node.agentBlockId)
    if block is None or block.block_type == BlockType.NOTE:
        return None
    credentials_key = _credentials_key(block, node.constantInput)
    if credentials_key is not None:
        return credentials_key
    kind = _KIND_BY_BLOCK_TYPE.get(block.block_type)
    return (kind, None) if kind else None


def _credentials_key(block: AnyBlockSchema, constant_input: object) -> ChainKey | None:
    """A single-provider field names the integration; a model-discriminated
    field resolves through the node's chosen model, and stays a generic AI
    step when the model is unknown."""
    inputs = constant_input if isinstance(constant_input, dict) else {}
    for info in block.input_schema.get_credentials_fields_info().values():
        if info.discriminator and info.discriminator_mapping:
            chosen = inputs.get(info.discriminator)
            mapped = (
                info.discriminator_mapping.get(str(chosen))
                if chosen is not None
                else None
            )
            return (
                ("integration", _provider_slug(mapped))
                if mapped is not None
                else ("ai", None)
            )
        if len(info.provider) == 1:
            return ("integration", _provider_slug(next(iter(info.provider))))
        return ("ai", None)
    return None


def _provider_slug(provider: ProviderName | str) -> str:
    return provider.value if isinstance(provider, ProviderName) else provider
