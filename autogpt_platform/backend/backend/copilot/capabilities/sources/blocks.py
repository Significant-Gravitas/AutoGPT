"""Blocks as capability entries.

Mirrors ``find_block``: disabled blocks are dropped, graph-only block types
(inputs, outputs, webhooks, MCP tool nodes...) stay ``graph``-context so an
agent-building search still finds them, and every other block is usable in
both contexts.  The class comes from ``Block.capability_kind``.
"""

import logging

from backend.blocks import get_blocks
from backend.blocks._base import AnyBlockSchema
from backend.copilot.capabilities.models import (
    CapabilityEntry,
    Connection,
    Implementation,
    clip_purpose,
)
from backend.copilot.capabilities.text import tokenize
from backend.copilot.tools.find_block import (
    COPILOT_EXCLUDED_BLOCK_IDS,
    COPILOT_EXCLUDED_BLOCK_TYPES,
)
from backend.copilot.tools.helpers import get_block_provider
from backend.data.model import CredentialsFieldInfo

logger = logging.getLogger(__name__)


def block_entries() -> list[CapabilityEntry]:
    entries: list[CapabilityEntry] = []
    for block_id, block_cls in get_blocks().items():
        try:
            block = block_cls()
        except Exception:
            logger.debug(
                "Skipping block %s: cannot instantiate", block_id, exc_info=True
            )
            continue
        if not block.disabled:
            entries.append(_block_entry(block))
    return entries


def _block_entry(block: AnyBlockSchema) -> CapabilityEntry:
    graph_only = (
        block.block_type in COPILOT_EXCLUDED_BLOCK_TYPES
        or block.id in COPILOT_EXCLUDED_BLOCK_IDS
    )
    provider = get_block_provider(block)
    # Credentials may sit inside a nested input (the Google Sheets picker),
    # which ``get_credentials_fields`` does not see; the info view does.
    credential_infos = block.input_schema.get_credentials_fields_info()
    tags = sorted(set(tokenize(block.name)))
    tags += sorted(category.value.lower() for category in block.categories)
    if provider:
        tags.append(provider)
    if block.capability_kind == "primitive":
        tags.append("primitive")
    return CapabilityEntry(
        id=f"block:{block.id}",
        kind="block",
        klass=block.capability_kind,
        name=block.name,
        purpose=clip_purpose(block.optimized_description or block.description),
        tags=tags,
        context="graph" if graph_only else "both",
        implementations=[Implementation(kind="block", ref=block.id, name=block.name)],
        connection=_connection(provider, credential_infos),
        argument_names=[
            field
            for field in block.input_schema.model_fields
            if field not in credential_infos
        ],
        schema_ref=f"block:{block.id}",
        sensitive=block.is_sensitive_action,
    )


def _connection(
    provider: str | None, credential_infos: dict[str, CredentialsFieldInfo]
) -> Connection:
    if not credential_infos:
        return Connection(required=False)
    # Credentials chosen by request URL (authenticated HTTP) are host-keyed.
    if any(info.discriminator == "url" for info in credential_infos.values()):
        return Connection(required=True, key_type="host")
    # ``provider`` is None for blocks offering several providers (the LLM
    # blocks); those still need *a* credential, just not a fixed one.
    return Connection(required=True, key_type="provider", key=provider)
