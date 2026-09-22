"""Blocks as capability entries.

Mirrors ``find_block``: disabled blocks are dropped, graph-only block types
(inputs, outputs, webhooks, MCP tool nodes...) stay ``graph``-context so an
agent-building search still finds them, and every other block is usable in
both contexts.  The class comes from ``Block.capability_kind``.
"""

import logging

from backend.blocks import get_blocks
from backend.blocks._base import AnyBlockSchema
from backend.copilot.capabilities.block_meta import (
    get_block_provider,
    is_graph_only_block,
)
from backend.copilot.capabilities.models import (
    CapabilityEntry,
    Connection,
    Implementation,
    clip_purpose,
    normalize_text,
)
from backend.copilot.capabilities.text import tokenize
from backend.data.model import CredentialsFieldInfo

logger = logging.getLogger(__name__)


def block_entries(*, include_disabled: bool = False) -> list[CapabilityEntry]:
    """Every indexable block.

    ``include_disabled`` keeps blocks the running environment has switched
    off, which is how the retrieval benchmark sees the same catalogue the
    recorded results were measured against: a block whose provider OAuth is
    unconfigured is disabled, and scoring retrieval on questions whose
    answer has been removed measures the environment, not the ranking.
    Entries are metadata only, so nothing here can run a disabled block.
    """
    entries: list[CapabilityEntry] = []
    for block_id, block_cls in get_blocks().items():
        try:
            block = block_cls()
            if include_disabled or not block.disabled:
                # Building the entry reads the block's schema, which a
                # malformed field definition can reject. One bad block must
                # cost its own entry, not the whole registry.
                entries.append(_block_entry(block))
        except Exception:
            logger.debug("Skipping block %s: cannot index", block_id, exc_info=True)
            continue
    return entries


def _block_entry(block: AnyBlockSchema) -> CapabilityEntry:
    graph_only = is_graph_only_block(block)
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
        # The optimized description is curated for retrieval (it is loaded from
        # the database at runtime), so the index sees it as well as the source.
        description=normalize_text(
            " ".join(
                text
                for text in (block.optimized_description, block.description)
                if text
            )
        ),
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
