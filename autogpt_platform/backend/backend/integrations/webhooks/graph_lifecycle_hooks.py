import asyncio
import logging
from typing import TYPE_CHECKING, Optional, cast, overload

from backend.blocks._base import BlockSchema
from backend.data.graph import set_node_webhook
from backend.data.integrations import get_webhook
from backend.integrations.creds_manager import IntegrationCredentialsManager

from . import get_webhook_manager, supports_webhooks

if TYPE_CHECKING:
    from backend.data.graph import BaseGraph, GraphModel, NodeModel
    from backend.data.model import Credentials

    from ._base import BaseWebhooksManager

logger = logging.getLogger(__name__)
credentials_manager = IntegrationCredentialsManager()


async def on_graph_activate(graph: "GraphModel", user_id: str) -> "GraphModel":
    """
    Hook to be called when a graph is activated/created.

    ⚠️ Assuming node entities are not re-used between graph versions, ⚠️
    this hook calls `on_node_activate` on all nodes in this graph.
    """
    graph = await _on_graph_activate(graph, user_id)
    graph.sub_graphs = await asyncio.gather(
        *(_on_graph_activate(sub_graph, user_id) for sub_graph in graph.sub_graphs)
    )
    return graph


@overload
async def _on_graph_activate(graph: "GraphModel", user_id: str) -> "GraphModel": ...


@overload
async def _on_graph_activate(graph: "BaseGraph", user_id: str) -> "BaseGraph": ...


async def _on_graph_activate(graph: "BaseGraph | GraphModel", user_id: str):
    get_credentials = credentials_manager.cached_getter(user_id)
    for new_node in graph.nodes:
        block_input_schema = cast(BlockSchema, new_node.block.input_schema)

        for creds_field_name in block_input_schema.get_credentials_fields().keys():
            # Prevent saving graph with non-existent credentials
            if (
                creds_meta := new_node.input_default.get(creds_field_name)
            ) and not await get_credentials(creds_meta["id"]):
                # If the credential field is optional (has a default in the
                # schema, or node metadata marks it optional), clear the stale
                # reference instead of blocking the save.
                creds_field_optional = (
                    new_node.credentials_optional
                    or creds_field_name not in block_input_schema.get_required_fields()
                )
                if creds_field_optional:
                    new_node.input_default[creds_field_name] = {}
                    logger.warning(
                        f"Node #{new_node.id}: cleared stale optional "
                        f"credentials #{creds_meta['id']} for "
                        f"'{creds_field_name}'"
                    )
                    continue
                raise ValueError(
                    f"Node #{new_node.id} input '{creds_field_name}' updated with "
                    f"non-existent credentials #{creds_meta['id']}"
                )

    return graph


async def on_graph_deactivate(graph: "GraphModel", user_id: str):
    """
    Hook to be called when a graph is deactivated/deleted.

    ⚠️ Assuming node entities are not re-used between graph versions, ⚠️
    this hook calls `on_node_deactivate` on all nodes in `graph`.
    """
    get_credentials = credentials_manager.cached_getter(user_id)
    updated_nodes = []
    for node in graph.nodes:
        block_input_schema = cast(BlockSchema, node.block.input_schema)

        node_credentials = None
        for creds_field_name in block_input_schema.get_credentials_fields().keys():
            if (creds_meta := node.input_default.get(creds_field_name)) and not (
                node_credentials := await get_credentials(creds_meta["id"])
            ):
                logger.warning(
                    f"Node #{node.id} input '{creds_field_name}' referenced "
                    f"non-existent credentials #{creds_meta['id']}"
                )

        updated_node = await on_node_deactivate(
            user_id, node, credentials=node_credentials
        )
        updated_nodes.append(updated_node)

    graph.nodes = updated_nodes
    return graph


async def on_node_deactivate(
    user_id: str,
    node: "NodeModel",
    *,
    credentials: Optional["Credentials"] = None,
    webhooks_manager: Optional["BaseWebhooksManager"] = None,
) -> "NodeModel":
    """Hook to be called when node is deactivated/deleted"""

    logger.debug(f"Deactivating node #{node.id}")
    block = node.block

    if not block.webhook_config:
        return node

    provider = block.webhook_config.provider
    if not supports_webhooks(provider):
        raise ValueError(
            f"Block #{block.id} has webhook_config for provider {provider} "
            "which does not support webhooks"
        )

    webhooks_manager = get_webhook_manager(provider)

    if webhook_id := node.webhook_id:
        logger.warning(
            f"Node #{node.id} still attached to webhook #{webhook_id} - "
            "did migration by `migrate_legacy_triggered_graphs` fail? "
            "Triggered nodes are deprecated since Significant-Gravitas/AutoGPT#10418."
        )
        webhook = await get_webhook(webhook_id)

        # Detach webhook from node
        logger.debug(f"Detaching webhook from node #{node.id}")
        updated_node = await set_node_webhook(node.id, None)

        # Prune and deregister the webhook if it is no longer used anywhere
        logger.debug(
            f"Pruning{' and deregistering' if credentials else ''} "
            f"webhook #{webhook_id}"
        )
        await webhooks_manager.prune_webhook_if_dangling(
            user_id, webhook_id, credentials
        )
        if (
            cast(BlockSchema, block.input_schema).get_credentials_fields()
            and not credentials
        ):
            logger.warning(
                f"Cannot deregister webhook #{webhook_id}: credentials "
                f"#{webhook.credentials_id} not available "
                f"({webhook.provider.value} webhook ID: {webhook.provider_webhook_id})"
            )
        return updated_node

    logger.debug(f"Node #{node.id} has no webhook_id, returning")
    return node
