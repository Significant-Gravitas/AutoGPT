import logging
from typing import TYPE_CHECKING, Optional, cast

from backend.data.block import BlockSchema
from backend.data.graph import set_node_webhook
from backend.integrations.creds_manager import IntegrationCredentialsManager

from . import get_webhook_manager, supports_webhooks
from .utils import setup_webhook_for_block

if TYPE_CHECKING:
    from backend.data.graph import GraphModel, NodeModel
    from backend.data.model import Credentials

    from ._base import BaseWebhooksManager

logger = logging.getLogger(__name__)
credentials_manager = IntegrationCredentialsManager()


async def on_graph_activate(graph: "GraphModel", user_id: str):
    """
    Hook to be called when a graph is activated/created.

    ⚠️ Assuming node entities are not re-used between graph versions, ⚠️
    this hook calls `on_node_activate` on all nodes in this graph.
    """
    get_credentials = credentials_manager.cached_getter(user_id)
    updated_nodes = []
    for new_node in graph.nodes:
        block_input_schema = cast(BlockSchema, new_node.block.input_schema)

        node_credentials = None
        if (
            # Webhook-triggered blocks are only allowed to have 1 credentials input
            (
                creds_field_name := next(
                    iter(block_input_schema.get_credentials_fields()), None
                )
            )
            and (creds_meta := new_node.input_default.get(creds_field_name))
            and not (node_credentials := await get_credentials(creds_meta["id"]))
        ):
            raise ValueError(
                f"Node #{new_node.id} input '{creds_field_name}' updated with "
                f"non-existent credentials #{creds_meta['id']}"
            )

        updated_node = await on_node_activate(
            graph.user_id, new_node, credentials=node_credentials
        )
        updated_nodes.append(updated_node)

    graph.nodes = updated_nodes
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
        if (
            # Webhook-triggered blocks are only allowed to have 1 credentials input
            (
                creds_field_name := next(
                    iter(block_input_schema.get_credentials_fields()), None
                )
            )
            and (creds_meta := node.input_default.get(creds_field_name))
            and not (node_credentials := await get_credentials(creds_meta["id"]))
        ):
            logger.error(
                f"Node #{node.id} input '{creds_field_name}' referenced non-existent "
                f"credentials #{creds_meta['id']}"
            )

        updated_node = await on_node_deactivate(
            user_id, node, credentials=node_credentials
        )
        updated_nodes.append(updated_node)

    graph.nodes = updated_nodes
    return graph


async def on_node_activate(
    user_id: str,
    node: "NodeModel",
    *,
    credentials: Optional["Credentials"] = None,
) -> "NodeModel":
    """Hook to be called when the node is activated/created"""

    if node.block.webhook_config:
        new_webhook, feedback = await setup_webhook_for_block(
            user_id=user_id,
            trigger_block=node.block,
            trigger_config=node.input_default,
            for_graph_id=node.graph_id,
        )
        if new_webhook:
            node = await set_node_webhook(node.id, new_webhook.id)
        else:
            logger.debug(
                f"Node #{node.id} does not have everything for a webhook: {feedback}"
            )

    return node


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

    if node.webhook_id:
        logger.debug(f"Node #{node.id} has webhook_id {node.webhook_id}")
        if not node.webhook:
            logger.error(f"Node #{node.id} has webhook_id but no webhook object")
            raise ValueError("node.webhook not included")

        # Detach webhook from node
        logger.debug(f"Detaching webhook from node #{node.id}")
        updated_node = await set_node_webhook(node.id, None)

        # Prune and deregister the webhook if it is no longer used anywhere
        webhook = node.webhook
        logger.debug(
            f"Pruning{' and deregistering' if credentials else ''} "
            f"webhook #{webhook.id}"
        )
        await webhooks_manager.prune_webhook_if_dangling(
            user_id, webhook.id, credentials
        )
        if (
            cast(BlockSchema, block.input_schema).get_credentials_fields()
            and not credentials
        ):
            logger.warning(
                f"Cannot deregister webhook #{webhook.id}: credentials "
                f"#{webhook.credentials_id} not available "
                f"({webhook.provider.value} webhook ID: {webhook.provider_webhook_id})"
            )
        return updated_node

    logger.debug(f"Node #{node.id} has no webhook_id, returning")
    return node
