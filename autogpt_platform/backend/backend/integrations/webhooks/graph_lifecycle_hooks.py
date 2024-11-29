import logging
from typing import TYPE_CHECKING, Callable, Optional, cast

from backend.data.block import get_block
from backend.data.graph import set_node_webhook
from backend.data.model import CREDENTIALS_FIELD_NAME
from backend.integrations.webhooks import WEBHOOK_MANAGERS_BY_NAME

if TYPE_CHECKING:
    from backend.data.graph import GraphModel, NodeModel
    from backend.data.model import Credentials

    from .base import BaseWebhooksManager

logger = logging.getLogger(__name__)


async def on_graph_activate(
    graph: "GraphModel", get_credentials: Callable[[str], "Credentials | None"]
):
    """
    Hook to be called when a graph is activated/created.

    ⚠️ Assuming node entities are not re-used between graph versions, ⚠️
    this hook calls `on_node_activate` on all nodes in this graph.

    Params:
        get_credentials: `credentials_id` -> Credentials
    """
    # Compare nodes in new_graph_version with previous_graph_version
    updated_nodes = []
    for new_node in graph.nodes:
        node_credentials = None
        if creds_meta := new_node.input_default.get(CREDENTIALS_FIELD_NAME):
            node_credentials = get_credentials(creds_meta["id"])
            if not node_credentials:
                raise ValueError(
                    f"Node #{new_node.id} updated with non-existent "
                    f"credentials #{node_credentials}"
                )

        updated_node = await on_node_activate(
            graph.user_id, new_node, credentials=node_credentials
        )
        updated_nodes.append(updated_node)

    graph.nodes = updated_nodes
    return graph


async def on_graph_deactivate(
    graph: "GraphModel", get_credentials: Callable[[str], "Credentials | None"]
):
    """
    Hook to be called when a graph is deactivated/deleted.

    ⚠️ Assuming node entities are not re-used between graph versions, ⚠️
    this hook calls `on_node_deactivate` on all nodes in `graph`.

    Params:
        get_credentials: `credentials_id` -> Credentials
    """
    updated_nodes = []
    for node in graph.nodes:
        node_credentials = None
        if creds_meta := node.input_default.get(CREDENTIALS_FIELD_NAME):
            node_credentials = get_credentials(creds_meta["id"])
            if not node_credentials:
                logger.error(
                    f"Node #{node.id} referenced non-existent "
                    f"credentials #{creds_meta['id']}"
                )

        updated_node = await on_node_deactivate(node, credentials=node_credentials)
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

    block = get_block(node.block_id)
    if not block:
        raise ValueError(
            f"Node #{node.id} is instance of unknown block #{node.block_id}"
        )

    if not block.webhook_config:
        return node

    logger.debug(
        f"Activating webhook node #{node.id} with config {block.webhook_config}"
    )

    webhooks_manager = WEBHOOK_MANAGERS_BY_NAME[block.webhook_config.provider]()

    try:
        resource = block.webhook_config.resource_format.format(**node.input_default)
    except KeyError:
        resource = None
    logger.debug(
        f"Constructed resource string {resource} from input {node.input_default}"
    )

    event_filter_input_name = block.webhook_config.event_filter_input
    has_everything_for_webhook = (
        resource is not None
        and CREDENTIALS_FIELD_NAME in node.input_default
        and event_filter_input_name in node.input_default
        and any(is_on for is_on in node.input_default[event_filter_input_name].values())
    )

    if has_everything_for_webhook and resource:
        logger.debug(f"Node #{node} has everything for a webhook!")
        if not credentials:
            credentials_meta = node.input_default[CREDENTIALS_FIELD_NAME]
            raise ValueError(
                f"Cannot set up webhook for node #{node.id}: "
                f"credentials #{credentials_meta['id']} not available"
            )

        # Shape of the event filter is enforced in Block.__init__
        event_filter = cast(dict, node.input_default[event_filter_input_name])
        events = [
            block.webhook_config.event_format.format(event=event)
            for event, enabled in event_filter.items()
            if enabled is True
        ]
        logger.debug(f"Webhook events to subscribe to: {', '.join(events)}")

        # Find/make and attach a suitable webhook to the node
        new_webhook = await webhooks_manager.get_suitable_webhook(
            user_id,
            credentials,
            block.webhook_config.webhook_type,
            resource,
            events,
        )
        logger.debug(f"Acquired webhook: {new_webhook}")
        return await set_node_webhook(node.id, new_webhook.id)

    return node


async def on_node_deactivate(
    node: "NodeModel",
    *,
    credentials: Optional["Credentials"] = None,
    webhooks_manager: Optional["BaseWebhooksManager"] = None,
) -> "NodeModel":
    """Hook to be called when node is deactivated/deleted"""

    logger.debug(f"Deactivating node #{node.id}")
    block = get_block(node.block_id)
    if not block:
        raise ValueError(
            f"Node #{node.id} is instance of unknown block #{node.block_id}"
        )

    if not block.webhook_config:
        return node

    webhooks_manager = WEBHOOK_MANAGERS_BY_NAME[block.webhook_config.provider]()

    if node.webhook_id:
        logger.debug(f"Node #{node.id} has webhook_id {node.webhook_id}")
        if not node.webhook:
            logger.error(f"Node #{node.id} has webhook_id but no webhook object")
            raise ValueError("node.webhook not included")

        # Detach webhook from node
        logger.debug(f"Detaching webhook from node #{node.id}")
        updated_node = await set_node_webhook(node.id, None)

        # Prune and deregister the webhook if it is no longer used anywhere
        logger.debug("Pruning and deregistering webhook if dangling")
        webhook = node.webhook
        if credentials:
            logger.debug(f"Pruning webhook #{webhook.id} with credentials")
            await webhooks_manager.prune_webhook_if_dangling(webhook.id, credentials)
        else:
            logger.warning(
                f"Cannot deregister webhook #{webhook.id}: credentials "
                f"#{webhook.credentials_id} not available "
                f"({webhook.provider} webhook ID: {webhook.provider_webhook_id})"
            )
        return updated_node

    logger.debug(f"Node #{node.id} has no webhook_id, returning")
    return node
