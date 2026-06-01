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


class GraphActivationError(Exception):
    """Raised when a graph cannot be activated (e.g. a required credential is
    missing, revoked, or its OAuth token can no longer be refreshed).

    Callers in the API layer should map this to HTTP 400 so the user sees a
    clear, actionable message instead of an opaque 500.
    """


async def before_graph_activate(graph: "GraphModel", user_id: str) -> "GraphModel":
    """
    Pre-activation hook: validates node credentials and clears stale optional
    credential references in-memory. MUST be called BEFORE the graph is
    persisted (and before it is marked active) — a failure here means nothing
    should be saved, and the returned graph carries cleanup mutations that
    need to be persisted by the caller.

    Do not use this for post-activation state changes; it has no DB writes
    and is intentionally side-effect-free outside the passed-in graph object.

    ⚠️ Assumes node entities are not re-used between graph versions. ⚠️

    Raises:
        GraphActivationError: when a required node credential is missing or
            unusable.
    """
    graph = await _before_graph_activate(graph, user_id)
    graph.sub_graphs = await asyncio.gather(
        *(_before_graph_activate(sub_graph, user_id) for sub_graph in graph.sub_graphs)
    )
    return graph


@overload
async def _before_graph_activate(graph: "GraphModel", user_id: str) -> "GraphModel": ...


@overload
async def _before_graph_activate(graph: "BaseGraph", user_id: str) -> "BaseGraph": ...


async def _before_graph_activate(graph: "BaseGraph | GraphModel", user_id: str):
    get_credentials = credentials_manager.cached_getter(user_id)
    for new_node in graph.nodes:
        block_input_schema = cast(BlockSchema, new_node.block.input_schema)

        for creds_field_name in block_input_schema.get_credentials_fields().keys():
            creds_meta = new_node.input_default.get(creds_field_name)
            if not creds_meta:
                continue

            # Treat a credential as unusable both when it is missing from the
            # DB (get_credentials returns None) and when loading it raised —
            # e.g. an OAuth refresh that fails with `invalid_grant` because
            # the refresh token has been revoked. Surfacing that as a 500 is
            # unhelpful; we want a clear "please reconnect" message.
            refresh_error: str | None = None
            try:
                resolved = await get_credentials(creds_meta["id"])
            except Exception as e:
                logger.warning(
                    f"Node #{new_node.id}: failed to load credentials "
                    f"#{creds_meta['id']} for '{creds_field_name}': {e!r}"
                )
                resolved = None
                refresh_error = str(e) or e.__class__.__name__

            if resolved:
                continue

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

            if refresh_error:
                raise GraphActivationError(
                    f"Credential #{creds_meta['id']} for '{creds_field_name}' "
                    f"on node #{new_node.id} could not be loaded "
                    f"({refresh_error}). It may have been revoked or its "
                    "access expired — please reconnect this integration and "
                    "try again."
                )
            raise GraphActivationError(
                f"Credential #{creds_meta['id']} for '{creds_field_name}' on "
                f"node #{new_node.id} no longer exists. Please pick a "
                "different credential and try again."
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
