import asyncio
import logging
from typing import TYPE_CHECKING, Optional, cast, overload

from backend.blocks._base import BlockSchema
from backend.data.graph import set_node_webhook
from backend.data.integrations import get_webhook
from backend.integrations.creds_manager import IntegrationCredentialsManager

from . import get_webhook_manager, supports_webhooks

if TYPE_CHECKING:
    from backend.data.graph import BaseGraph, GraphModel, Node, NodeModel
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

    # Collect every (node, field) credential reference up front so we can
    # resolve them in one parallel batch — important when a graph has
    # several distinct OAuth credentials that each need a refresh round-trip.
    refs: list[tuple["Node | NodeModel", str, dict, BlockSchema]] = []
    for new_node in graph.nodes:
        block_input_schema = cast(BlockSchema, new_node.block.input_schema)
        for creds_field_name in block_input_schema.get_credentials_fields().keys():
            creds_meta = new_node.input_default.get(creds_field_name)
            if not creds_meta:
                continue
            refs.append((new_node, creds_field_name, creds_meta, block_input_schema))

    unique_ids = list({m["id"] for _, _, m, _ in refs})
    results = await asyncio.gather(
        *(get_credentials(cid) for cid in unique_ids),
        return_exceptions=True,
    )
    cred_by_id: dict[str, "Credentials | None | BaseException"] = dict(
        zip(unique_ids, results)
    )

    for new_node, creds_field_name, creds_meta, block_input_schema in refs:
        _apply_credential_result(
            new_node,
            creds_field_name,
            creds_meta,
            block_input_schema,
            cred_by_id[creds_meta["id"]],
        )

    return graph


def _apply_credential_result(
    new_node: "Node | NodeModel",
    creds_field_name: str,
    creds_meta: dict,
    block_input_schema: BlockSchema,
    result: "Credentials | None | BaseException",
) -> None:
    """Apply the resolution outcome for one credential reference: leave
    usable ones alone, clear stale optional ones in-memory, or raise
    `GraphActivationError` for required + unusable ones.

    Treats both `None` (credential missing from DB) and an exception
    (OAuth refresh raised, infra error) as "unusable" — failures are
    logged here so the caller doesn't have to.
    """
    refresh_error: str | None = None
    if isinstance(result, BaseException):
        # Distinguish known credential-side failures (OAuth refresh rejected,
        # 401/403 from the provider) from infra failures (DB pool exhaustion,
        # Redis timeout, TypeError, …) so the latter show up at error level
        # with a stack trace instead of being silently misreported as
        # "please reconnect".
        error_str = repr(result).lower()
        is_known_credential_error = any(
            sig in error_str
            for sig in (
                "invalid_grant",
                "invalid_token",
                "unauthorized",
                "forbidden",
                " 401",
                " 403",
            )
        )
        log_message = (
            f"Node #{new_node.id}: failed to load credentials "
            f"#{creds_meta['id']} for '{creds_field_name}': {result!r}"
        )
        if is_known_credential_error:
            logger.warning(log_message)
        else:
            logger.error(log_message, exc_info=result)
        refresh_error = str(result) or result.__class__.__name__
        resolved = None
    else:
        resolved = result

    if resolved:
        return

    # If the credential field is optional (has a default in the schema, or
    # node metadata marks it optional), clear the stale reference instead
    # of blocking the save.
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
        return

    # User-facing reference: prefer the credential's user-set title and the
    # block name over internal UUIDs, since users can't act on them. UUIDs
    # still appear in the warning/error log above for support lookups.
    credential_label = (
        f'"{creds_meta["title"]}" {creds_meta["provider"]}'
        if creds_meta.get("title")
        else creds_meta.get("provider", "unknown")
    )
    credential_ref = (
        f"The {credential_label} credential used by the " f"{new_node.block.name} node"
    )

    if refresh_error:
        raise GraphActivationError(
            f"{credential_ref} could not be loaded ({refresh_error}). "
            "It may have been revoked or its access expired — please "
            "reconnect this integration and try again."
        )
    raise GraphActivationError(
        f"{credential_ref} no longer exists. Please pick a different "
        "credential and try again."
    )


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
