"""Setting up webhook-triggered presets for library agents.

Shared by the ``POST /presets/setup-trigger`` route and the copilot
``setup_agent_webhook_trigger`` tool, so the webhook-preset creation logic lives
in one place. Exposed as a DatabaseManager RPC endpoint (see ``db_manager.py``)
because the copilot tool runs without a connected Prisma client.
"""

import logging
from typing import Any

from backend.data.graph import get_graph
from backend.data.integrations import get_webhook
from backend.data.model import CredentialsMetaInput, GraphInput
from backend.executor.utils import make_node_credentials_input_map
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.webhooks import get_webhook_manager
from backend.integrations.webhooks.utils import setup_webhook_for_block
from backend.util.exceptions import InvalidInputError, NotFoundError

from . import db
from . import model as models

logger = logging.getLogger(__name__)

credentials_manager = IntegrationCredentialsManager()


async def setup_triggered_preset(
    *,
    user_id: str,
    graph_id: str,
    graph_version: int,
    name: str,
    description: str,
    trigger_config: dict[str, Any],
    agent_credentials: dict[str, CredentialsMetaInput],
) -> models.LibraryAgentPreset:
    """Create a webhook-triggered ``LibraryAgentPreset`` for the given graph.

    Sets up the webhook for the graph's trigger node — auto-registering it with
    the provider for auto-setup webhooks, or just minting the ingress URL for
    manual-setup webhooks — then creates a preset linked to that webhook. The
    returned preset has ``.webhook`` populated, so ``preset.webhook.url`` is the
    ingress URL to hand to the user for manual-setup webhooks.

    Fetches the graph itself (rather than taking a ``GraphModel``) so it can run
    as an RPC endpoint without serializing the whole graph across the boundary.

    Raises:
        NotFoundError: if the graph no longer exists / isn't accessible.
        InvalidInputError: if the graph has no webhook node, or the webhook
            backend rejects the trigger config / credentials.
    """
    graph = await get_graph(graph_id, version=graph_version, user_id=user_id)
    if not graph:
        raise NotFoundError(f"Graph #{graph_id} is not accessible (anymore)")
    if not (trigger_node := graph.webhook_input_node):
        raise InvalidInputError(
            f"Graph #{graph_id} does not have a webhook trigger node"
        )

    trigger_config_with_credentials = {
        **trigger_config,
        **(
            make_node_credentials_input_map(graph, agent_credentials).get(
                trigger_node.id
            )
            or {}
        ),
    }

    new_webhook, feedback = await setup_webhook_for_block(
        user_id=user_id,
        trigger_block=trigger_node.block,
        trigger_config=trigger_config_with_credentials,
    )
    if not new_webhook:
        raise InvalidInputError(f"Could not set up webhook: {feedback}")

    return await db.create_preset(
        user_id=user_id,
        preset=models.LibraryAgentPresetCreatable(
            graph_id=graph.id,
            graph_version=graph.version,
            name=name,
            description=description,
            inputs=trigger_config_with_credentials,
            credentials=agent_credentials,
            webhook_id=new_webhook.id,
            is_active=True,
        ),
    )


async def update_triggered_preset(
    *,
    user_id: str,
    preset_id: str,
    inputs: GraphInput | None = None,
    credentials: dict[str, CredentialsMetaInput] | None = None,
    name: str | None = None,
    description: str | None = None,
    is_active: bool | None = None,
) -> models.LibraryAgentPreset:
    """Update a preset, re-registering its webhook if the trigger config changed.

    Shared by the ``PATCH /presets/{id}`` route and the copilot ``update_preset``
    tool. When both ``inputs`` and ``credentials`` are provided and the preset's
    graph has a webhook trigger node, the webhook is re-registered with the new
    config and the previously-attached webhook is pruned if it becomes dangling.
    Name/description/active-status changes don't touch the webhook.

    Raises:
        NotFoundError: if the preset (or, when reconfiguring, its graph) is gone.
        InvalidInputError: if the webhook backend rejects the new trigger config.
    """
    current = await db.get_preset(user_id, preset_id)
    if not current:
        raise NotFoundError(f"Preset #{preset_id} not found")

    trigger_inputs_updated, new_webhook = False, None
    if inputs is not None and credentials is not None:
        graph = await get_graph(
            current.graph_id, current.graph_version, user_id=user_id
        )
        if not graph:
            raise NotFoundError(
                f"Graph #{current.graph_id} is not accessible (anymore)"
            )
        if trigger_node := graph.webhook_input_node:
            trigger_config_with_credentials = {
                **inputs,
                **(
                    make_node_credentials_input_map(graph, credentials).get(
                        trigger_node.id
                    )
                    or {}
                ),
            }
            new_webhook, feedback = await setup_webhook_for_block(
                user_id=user_id,
                trigger_block=trigger_node.block,
                trigger_config=trigger_config_with_credentials,
                for_preset_id=preset_id,
            )
            trigger_inputs_updated = True
            if not new_webhook:
                raise InvalidInputError(
                    f"Could not update trigger configuration: {feedback}"
                )

    updated = await db.update_preset(
        user_id=user_id,
        preset_id=preset_id,
        inputs=inputs,
        credentials=credentials,
        name=name,
        description=description,
        is_active=is_active,
    )

    if trigger_inputs_updated:
        new_webhook_id = new_webhook.id if new_webhook else None
        updated = await db.set_preset_webhook(user_id, preset_id, new_webhook_id)
        # Clean up the old webhook if it's no longer referenced.
        if current.webhook_id and current.webhook_id != new_webhook_id:
            await _prune_dangling_webhook(user_id, current.webhook_id)

    return updated


async def delete_preset_with_webhook_cleanup(*, user_id: str, preset_id: str) -> None:
    """Delete a preset and prune its webhook if it becomes dangling.

    Shared by the ``DELETE /presets/{id}`` route and the copilot ``delete_preset``
    tool. Detaches the webhook from the preset first, then prunes it (deregisters
    with the provider + deletes) if no other trigger references it.

    Raises:
        NotFoundError: if the preset doesn't exist for this user.
    """
    preset = await db.get_preset(user_id, preset_id)
    if not preset:
        raise NotFoundError(f"Preset #{preset_id} not found")

    if preset.webhook_id:
        await db.set_preset_webhook(user_id, preset_id, None)
        await _prune_dangling_webhook(user_id, preset.webhook_id)

    await db.delete_preset(user_id, preset_id)


async def _prune_dangling_webhook(user_id: str, webhook_id: str) -> None:
    """Deregister + delete a webhook if no trigger references it anymore.

    Best-effort: webhook cleanup runs *after* the preset update/delete is
    committed, so a failure here (e.g. provider deregister error) must not fail
    the whole mutation — it would leave the preset state correct but raise to the
    caller. We log and move on; at worst the webhook lingers and is pruned later.
    """
    try:
        webhook = await get_webhook(webhook_id)
        credentials = (
            await credentials_manager.get(user_id, webhook.credentials_id)
            if webhook.credentials_id
            else None
        )
        await get_webhook_manager(webhook.provider).prune_webhook_if_dangling(
            user_id, webhook.id, credentials
        )
    except Exception as e:
        logger.warning(f"Best-effort prune of webhook #{webhook_id} failed: {e}")
