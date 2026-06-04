"""Setting up webhook-triggered presets for library agents.

Shared by the ``POST /presets/setup-trigger`` route and the copilot
``setup_agent_webhook_trigger`` tool, so the webhook-preset creation logic lives
in one place. Exposed as a DatabaseManager RPC endpoint (see ``db_manager.py``)
because the copilot tool runs without a connected Prisma client.
"""

import logging
from typing import Any

from backend.data.graph import get_graph
from backend.data.model import CredentialsMetaInput
from backend.executor.utils import make_node_credentials_input_map
from backend.integrations.webhooks.utils import setup_webhook_for_block
from backend.util.exceptions import InvalidInputError, NotFoundError

from . import db
from . import model as models

logger = logging.getLogger(__name__)


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
