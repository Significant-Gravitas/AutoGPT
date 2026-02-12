import logging
from typing import TYPE_CHECKING, Optional, cast

from pydantic import JsonValue

from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.settings import Config

from . import get_webhook_manager, supports_webhooks

if TYPE_CHECKING:
    from backend.blocks._base import AnyBlockSchema
    from backend.data.integrations import Webhook
    from backend.data.model import Credentials
    from backend.integrations.providers import ProviderName

logger = logging.getLogger(__name__)
app_config = Config()
credentials_manager = IntegrationCredentialsManager()


# TODO: add test to assert this matches the actual API route
def webhook_ingress_url(provider_name: "ProviderName", webhook_id: str) -> str:
    return (
        f"{app_config.platform_base_url}/api/integrations/{provider_name.value}"
        f"/webhooks/{webhook_id}/ingress"
    )


async def setup_webhook_for_block(
    user_id: str,
    trigger_block: "AnyBlockSchema",
    trigger_config: dict[str, JsonValue],  # = Trigger block inputs
    for_graph_id: Optional[str] = None,
    for_preset_id: Optional[str] = None,
    credentials: Optional["Credentials"] = None,
) -> tuple["Webhook", None] | tuple[None, str]:
    """
    Utility function to create (and auto-setup if possible) a webhook for a given provider.

    Returns:
        Webhook: The created or found webhook object, if successful.
        str: A feedback message, if any required inputs are missing.
    """
    from backend.blocks._base import BlockWebhookConfig

    if not (trigger_base_config := trigger_block.webhook_config):
        raise ValueError(f"Block #{trigger_block.id} does not have a webhook_config")

    provider = trigger_base_config.provider
    if not supports_webhooks(provider):
        raise NotImplementedError(
            f"Block #{trigger_block.id} has webhook_config for provider {provider} "
            "for which we do not have a WebhooksManager"
        )

    logger.debug(
        f"Setting up webhook for block #{trigger_block.id} with config {trigger_config}"
    )

    # Check & parse the event filter input, if any
    events: list[str] = []
    if event_filter_input_name := trigger_base_config.event_filter_input:
        if not (event_filter := trigger_config.get(event_filter_input_name)):
            return None, (
                f"Cannot set up {provider.value} webhook without event filter input: "
                f"missing input for '{event_filter_input_name}'"
            )
        elif not (
            # Shape of the event filter is enforced in Block.__init__
            any((event_filter := cast(dict[str, bool], event_filter)).values())
        ):
            return None, (
                f"Cannot set up {provider.value} webhook without any enabled events "
                f"in event filter input '{event_filter_input_name}'"
            )

        events = [
            trigger_base_config.event_format.format(event=event)
            for event, enabled in event_filter.items()
            if enabled is True
        ]
        logger.debug(f"Webhook events to subscribe to: {', '.join(events)}")

    # Check & process prerequisites for auto-setup webhooks
    if auto_setup_webhook := isinstance(trigger_base_config, BlockWebhookConfig):
        try:
            resource = trigger_base_config.resource_format.format(**trigger_config)
        except KeyError as missing_key:
            return None, (
                f"Cannot auto-setup {provider.value} webhook without resource: "
                f"missing input for '{missing_key}'"
            )
        logger.debug(
            f"Constructed resource string {resource} from input {trigger_config}"
        )

        creds_field_name = next(
            # presence of this field is enforced in Block.__init__
            iter(trigger_block.input_schema.get_credentials_fields())
        )

        if not (
            credentials_meta := cast(dict, trigger_config.get(creds_field_name, None))
        ):
            return None, f"Cannot set up {provider.value} webhook without credentials"
        elif not (
            credentials := credentials
            or await credentials_manager.get(user_id, credentials_meta["id"])
        ):
            raise ValueError(
                f"Cannot set up {provider.value} webhook without credentials: "
                f"credentials #{credentials_meta['id']} not found for user #{user_id}"
            )
        elif credentials.provider != provider:
            raise ValueError(
                f"Credentials #{credentials.id} do not match provider {provider.value}"
            )
    else:
        # not relevant for manual webhooks:
        resource = ""
        credentials = None

    webhooks_manager = get_webhook_manager(provider)

    # Find/make and attach a suitable webhook to the node
    if auto_setup_webhook:
        assert credentials is not None
        webhook = await webhooks_manager.get_suitable_auto_webhook(
            user_id=user_id,
            credentials=credentials,
            webhook_type=trigger_base_config.webhook_type,
            resource=resource,
            events=events,
        )
    else:
        # Manual webhook -> no credentials -> don't register but do create
        webhook = await webhooks_manager.get_manual_webhook(
            user_id=user_id,
            webhook_type=trigger_base_config.webhook_type,
            events=events,
            graph_id=for_graph_id,
            preset_id=for_preset_id,
        )
    logger.debug(f"Acquired webhook: {webhook}")
    return webhook, None


async def migrate_legacy_triggered_graphs():
    from prisma.models import AgentGraph

    from backend.api.features.library.db import create_preset
    from backend.api.features.library.model import LibraryAgentPresetCreatable
    from backend.data.graph import AGENT_GRAPH_INCLUDE, GraphModel, set_node_webhook
    from backend.data.model import is_credentials_field_name

    triggered_graphs = [
        GraphModel.from_db(_graph)
        for _graph in await AgentGraph.prisma().find_many(
            where={
                "isActive": True,
                "Nodes": {"some": {"NOT": [{"webhookId": None}]}},
            },
            include=AGENT_GRAPH_INCLUDE,
        )
    ]

    n_migrated_webhooks = 0

    for graph in triggered_graphs:
        try:
            if not (
                (trigger_node := graph.webhook_input_node) and trigger_node.webhook_id
            ):
                continue

            # Use trigger node's inputs for the preset
            preset_credentials = {
                field_name: creds_meta
                for field_name, creds_meta in trigger_node.input_default.items()
                if is_credentials_field_name(field_name)
            }
            preset_inputs = {
                field_name: value
                for field_name, value in trigger_node.input_default.items()
                if not is_credentials_field_name(field_name)
            }

            # Create a triggered preset for the graph
            await create_preset(
                graph.user_id,
                LibraryAgentPresetCreatable(
                    graph_id=graph.id,
                    graph_version=graph.version,
                    inputs=preset_inputs,
                    credentials=preset_credentials,
                    name=graph.name,
                    description=graph.description,
                    webhook_id=trigger_node.webhook_id,
                    is_active=True,
                ),
            )

            # Detach webhook from the graph node
            await set_node_webhook(trigger_node.id, None)

            n_migrated_webhooks += 1
        except Exception as e:
            logger.error(f"Failed to migrate graph #{graph.id} trigger to preset: {e}")
            continue

    logger.info(f"Migrated {n_migrated_webhooks} node triggers to triggered presets")
