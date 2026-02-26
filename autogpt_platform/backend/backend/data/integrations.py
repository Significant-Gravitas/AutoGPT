import logging
from typing import AsyncGenerator, Literal, Optional, overload

from prisma.models import AgentNode, AgentPreset, IntegrationWebhook
from prisma.types import (
    IntegrationWebhookCreateInput,
    IntegrationWebhookUpdateInput,
    IntegrationWebhookWhereInput,
    Serializable,
)
from pydantic import Field, computed_field

from backend.data.event_bus import AsyncRedisEventBus
from backend.data.includes import (
    INTEGRATION_WEBHOOK_INCLUDE,
    MAX_INTEGRATION_WEBHOOKS_FETCH,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks import get_webhook_manager
from backend.integrations.webhooks.utils import webhook_ingress_url
from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson

from .db import BaseDbModel
from .graph import NodeModel

logger = logging.getLogger(__name__)


class Webhook(BaseDbModel):
    user_id: str
    provider: ProviderName
    credentials_id: str
    webhook_type: str
    resource: str
    events: list[str]
    config: dict = Field(default_factory=dict)
    secret: str

    provider_webhook_id: str

    @computed_field
    @property
    def url(self) -> str:
        return webhook_ingress_url(self.provider, self.id)

    @staticmethod
    def from_db(webhook: IntegrationWebhook):
        return Webhook(
            id=webhook.id,
            user_id=webhook.userId,
            provider=ProviderName(webhook.provider),
            credentials_id=webhook.credentialsId,
            webhook_type=webhook.webhookType,
            resource=webhook.resource,
            events=webhook.events,
            config=dict(webhook.config),
            secret=webhook.secret,
            provider_webhook_id=webhook.providerWebhookId,
        )


# LibraryAgentPreset import must be after Webhook definition to avoid
# broken circular import:
# integrations.py → library/model.py → integrations.py (for Webhook)
from backend.api.features.library.model import LibraryAgentPreset  # noqa: E402

# Resolve forward refs
LibraryAgentPreset.model_rebuild()


class WebhookWithRelations(Webhook):
    triggered_nodes: list[NodeModel]
    triggered_presets: list[LibraryAgentPreset]

    @staticmethod
    def from_db(webhook: IntegrationWebhook):
        if webhook.AgentNodes is None or webhook.AgentPresets is None:
            raise ValueError(
                "AgentNodes and AgentPresets must be included in "
                "IntegrationWebhook query with relations"
            )

        return WebhookWithRelations(
            **Webhook.from_db(webhook).model_dump(),
            triggered_nodes=[NodeModel.from_db(node) for node in webhook.AgentNodes],
            triggered_presets=[
                LibraryAgentPreset.from_db(preset) for preset in webhook.AgentPresets
            ],
        )


# --------------------- CRUD functions --------------------- #


async def create_webhook(webhook: Webhook) -> Webhook:
    created_webhook = await IntegrationWebhook.prisma().create(
        data=IntegrationWebhookCreateInput(
            id=webhook.id,
            userId=webhook.user_id,
            provider=webhook.provider.value,
            credentialsId=webhook.credentials_id,
            webhookType=webhook.webhook_type,
            resource=webhook.resource,
            events=webhook.events,
            config=SafeJson(webhook.config),
            secret=webhook.secret,
            providerWebhookId=webhook.provider_webhook_id,
        )
    )
    return Webhook.from_db(created_webhook)


@overload
async def get_webhook(
    webhook_id: str, *, include_relations: Literal[True]
) -> WebhookWithRelations: ...
@overload
async def get_webhook(
    webhook_id: str, *, include_relations: Literal[False] = False
) -> Webhook: ...


async def get_webhook(
    webhook_id: str, *, include_relations: bool = False
) -> Webhook | WebhookWithRelations:
    """
    ⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints.

    Raises:
        NotFoundError: if no record with the given ID exists
    """
    webhook = await IntegrationWebhook.prisma().find_unique(
        where={"id": webhook_id},
        include=INTEGRATION_WEBHOOK_INCLUDE if include_relations else None,
    )
    if not webhook:
        raise NotFoundError(f"Webhook #{webhook_id} not found")
    return (WebhookWithRelations if include_relations else Webhook).from_db(webhook)


@overload
async def get_all_webhooks_by_creds(
    user_id: str,
    credentials_id: str,
    *,
    include_relations: Literal[True],
    limit: int = MAX_INTEGRATION_WEBHOOKS_FETCH,
) -> list[WebhookWithRelations]: ...
@overload
async def get_all_webhooks_by_creds(
    user_id: str,
    credentials_id: str,
    *,
    include_relations: Literal[False] = False,
    limit: int = MAX_INTEGRATION_WEBHOOKS_FETCH,
) -> list[Webhook]: ...


async def get_all_webhooks_by_creds(
    user_id: str,
    credentials_id: str,
    *,
    include_relations: bool = False,
    limit: int = MAX_INTEGRATION_WEBHOOKS_FETCH,
) -> list[Webhook] | list[WebhookWithRelations]:
    if not credentials_id:
        raise ValueError("credentials_id must not be empty")
    webhooks = await IntegrationWebhook.prisma().find_many(
        where={"userId": user_id, "credentialsId": credentials_id},
        include=INTEGRATION_WEBHOOK_INCLUDE if include_relations else None,
        order={"createdAt": "desc"},
        take=limit,
    )
    return [
        (WebhookWithRelations if include_relations else Webhook).from_db(webhook)
        for webhook in webhooks
    ]


async def find_webhook_by_credentials_and_props(
    user_id: str,
    credentials_id: str,
    webhook_type: str,
    resource: str,
    events: Optional[list[str]],
) -> Webhook | None:
    webhook = await IntegrationWebhook.prisma().find_first(
        where={
            "userId": user_id,
            "credentialsId": credentials_id,
            "webhookType": webhook_type,
            "resource": resource,
            **({"events": {"has_every": events}} if events else {}),
        },
    )
    return Webhook.from_db(webhook) if webhook else None


async def find_webhook_by_graph_and_props(
    user_id: str,
    provider: str,
    webhook_type: str,
    graph_id: Optional[str] = None,
    preset_id: Optional[str] = None,
) -> Webhook | None:
    """Either `graph_id` or `preset_id` must be provided."""
    where_clause: IntegrationWebhookWhereInput = {
        "userId": user_id,
        "provider": provider,
        "webhookType": webhook_type,
    }

    if preset_id:
        where_clause["AgentPresets"] = {"some": {"id": preset_id}}
    elif graph_id:
        where_clause["AgentNodes"] = {"some": {"agentGraphId": graph_id}}
    else:
        raise ValueError("Either graph_id or preset_id must be provided")

    webhook = await IntegrationWebhook.prisma().find_first(
        where=where_clause,
    )
    return Webhook.from_db(webhook) if webhook else None


async def update_webhook(
    webhook_id: str,
    config: Optional[dict[str, Serializable]] = None,
    events: Optional[list[str]] = None,
) -> Webhook:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    data: IntegrationWebhookUpdateInput = {}
    if config is not None:
        data["config"] = SafeJson(config)
    if events is not None:
        data["events"] = events
    if not data:
        raise ValueError("Empty update query")

    _updated_webhook = await IntegrationWebhook.prisma().update(
        where={"id": webhook_id},
        data=data,
    )
    if _updated_webhook is None:
        raise NotFoundError(f"Webhook #{webhook_id} not found")
    return Webhook.from_db(_updated_webhook)


async def find_webhooks_by_graph_id(graph_id: str, user_id: str) -> list[Webhook]:
    """
    Find all webhooks that trigger nodes OR presets in a specific graph for a user.

    Args:
        graph_id: The ID of the graph
        user_id: The ID of the user

    Returns:
        list[Webhook]: List of webhooks associated with the graph
    """
    where_clause: IntegrationWebhookWhereInput = {
        "userId": user_id,
        "OR": [
            # Webhooks that trigger nodes in this graph
            {"AgentNodes": {"some": {"agentGraphId": graph_id}}},
            # Webhooks that trigger presets for this graph
            {"AgentPresets": {"some": {"agentGraphId": graph_id}}},
        ],
    }
    webhooks = await IntegrationWebhook.prisma().find_many(where=where_clause)
    return [Webhook.from_db(webhook) for webhook in webhooks]


async def unlink_webhook_from_graph(
    webhook_id: str, graph_id: str, user_id: str
) -> None:
    """
    Unlink a webhook from all nodes and presets in a specific graph.
    If the webhook has no remaining triggers, it will be automatically deleted
    and deregistered with the provider.

    Args:
        webhook_id: The ID of the webhook
        graph_id: The ID of the graph to unlink from
        user_id: The ID of the user (for authorization)
    """
    # Avoid circular imports
    from backend.api.features.library.db import set_preset_webhook
    from backend.data.graph import set_node_webhook

    # Find all nodes in this graph that use this webhook
    nodes = await AgentNode.prisma().find_many(
        where={"agentGraphId": graph_id, "webhookId": webhook_id}
    )

    # Unlink webhook from each node
    for node in nodes:
        await set_node_webhook(node.id, None)

    # Find all presets for this graph that use this webhook
    presets = await AgentPreset.prisma().find_many(
        where={"agentGraphId": graph_id, "webhookId": webhook_id, "userId": user_id}
    )

    # Unlink webhook from each preset
    for preset in presets:
        await set_preset_webhook(user_id, preset.id, None)

    # Check if webhook needs cleanup (prune_webhook_if_dangling handles the trigger check)
    webhook = await get_webhook(webhook_id, include_relations=False)
    webhook_manager = get_webhook_manager(webhook.provider)
    creds_manager = IntegrationCredentialsManager()
    credentials = (
        await creds_manager.get(user_id, webhook.credentials_id)
        if webhook.credentials_id
        else None
    )
    await webhook_manager.prune_webhook_if_dangling(user_id, webhook.id, credentials)


async def delete_webhook(user_id: str, webhook_id: str) -> None:
    deleted = await IntegrationWebhook.prisma().delete_many(
        where={"id": webhook_id, "userId": user_id}
    )
    if deleted < 1:
        raise NotFoundError(f"Webhook #{webhook_id} not found")


# --------------------- WEBHOOK EVENTS --------------------- #


class WebhookEvent(BaseDbModel):
    provider: str
    webhook_id: str
    event_type: str
    payload: dict


class WebhookEventBus(AsyncRedisEventBus[WebhookEvent]):
    Model = WebhookEvent

    @property
    def event_bus_name(self) -> str:
        return "webhooks"


_webhook_event_bus = WebhookEventBus()


async def publish_webhook_event(event: WebhookEvent):
    await _webhook_event_bus.publish_event(
        event, f"{event.webhook_id}/{event.event_type}"
    )


async def listen_for_webhook_events(
    webhook_id: str, event_type: Optional[str] = None
) -> AsyncGenerator[WebhookEvent, None]:
    async for event in _webhook_event_bus.listen_events(
        f"{webhook_id}/{event_type or '*'}"
    ):
        yield event


async def wait_for_webhook_event(
    webhook_id: str, event_type: Optional[str] = None, timeout: Optional[float] = None
) -> WebhookEvent | None:
    return await _webhook_event_bus.wait_for_event(
        f"{webhook_id}/{event_type or '*'}", timeout
    )
