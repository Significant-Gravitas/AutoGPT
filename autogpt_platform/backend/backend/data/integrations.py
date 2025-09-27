import logging
from typing import AsyncGenerator, Literal, Optional, overload

from prisma.models import IntegrationWebhook
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
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks.utils import webhook_ingress_url
from backend.server.v2.library.model import LibraryAgentPreset
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
    events: list[str],
) -> Webhook | None:
    webhook = await IntegrationWebhook.prisma().find_first(
        where={
            "userId": user_id,
            "credentialsId": credentials_id,
            "webhookType": webhook_type,
            "resource": resource,
            "events": {"has_every": events},
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
