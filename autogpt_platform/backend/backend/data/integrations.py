import logging
from typing import TYPE_CHECKING, AsyncGenerator, Optional

from prisma import Json
from prisma.models import IntegrationWebhook
from pydantic import Field, computed_field

from backend.data.includes import INTEGRATION_WEBHOOK_INCLUDE
from backend.data.queue import AsyncRedisEventBus
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks.utils import webhook_ingress_url

from .db import BaseDbModel

if TYPE_CHECKING:
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

    attached_nodes: Optional[list["NodeModel"]] = None

    @computed_field
    @property
    def url(self) -> str:
        return webhook_ingress_url(self.provider, self.id)

    @staticmethod
    def from_db(webhook: IntegrationWebhook):
        from .graph import NodeModel

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
            attached_nodes=(
                [NodeModel.from_db(node) for node in webhook.AgentNodes]
                if webhook.AgentNodes is not None
                else None
            ),
        )


# --------------------- CRUD functions --------------------- #


async def create_webhook(webhook: Webhook) -> Webhook:
    created_webhook = await IntegrationWebhook.prisma().create(
        data={
            "id": webhook.id,
            "userId": webhook.user_id,
            "provider": webhook.provider.value,
            "credentialsId": webhook.credentials_id,
            "webhookType": webhook.webhook_type,
            "resource": webhook.resource,
            "events": webhook.events,
            "config": Json(webhook.config),
            "secret": webhook.secret,
            "providerWebhookId": webhook.provider_webhook_id,
        }
    )
    return Webhook.from_db(created_webhook)


async def get_webhook(webhook_id: str) -> Webhook:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    webhook = await IntegrationWebhook.prisma().find_unique_or_raise(
        where={"id": webhook_id},
        include=INTEGRATION_WEBHOOK_INCLUDE,
    )
    return Webhook.from_db(webhook)


async def get_all_webhooks_by_creds(credentials_id: str) -> list[Webhook]:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    if not credentials_id:
        raise ValueError("credentials_id must not be empty")
    webhooks = await IntegrationWebhook.prisma().find_many(
        where={"credentialsId": credentials_id},
        include=INTEGRATION_WEBHOOK_INCLUDE,
    )
    return [Webhook.from_db(webhook) for webhook in webhooks]


async def find_webhook_by_credentials_and_props(
    credentials_id: str, webhook_type: str, resource: str, events: list[str]
) -> Webhook | None:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    webhook = await IntegrationWebhook.prisma().find_first(
        where={
            "credentialsId": credentials_id,
            "webhookType": webhook_type,
            "resource": resource,
            "events": {"has_every": events},
        },
        include=INTEGRATION_WEBHOOK_INCLUDE,
    )
    return Webhook.from_db(webhook) if webhook else None


async def find_webhook_by_graph_and_props(
    graph_id: str, provider: str, webhook_type: str, events: list[str]
) -> Webhook | None:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    webhook = await IntegrationWebhook.prisma().find_first(
        where={
            "provider": provider,
            "webhookType": webhook_type,
            "events": {"has_every": events},
            "AgentNodes": {"some": {"agentGraphId": graph_id}},
        },
        include=INTEGRATION_WEBHOOK_INCLUDE,
    )
    return Webhook.from_db(webhook) if webhook else None


async def update_webhook_config(webhook_id: str, updated_config: dict) -> Webhook:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    _updated_webhook = await IntegrationWebhook.prisma().update(
        where={"id": webhook_id},
        data={"config": Json(updated_config)},
        include=INTEGRATION_WEBHOOK_INCLUDE,
    )
    if _updated_webhook is None:
        raise ValueError(f"Webhook #{webhook_id} not found")
    return Webhook.from_db(_updated_webhook)


async def delete_webhook(webhook_id: str) -> None:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    deleted = await IntegrationWebhook.prisma().delete(where={"id": webhook_id})
    if not deleted:
        raise ValueError(f"Webhook #{webhook_id} not found")


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
