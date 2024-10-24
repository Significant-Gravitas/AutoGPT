import asyncio
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, cast

from prisma import Json
from prisma.models import IntegrationWebhook
from prisma.types import IntegrationWebhookInclude
from pydantic import Field

from backend.data import redis
from backend.integrations.providers import ProviderName

from .db import BaseDbModel

if TYPE_CHECKING:
    from .graph import NodeModel

logger = logging.getLogger(__name__)


class Webhook(BaseDbModel):
    user_id: str
    provider: str
    credentials_id: str
    webhook_type: str
    resource: str
    events: list[str]
    config: dict = Field(default_factory=dict)
    secret: str

    provider_webhook_id: str

    attached_nodes: Optional[list["NodeModel"]] = None

    @staticmethod
    def from_db(webhook: IntegrationWebhook):
        from .graph import NodeModel

        return Webhook(
            id=webhook.id,
            user_id=webhook.userId,
            provider=webhook.provider,
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


INTEGRATION_WEBHOOK_INCLUDE: IntegrationWebhookInclude = {"AgentNodes": True}


# --------------------- CRUD functions --------------------- #


async def create_webhook(webhook: Webhook) -> Webhook:
    created_webhook = await IntegrationWebhook.prisma().create(
        data={
            "id": webhook.id,
            "userId": webhook.user_id,
            "provider": webhook.provider,
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
    webhook = await IntegrationWebhook.prisma().find_unique_or_raise(
        where={"id": webhook_id},
        include=INTEGRATION_WEBHOOK_INCLUDE,
    )
    return Webhook.from_db(webhook)


async def find_webhook(
    credentials_id: str, webhook_type: str, resource: str, events: list[str]
) -> Webhook | None:
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


async def update_webhook_config(webhook_id: str, updated_config: dict) -> Webhook:
    _updated_webhook = await IntegrationWebhook.prisma().update(
        where={"id": webhook_id},
        data={"config": Json(updated_config)},
        include=INTEGRATION_WEBHOOK_INCLUDE,
    )
    if _updated_webhook is None:
        raise ValueError(f"Webhook #{webhook_id} not found")
    return Webhook.from_db(_updated_webhook)


async def delete_webhook(webhook_id: str) -> None:
    deleted = await IntegrationWebhook.prisma().delete(where={"id": webhook_id})
    if not deleted:
        raise ValueError(f"Webhook #{webhook_id} not found")


# --------------------- WEBHOOK EVENTS --------------------- #


class WebhookEvent(BaseDbModel):
    provider: ProviderName
    webhook_id: str
    event_type: str
    payload: dict


def publish_webhook_event(event: WebhookEvent):
    n_receipts = redis.get_redis().publish(
        channel=f"webhooks/{event.webhook_id}/{event.event_type}",
        message=event.model_dump_json(),
    )
    return cast(int, n_receipts)


def wait_for_webhook_event(
    webhook_id: str, timeout: Optional[float] = None, event_type: Optional[str] = None
) -> WebhookEvent:
    redis_client = redis.get_redis()
    pubsub = redis_client.pubsub()
    channel = f"webhooks/{webhook_id}"
    if event_type:
        channel += f"/{event_type}"
        pubsub.subscribe(channel)
    else:
        channel += "/*"
        pubsub.psubscribe(channel)

    try:
        while message := pubsub.get_message(
            ignore_subscribe_messages=True,
            timeout=timeout,  # type: ignore - timeout param isn't typed correctly
        ):
            if message["type"] == "message":
                return WebhookEvent.model_validate_json(message["data"])
    finally:
        pubsub.unsubscribe(channel)

    raise TimeoutError(f"No message arrived on {channel} after {timeout} seconds")


@contextmanager
def listen_for_webhook_event(
    webhook_id: str, timeout: Optional[float] = None, event_type: Optional[str] = None
):
    webhook_event = asyncio.Future()
    yield webhook_event
    try:
        result = wait_for_webhook_event(webhook_id, timeout, event_type)
        webhook_event.set_result(result)
    except TimeoutError as e:
        webhook_event.set_exception(e)
