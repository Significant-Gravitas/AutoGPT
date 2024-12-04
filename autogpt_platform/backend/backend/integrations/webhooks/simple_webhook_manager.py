import logging

from fastapi import Request
from strenum import StrEnum

from backend.data import integrations
from backend.data.model import APIKeyCredentials, Credentials, OAuth2Credentials

from .base import BaseWebhooksManager

logger = logging.getLogger(__name__)


class CompassWebhookType(StrEnum):
    TRANSCRIPTION = "transcription"
    TASK = "task"


class SimpleWebhooksManager(BaseWebhooksManager):
    PROVIDER_NAME = "simple_hook_manager"

    DEFAULT_HEADERS = {"Accept": "application/json"}

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: CompassWebhookType,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        print(ingress_url)

        return "", {}

    async def _deregister_webhook(
        self,
        webhook: integrations.Webhook,
        credentials: OAuth2Credentials | APIKeyCredentials,
    ) -> None:
        pass


class CompassWebhookManager(SimpleWebhooksManager):
    WebhookType = CompassWebhookType
    PROVIDER_NAME = "compass"

    @classmethod
    async def validate_payload(
        cls, webhook: integrations.Webhook, request: Request
    ) -> tuple[dict, str]:
        payload = await request.json()
        event_type = CompassWebhookType.TRANSCRIPTION  # currently the only type

        return payload, event_type
