import hashlib
import hmac
import logging
from typing import Generic, TypeVar
import uuid

import requests
from autogpt_libs.supabase_integration_credentials_store import Credentials
from fastapi import HTTPException, Request
from strenum import StrEnum

from backend.data import integrations

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

        webhook_id = uuid.uuid4()
        config = {}

        return str(webhook_id), config

    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None:
        if webhook.credentials_id != credentials.id:
            raise ValueError(
                f"Webhook #{webhook.id} does not belong to credentials {credentials.id}"
            )
        # If we reach here, the webhook was successfully deleted or didn't exist


class CompassWebhookManager(SimpleWebhooksManager):
    WebhookType = CompassWebhookType
    PROVIDER_NAME = "compass"

    @classmethod
    async def validate_payload(
        cls, webhook: integrations.Webhook, request: Request
    ) -> tuple[dict, str]:
        payload = await request.json()
        event_type = CompassWebhookType.TRANSCRIPTION

        return payload, event_type
