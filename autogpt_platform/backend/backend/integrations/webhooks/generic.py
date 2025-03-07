import logging

from fastapi import Request
from strenum import StrEnum

from backend.data import integrations
from backend.integrations.providers import ProviderName

from ._manual_base import ManualWebhookManagerBase

logger = logging.getLogger(__name__)


class GenericWebhookType(StrEnum):
    PLAIN = "plain"


class GenericWebhooksManager(ManualWebhookManagerBase):
    PROVIDER_NAME = ProviderName.GENERIC_WEBHOOK
    WebhookType = GenericWebhookType

    @classmethod
    async def validate_payload(
        cls, webhook: integrations.Webhook, request: Request
    ) -> tuple[dict, str]:
        payload = await request.json()
        event_type = GenericWebhookType.PLAIN

        return payload, event_type
