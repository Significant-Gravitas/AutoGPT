"""
Webhook management for Meeting BaaS blocks.
"""

from enum import Enum
from typing import Tuple

from backend.sdk import (
    APIKeyCredentials,
    BaseWebhooksManager,
    Credentials,
    ProviderName,
    Webhook,
)


class BaasWebhookManager(BaseWebhooksManager):
    """Webhook manager for Meeting BaaS API."""

    PROVIDER_NAME = ProviderName("baas")

    class WebhookType(str, Enum):
        MEETING_EVENT = "meeting_event"
        CALENDAR_EVENT = "calendar_event"

    @classmethod
    async def validate_payload(cls, webhook: Webhook, request) -> Tuple[dict, str]:
        """Validate incoming webhook payload."""
        payload = await request.json()

        # Verify API key in header
        api_key_header = request.headers.get("x-meeting-baas-api-key")
        if webhook.secret and api_key_header != webhook.secret:
            raise ValueError("Invalid webhook API key")

        # Extract event type from payload
        event_type = payload.get("event", "unknown")

        return payload, event_type

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> Tuple[str, dict]:
        """
        Register webhook with Meeting BaaS.

        Note: Meeting BaaS doesn't have a webhook registration API.
        Webhooks are configured per-bot or as account defaults.
        This returns a synthetic webhook ID.
        """
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("Meeting BaaS webhooks require API key credentials")

        # Generate a synthetic webhook ID since BaaS doesn't provide one
        import uuid

        webhook_id = str(uuid.uuid4())

        return webhook_id, {
            "webhook_type": webhook_type,
            "resource": resource,
            "events": events,
            "ingress_url": ingress_url,
            "api_key": credentials.api_key.get_secret_value(),
        }

    async def _deregister_webhook(
        self, webhook: Webhook, credentials: Credentials
    ) -> None:
        """
        Deregister webhook from Meeting BaaS.

        Note: Meeting BaaS doesn't have a webhook deregistration API.
        Webhooks are removed by updating bot/calendar configurations.
        """
        # No-op since BaaS doesn't have webhook deregistration
        pass
