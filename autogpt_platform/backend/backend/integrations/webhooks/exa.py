import logging

import requests
from fastapi import Request

from backend.data import integrations
from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks._base import BaseWebhooksManager

logger = logging.getLogger(__name__)


class ExaWebhooksManager(BaseWebhooksManager):
    """Manager for Exa webhooks"""

    PROVIDER_NAME = ProviderName.EXA
    BASE_URL = "https://api.exa.ai/websets/v0"

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        """Register a new webhook with Exa"""

        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("API key is required to register a webhook")

        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

        payload = {
            "events": events,
            "url": ingress_url,
            "metadata": {}  # Optional metadata can be added here
        }

        response = requests.post(
            f"{self.BASE_URL}/webhooks", headers=headers, json=payload
        )

        if not response.ok:
            error = response.json().get("error", "Unknown error")
            raise RuntimeError(f"Failed to register webhook: {error}")

        response_data = response.json()
        webhook_id = response_data.get("id", "")
        
        webhook_config = {
            "endpoint": ingress_url,
            "provider": self.PROVIDER_NAME,
            "events": events,
            "type": webhook_type,
            "webhook_id": webhook_id,
            "secret": response_data.get("secret", "")
        }

        return webhook_id, webhook_config

    @classmethod
    async def validate_payload(
        cls, webhook: integrations.Webhook, request: Request
    ) -> tuple[dict, str]:
        """Validate incoming webhook payload from Exa"""

        payload = await request.json()

        # Validate required fields from Exa API spec
        required_fields = ["id", "object", "type", "data", "createdAt"]
        missing_fields = [field for field in required_fields if field not in payload]

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Normalize payload structure
        normalized_payload = {
            "id": payload["id"],
            "type": payload["type"],
            "data": payload["data"],
            "createdAt": payload["createdAt"]
        }

        # Extract event type from the payload
        event_type = payload["type"]

        return normalized_payload, event_type

    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None:
        """Deregister a webhook with Exa"""
        
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("API key is required to deregister a webhook")
            
        webhook_id = webhook.config.get("webhook_id")
        if not webhook_id:
            logger.warning(f"No webhook ID found for webhook {webhook.id}, cannot deregister")
            return
            
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }
        
        response = requests.delete(
            f"{self.BASE_URL}/webhooks/{webhook_id}", headers=headers
        )
        
        if not response.ok:
            error = response.json().get("error", "Unknown error")
            logger.error(f"Failed to deregister webhook {webhook_id}: {error}")
            raise RuntimeError(f"Failed to deregister webhook: {error}")
