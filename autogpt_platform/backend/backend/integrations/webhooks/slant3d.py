import logging

import requests
from fastapi import Request

from backend.data import integrations
from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks._base import BaseWebhooksManager

logger = logging.getLogger(__name__)


class Slant3DWebhooksManager(BaseWebhooksManager):
    """Manager for Slant3D webhooks"""

    PROVIDER_NAME = ProviderName.SLANT3D
    BASE_URL = "https://www.slant3dapi.com/api"

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        """Register a new webhook with Slant3D"""

        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("API key is required to register a webhook")

        headers = {
            "api-key": credentials.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

        # Slant3D's API doesn't use events list, just register for all order updates
        payload = {"endPoint": ingress_url}

        response = requests.post(
            f"{self.BASE_URL}/customer/webhookSubscribe", headers=headers, json=payload
        )

        if not response.ok:
            error = response.json().get("error", "Unknown error")
            raise RuntimeError(f"Failed to register webhook: {error}")

        webhook_config = {
            "endpoint": ingress_url,
            "provider": self.PROVIDER_NAME,
            "events": ["order.shipped"],  # Currently the only supported event
            "type": webhook_type,
        }

        return "", webhook_config

    @classmethod
    async def validate_payload(
        cls, webhook: integrations.Webhook, request: Request
    ) -> tuple[dict, str]:
        """Validate incoming webhook payload from Slant3D"""

        payload = await request.json()

        # Validate required fields from Slant3D API spec
        required_fields = ["orderId", "status", "trackingNumber", "carrierCode"]
        missing_fields = [field for field in required_fields if field not in payload]

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Normalize payload structure
        normalized_payload = {
            "orderId": payload["orderId"],
            "status": payload["status"],
            "trackingNumber": payload["trackingNumber"],
            "carrierCode": payload["carrierCode"],
        }

        # Currently Slant3D only sends shipping notifications
        # Convert status to lowercase for event format compatibility
        event_type = f"order.{payload['status'].lower()}"

        return normalized_payload, event_type

    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None:
        """
        Note: Slant3D API currently doesn't provide a deregistration endpoint.
        This would need to be handled through support.
        """
        # Log warning since we can't properly deregister
        logger.warning(
            f"Warning: Manual deregistration required for webhook {webhook.id}"
        )
        pass
