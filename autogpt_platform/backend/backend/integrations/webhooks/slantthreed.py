from typing import Dict, Tuple, List, ClassVar
from autogpt_libs.supabase_integration_credentials_store.types import (
    APIKeyCredentials,
)
from backend.data import integrations
from fastapi import Request
from backend.integrations.webhooks.base import BaseWebhooksManager
import requests


class Slant3DWebhooksManager(BaseWebhooksManager):
    """Manager for Slant3D webhooks"""

    PROVIDER_NAME: ClassVar[str] = "slant3d"
    BASE_URL = "https://www.slant3dapi.com/api"

    async def _register_webhook(
        self,
        credentials: APIKeyCredentials,
        webhook_type: str,
        resource: str,
        events: List[str],
        ingress_url: str,
        secret: str,
    ) -> Tuple[str, Dict]:
        """Register a new webhook with Slant3D"""

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

        # Slant3D doesn't return a webhook ID, so we generate one
        # The actual webhook is identified by its endpoint URL
        webhook_id = str(__import__("uuid").uuid4())

        webhook_config = {
            "endpoint": ingress_url,
            "provider": self.PROVIDER_NAME,
            "events": ["order.shipped"],  # Currently the only supported event
            "type": webhook_type,
        }

        return webhook_id, webhook_config

    @classmethod
    async def validate_payload(
        cls, webhook: integrations.Webhook, request: Request
    ) -> Tuple[Dict, str]:
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
        self, webhook: integrations.Webhook, credentials: APIKeyCredentials
    ) -> None:
        """
        Note: Slant3D API currently doesn't provide a deregistration endpoint.
        This would need to be handled through support.
        """
        # Log warning since we can't properly deregister
        print(f"Warning: Manual deregistration required for webhook {webhook.id}")
        pass

    async def trigger_ping(self, webhook: integrations.Webhook) -> None:
        """
        Note: Slant3D automatically sends a test payload during registration,
        so we don't need a separate ping implementation.
        """
        pass
