"""
Webhook management for Airtable blocks.
"""

import hashlib
import hmac
from enum import Enum
from typing import Tuple

from backend.sdk import (
    APIKeyCredentials,
    BaseWebhooksManager,
    Credentials,
    ProviderName,
    Requests,
    Webhook,
)


class AirtableWebhookManager(BaseWebhooksManager):
    """Webhook manager for Airtable API."""

    PROVIDER_NAME = ProviderName("airtable")

    class WebhookType(str, Enum):
        TABLE_CHANGE = "table_change"

    @classmethod
    async def validate_payload(cls, webhook: Webhook, request) -> Tuple[dict, str]:
        """Validate incoming webhook payload and signature."""
        payload = await request.json()

        # Verify webhook signature using HMAC-SHA256
        if webhook.secret:
            mac_secret = webhook.config.get("mac_secret")
            if mac_secret:
                # Get the raw body for signature verification
                body = await request.body()

                # Calculate expected signature
                expected_mac = hmac.new(
                    mac_secret.encode(), body, hashlib.sha256
                ).hexdigest()

                # Get signature from headers
                signature = request.headers.get("X-Airtable-Content-MAC")

                if signature and not hmac.compare_digest(signature, expected_mac):
                    raise ValueError("Invalid webhook signature")

        # Airtable sends the cursor in the payload
        event_type = "notification"
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
        """Register webhook with Airtable API."""
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("Airtable webhooks require API key credentials")

        api_key = credentials.api_key.get_secret_value()

        # Parse resource to get base_id and table_id/name
        # Resource format: "{base_id}/{table_id_or_name}"
        parts = resource.split("/", 1)
        if len(parts) != 2:
            raise ValueError("Resource must be in format: {base_id}/{table_id_or_name}")

        base_id, table_id_or_name = parts

        # Prepare webhook specification
        specification = {
            "filters": {
                "dataTypes": events or ["tableData", "tableFields", "tableMetadata"]
            }
        }

        # If specific table is provided, add to specification
        if table_id_or_name and table_id_or_name != "*":
            specification["filters"]["recordChangeScope"] = [table_id_or_name]

        # Create webhook
        response = await Requests().post(
            f"https://api.airtable.com/v0/bases/{base_id}/webhooks",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"notificationUrl": ingress_url, "specification": specification},
        )

        webhook_data = response.json()
        webhook_id = webhook_data["id"]
        mac_secret = webhook_data.get("macSecretBase64")

        return webhook_id, {
            "base_id": base_id,
            "table_id_or_name": table_id_or_name,
            "events": events,
            "mac_secret": mac_secret,
            "cursor": 1,  # Start from cursor 1
            "expiration_time": webhook_data.get("expirationTime"),
        }

    async def _deregister_webhook(
        self, webhook: Webhook, credentials: Credentials
    ) -> None:
        """Deregister webhook from Airtable API."""
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("Airtable webhooks require API key credentials")

        api_key = credentials.api_key.get_secret_value()
        base_id = webhook.config.get("base_id")

        if not base_id:
            raise ValueError("Missing base_id in webhook metadata")

        await Requests().delete(
            f"https://api.airtable.com/v0/bases/{base_id}/webhooks/{webhook.provider_webhook_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
