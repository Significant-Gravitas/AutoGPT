"""
Webhook management for Airtable blocks.
"""

import hashlib
import hmac
import logging
from enum import Enum
from typing import Tuple

from backend.sdk import (
    APIKeyCredentials,
    BaseWebhooksManager,
    Credentials,
    ProviderName,
    Requests,
    Webhook,
    update_webhook,
)

from ._api import (
    WebhookFilters,
    WebhookSpecification,
    create_webhook,
    list_webhook_payloads,
)

logger = logging.getLogger(__name__)


class AirtableWebhookManager(BaseWebhooksManager):
    """Webhook manager for Airtable API."""

    PROVIDER_NAME = ProviderName("airtable")

    class WebhookType(str, Enum):
        TABLE_CHANGE = "table_change"

    @classmethod
    async def validate_payload(
        cls, webhook: Webhook, request, credentials: Credentials | None
    ) -> Tuple[dict, str]:
        """Validate incoming webhook payload and signature."""
        payload = await request.json()

        # Verify webhook signature using HMAC-SHA256
        if webhook.secret:
            mac_secret = webhook.config.get("mac_secret")
            if mac_secret:
                # Get the raw body for signature verification
                body = await request.body()

                # Calculate expected signature
                mac_secret_decoded = mac_secret.encode()
                hmac_obj = hmac.new(mac_secret_decoded, body, hashlib.sha256)
                expected_mac = f"hmac-sha256={hmac_obj.hexdigest()}"

                # Get signature from headers
                signature = request.headers.get("X-Airtable-Content-MAC")

                if signature and not hmac.compare_digest(signature, expected_mac):
                    raise ValueError("Invalid webhook signature")

        # Validate payload structure
        required_fields = ["base", "webhook", "timestamp"]
        if not all(field in payload for field in required_fields):
            raise ValueError("Invalid webhook payload structure")

        if "id" not in payload["base"] or "id" not in payload["webhook"]:
            raise ValueError("Missing required IDs in webhook payload")

        # get payload request parameters
        base_id = webhook.config.get("base_id")
        cursor = webhook.config.get("cursor")
        webhook_id = webhook.config.get("webhook_id")

        assert base_id is not None
        assert webhook_id is not None
        assert credentials is not None

        response = await list_webhook_payloads(credentials, base_id, webhook_id, cursor)

        # update webhook config
        await update_webhook(
            webhook.id,
            config={"base_id": base_id, "cursor": response.cursor},
        )

        event_type = "notification"
        return response.model_dump(), event_type

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

        # Parse resource to get base_id and table_id/name
        # Resource format: "{base_id}/{table_id_or_name}"
        parts = resource.split("/", 1)
        if len(parts) != 2:
            raise ValueError("Resource must be in format: {base_id}/{table_id_or_name}")

        base_id, table_id_or_name = parts

        # Prepare webhook specification
        webhook_specification = WebhookSpecification(
            filters=WebhookFilters(
                dataTypes=events or ["tableData", "tableFields", "tableMetadata"],
            )
        )

        # Create webhook
        webhook_data = await create_webhook(
            credentials=credentials,
            base_id=base_id,
            webhook_specification=webhook_specification,
            notification_url=ingress_url,
        )

        webhook_id = webhook_data["id"]
        mac_secret = webhook_data.get("macSecretBase64")

        return webhook_id, {
            "webhook_id": webhook_id,
            "base_id": base_id,
            "table_id_or_name": table_id_or_name,
            "events": events,
            "mac_secret": mac_secret,
            "cursor": 1,
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
        webhook_id = webhook.config.get("webhook_id")

        if not base_id:
            raise ValueError("Missing base_id in webhook metadata")

        if not webhook_id:
            raise ValueError("Missing webhook_id in webhook metadata")

        await Requests().delete(
            f"https://api.airtable.com/v0/bases/{base_id}/webhooks/{webhook_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
