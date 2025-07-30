"""
Webhook management for Airtable blocks.
"""

import hashlib
import hmac
import logging
from enum import Enum

from backend.sdk import (
    BaseWebhooksManager,
    Credentials,
    ProviderName,
    Webhook,
    update_webhook,
)

from ._api import (
    WebhookFilters,
    WebhookSpecification,
    create_webhook,
    delete_webhook,
    list_webhook_payloads,
)

logger = logging.getLogger(__name__)


class AirtableWebhookEvent(str, Enum):
    TABLE_DATA = "tableData"
    TABLE_FIELDS = "tableFields"
    TABLE_METADATA = "tableMetadata"


class AirtableWebhookManager(BaseWebhooksManager):
    """Webhook manager for Airtable API."""

    PROVIDER_NAME = ProviderName("airtable")

    @classmethod
    async def validate_payload(
        cls, webhook: Webhook, request, credentials: Credentials | None
    ) -> tuple[dict, str]:
        """Validate incoming webhook payload and signature."""

        if not credentials:
            raise ValueError("Missing credentials in webhook metadata")

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
        base_id = payload["base"]["id"]
        webhook_id = payload["webhook"]["id"]

        # get payload request parameters
        cursor = webhook.config.get("cursor", 1)

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
    ) -> tuple[str, dict]:
        """Register webhook with Airtable API."""

        # Parse resource to get base_id and table_id/name
        # Resource format: "{base_id}/{table_id_or_name}"
        parts = resource.split("/", 1)
        if len(parts) != 2:
            raise ValueError("Resource must be in format: {base_id}/{table_id_or_name}")

        base_id, table_id_or_name = parts

        # Prepare webhook specification
        webhook_specification = WebhookSpecification(
            filters=WebhookFilters(
                dataTypes=events,
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

        base_id = webhook.config.get("base_id")
        webhook_id = webhook.config.get("webhook_id")

        if not base_id:
            raise ValueError("Missing base_id in webhook metadata")

        if not webhook_id:
            raise ValueError("Missing webhook_id in webhook metadata")

        await delete_webhook(credentials, base_id, webhook_id)
