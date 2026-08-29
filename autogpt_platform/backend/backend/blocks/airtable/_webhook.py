"""
Webhook management for Airtable blocks.
"""

import base64
import hashlib
import hmac
import logging
from enum import Enum
from typing import cast

from fastapi import HTTPException, Request
from prisma.types import Serializable

from backend.sdk import (
    BaseWebhooksManager,
    Credentials,
    ProviderName,
    Webhook,
    update_webhook,
)
from backend.util.request import HTTPClientError, HTTPServerError

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
    async def verify_signature(cls, webhook: Webhook, request: Request) -> None:
        # Airtable returns the signing secret base64-encoded as `macSecretBase64`
        # (stored in `config["mac_secret"]`); it must be base64-decoded before
        # use as the HMAC key.
        mac_secret_b64 = webhook.config.get("mac_secret")
        if not mac_secret_b64:
            raise HTTPException(
                status_code=403,
                detail="Webhook is missing Airtable MAC secret; re-register the webhook",
            )

        signature = request.headers.get("X-Airtable-Content-MAC")
        if not signature:
            raise HTTPException(
                status_code=403, detail="Missing X-Airtable-Content-MAC header"
            )

        try:
            mac_secret = base64.b64decode(mac_secret_b64)
        except Exception:
            raise HTTPException(
                status_code=403, detail="Stored Airtable MAC secret is not valid base64"
            )

        body = await request.body()
        hmac_obj = hmac.new(mac_secret, body, hashlib.sha256)
        expected_mac = f"hmac-sha256={hmac_obj.hexdigest()}"

        if not hmac.compare_digest(signature, expected_mac):
            raise HTTPException(status_code=403, detail="Invalid webhook signature")

    @classmethod
    async def validate_payload(
        cls, webhook: Webhook, request, credentials: Credentials | None
    ) -> tuple[dict, str]:
        """Validate incoming webhook payload structure."""

        if not credentials:
            raise ValueError("Missing credentials in webhook metadata")

        payload = await request.json()

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

        # Merge cursor update into existing config — `update_webhook` does a
        # full replace on the config blob, and dropping `mac_secret`/other
        # fields here would break subsequent signature verification.
        await update_webhook(
            webhook.id,
            config=cast(
                dict[str, Serializable],
                {**webhook.config, "base_id": base_id, "cursor": response.cursor},
            ),
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

        try:
            webhook_data = await create_webhook(
                credentials=credentials,
                base_id=base_id,
                webhook_specification=webhook_specification,
                notification_url=ingress_url,
            )
        except (HTTPClientError, HTTPServerError) as e:
            raise ValueError(
                "Airtable returned error "
                f"for webhook registration on base '{base_id}': {e}"
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
