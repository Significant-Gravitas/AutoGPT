"""
Exa Webhook Manager implementation.
"""

import hashlib
import hmac
from enum import Enum

from fastapi import HTTPException, Request

from backend.data.model import Credentials
from backend.sdk import (
    APIKeyCredentials,
    BaseWebhooksManager,
    ProviderName,
    Requests,
    Webhook,
)


class ExaWebhookType(str, Enum):
    """Available webhook types for Exa."""

    WEBSET = "webset"


class ExaEventType(str, Enum):
    """Available event types for Exa webhooks."""

    WEBSET_CREATED = "webset.created"
    WEBSET_DELETED = "webset.deleted"
    WEBSET_PAUSED = "webset.paused"
    WEBSET_IDLE = "webset.idle"
    WEBSET_SEARCH_CREATED = "webset.search.created"
    WEBSET_SEARCH_CANCELED = "webset.search.canceled"
    WEBSET_SEARCH_COMPLETED = "webset.search.completed"
    WEBSET_SEARCH_UPDATED = "webset.search.updated"
    IMPORT_CREATED = "import.created"
    IMPORT_COMPLETED = "import.completed"
    IMPORT_PROCESSING = "import.processing"
    WEBSET_ITEM_CREATED = "webset.item.created"
    WEBSET_ITEM_ENRICHED = "webset.item.enriched"
    WEBSET_EXPORT_CREATED = "webset.export.created"
    WEBSET_EXPORT_COMPLETED = "webset.export.completed"


class ExaWebhookManager(BaseWebhooksManager):
    """Webhook manager for Exa API."""

    PROVIDER_NAME = ProviderName("exa")

    class WebhookType(str, Enum):
        WEBSET = "webset"

    @classmethod
    async def verify_signature(cls, webhook: Webhook, request: Request) -> None:
        # Exa signs deliveries with the secret it returns at registration time,
        # stored in `config["exa_secret"]` (not `webhook.secret`). The
        # `Exa-Signature` header carries a timestamp and one or more signatures:
        # `t=<unix_ts>,v1=<hex>[,v1=<hex>...]`. The signed payload is
        # `<timestamp>.<raw body>`, HMAC-SHA256 keyed with the secret.
        # See https://docs.exa.ai/websets/api/webhooks/verifying-signatures
        signing_secret = webhook.config.get("exa_secret")
        if not signing_secret:
            raise HTTPException(
                status_code=403,
                detail="Webhook is missing Exa signing secret; re-register the webhook",
            )

        signature_header = request.headers.get("Exa-Signature")
        if not signature_header:
            raise HTTPException(status_code=403, detail="Missing Exa-Signature header")

        timestamp: str | None = None
        provided_signatures: list[str] = []
        for part in signature_header.split(","):
            key, _, value = part.strip().partition("=")
            if key == "t":
                timestamp = value
            elif key == "v1":
                provided_signatures.append(value)

        if not timestamp or not provided_signatures:
            raise HTTPException(
                status_code=403, detail="Malformed Exa-Signature header"
            )

        body = await request.body()
        signed_payload = f"{timestamp}.".encode() + body
        expected_signature = hmac.new(
            signing_secret.encode(), signed_payload, hashlib.sha256
        ).hexdigest()

        if not any(
            hmac.compare_digest(expected_signature, sig) for sig in provided_signatures
        ):
            raise HTTPException(status_code=403, detail="Invalid webhook signature")

    @classmethod
    async def validate_payload(
        cls, webhook: Webhook, request, credentials: Credentials | None
    ) -> tuple[dict, str]:
        """Validate incoming webhook payload."""
        payload = await request.json()
        event_type = payload.get("eventType", "unknown")
        return payload, event_type

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        """Register webhook with Exa API."""
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("Exa webhooks require API key credentials")
        api_key = credentials.api_key.get_secret_value()

        # Create webhook via Exa API
        response = await Requests(raise_for_status=False).post(
            "https://api.exa.ai/v0/webhooks",
            headers={"x-api-key": api_key},
            json={
                "url": ingress_url,
                "events": events,
                "metadata": {
                    "resource": resource,
                    "webhook_type": webhook_type,
                },
            },
        )

        if not response.ok:
            error_data = response.json()
            raise ValueError(f"Exa returned error: {error_data}")

        webhook_data = response.json()

        # Store the secret returned by Exa
        return webhook_data["id"], {
            "events": events,
            "resource": resource,
            "exa_secret": webhook_data.get("secret"),
        }

    async def _deregister_webhook(
        self, webhook: Webhook, credentials: Credentials
    ) -> None:
        """Deregister webhook from Exa API."""
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("Exa webhooks require API key credentials")
        api_key = credentials.api_key.get_secret_value()

        # Delete webhook via Exa API
        response = await Requests().delete(
            f"https://api.exa.ai/v0/webhooks/{webhook.provider_webhook_id}",
            headers={"x-api-key": api_key},
        )

        if not response.ok and response.status != 404:
            error_data = response.json()
            raise Exception(f"Failed to delete Exa webhook: {error_data}")
