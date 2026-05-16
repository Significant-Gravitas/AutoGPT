"""
Exa Webhook Manager implementation.
"""

import hashlib
import hmac
import logging
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
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)


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
        # Enforcement is gated on a feature flag during rollout: the previous
        # verification path was reading the wrong secret (`webhook.secret`
        # instead of the one Exa returns at registration in
        # `config["exa_secret"]`), so we have no production evidence that
        # signature verification ever succeeded on a real delivery. Once
        # smoke-tested against staging, the flag will be flipped on globally.
        if not await is_feature_enabled(
            Flag.ENFORCE_EXA_SIGNATURE, webhook.user_id, default=False
        ):
            return

        signing_secret = webhook.config.get("exa_secret")
        if not signing_secret:
            raise HTTPException(
                status_code=403,
                detail="Webhook is missing Exa signing secret; re-register the webhook",
            )

        signature = request.headers.get("X-Exa-Signature")
        if not signature:
            raise HTTPException(
                status_code=403, detail="Missing X-Exa-Signature header"
            )

        body = await request.body()
        expected_signature = hmac.new(
            signing_secret.encode(), body, hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
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
        response = await Requests().post(
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
            raise Exception(f"Failed to create Exa webhook: {error_data}")

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
