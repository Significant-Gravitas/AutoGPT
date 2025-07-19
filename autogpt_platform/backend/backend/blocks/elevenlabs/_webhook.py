"""
ElevenLabs webhook manager for handling webhook events.
"""

import hashlib
import hmac
from typing import Tuple

from backend.data.model import Credentials
from backend.sdk import BaseWebhooksManager, ProviderName, Webhook


class ElevenLabsWebhookManager(BaseWebhooksManager):
    """Manages ElevenLabs webhook events."""

    PROVIDER_NAME = ProviderName("elevenlabs")

    @classmethod
    async def validate_payload(cls, webhook: Webhook, request) -> Tuple[dict, str]:
        """
        Validate incoming webhook payload and signature.

        ElevenLabs supports HMAC authentication for webhooks.
        """
        payload = await request.json()

        # Verify webhook signature if configured
        if webhook.secret:
            webhook_secret = webhook.config.get("webhook_secret")
            if webhook_secret:
                # Get the raw body for signature verification
                body = await request.body()

                # Calculate expected signature
                expected_signature = hmac.new(
                    webhook_secret.encode(), body, hashlib.sha256
                ).hexdigest()

                # Get signature from headers
                signature = request.headers.get("x-elevenlabs-signature")

                if signature and not hmac.compare_digest(signature, expected_signature):
                    raise ValueError("Invalid webhook signature")

        # Extract event type from payload
        event_type = payload.get("type", "unknown")
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
        """
        Register a webhook with ElevenLabs.

        Note: ElevenLabs webhook registration is done through their dashboard,
        not via API. This is a placeholder implementation.
        """
        # ElevenLabs requires manual webhook setup through dashboard
        # Return empty webhook ID and config with instructions
        config = {
            "manual_setup_required": True,
            "webhook_secret": secret,
            "instructions": "Please configure webhook URL in ElevenLabs dashboard",
        }
        return "", config

    async def _deregister_webhook(
        self, webhook: Webhook, credentials: Credentials
    ) -> None:
        """
        Deregister a webhook with ElevenLabs.

        Note: ElevenLabs webhook removal is done through their dashboard.
        """
        # ElevenLabs requires manual webhook removal through dashboard
        pass
