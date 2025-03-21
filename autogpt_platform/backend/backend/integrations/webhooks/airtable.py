"""
Webhook manager for Airtable webhooks.

This module manages the registration and processing of webhooks from Airtable.
"""

import logging
from typing import Dict, Tuple

import requests
from fastapi import Request
from strenum import StrEnum

from backend.data import integrations
from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.providers import ProviderName

from ._manual_base import ManualWebhookManagerBase

logger = logging.getLogger(__name__)


class AirtableWebhookEventType(StrEnum):
    """Types of webhook events supported by Airtable."""

    RECORDS_CREATED = "records:created"
    RECORDS_UPDATED = "records:updated"
    RECORDS_DELETED = "records:deleted"


class AirtableWebhookManager(ManualWebhookManagerBase):
    """Manager class for Airtable webhooks."""

    # Provider name for this webhook manager
    PROVIDER_NAME = ProviderName.AIRTABLE
    # Define the webhook event types this manager can handle
    WebhookEventType = AirtableWebhookEventType

    # Airtable API URL for webhooks
    BASE_URL = "https://api.airtable.com/v0"

    @classmethod
    async def validate_payload(
        cls, webhook: integrations.Webhook, request: Request
    ) -> Tuple[Dict, str]:
        """
        Validate the incoming webhook payload.

        Args:
            webhook: The webhook object from the database.
            request: The incoming request containing the webhook payload.

        Returns:
            A tuple of (payload_dict, event_type)
        """
        # Extract the JSON payload from the request
        payload = await request.json()

        # Determine the event type from the payload
        event_type = payload.get("event", AirtableWebhookEventType.RECORDS_UPDATED)

        return payload, event_type

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> Tuple[str, Dict]:
        """
        Register a webhook with Airtable.

        Args:
            credentials: The API credentials.
            webhook_type: The type of webhook to register.
            resource: The base ID to register webhooks for.
            events: List of event types to listen for.
            ingress_url: URL where webhook notifications should be sent.
            secret: Secret for webhook security.

        Returns:
            Tuple of (webhook_id, webhook_config)
        """
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("API key is required to register Airtable webhook")

        headers = {
            "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        payload = {
            "url": ingress_url,
            "event": webhook_type,  # Use the webhook_type as the event type
        }

        response = requests.post(
            f"{self.BASE_URL}/bases/{resource}/webhooks",
            headers=headers,
            json=payload,
        )

        if not response.ok:
            error = response.json().get("error", "Unknown error")
            raise ValueError(f"Failed to register Airtable webhook: {error}")

        webhook_data = response.json()
        webhook_id = webhook_data.get("id", "")

        webhook_config = {
            "provider": self.PROVIDER_NAME,
            "base_id": resource,
            "event": webhook_type,
            "url": ingress_url,
        }

        return webhook_id, webhook_config
