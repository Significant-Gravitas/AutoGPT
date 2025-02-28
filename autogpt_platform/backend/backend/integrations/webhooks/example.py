import logging

import requests
from fastapi import Request
from strenum import StrEnum

from backend.data import integrations
from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.providers import ProviderName

from ._manual_base import ManualWebhookManagerBase

logger = logging.getLogger(__name__)


class ExampleWebhookType(StrEnum):
    EXAMPLE = "example"
    EXAMPLE_2 = "example_2"


# ExampleWebhookManager is a class that manages webhooks for a hypothetical provider.
# It extends ManualWebhookManagerBase, which provides base functionality for manual webhook management.
class ExampleWebhookManager(ManualWebhookManagerBase):
    # Define the provider name for this webhook manager.
    PROVIDER_NAME = ProviderName.EXAMPLE_PROVIDER
    # Define the types of webhooks this manager can handle.
    WebhookType = ExampleWebhookType

    BASE_URL = "https://api.example.com"

    @classmethod
    async def validate_payload(
        cls, webhook: integrations.Webhook, request: Request
    ) -> tuple[dict, str]:
        """
        Validate the incoming webhook payload.

        Args:
            webhook (integrations.Webhook): The webhook object.
            request (Request): The incoming request object.

        Returns:
            tuple: A tuple containing the payload as a dictionary and the event type as a string.
        """
        # Extract the JSON payload from the request.
        payload = await request.json()
        # Set the event type based on the webhook type in the payload.
        event_type = payload.get("webhook_type", ExampleWebhookType.EXAMPLE)

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
        Register a new webhook with the provider.

        Args:
            credentials (Credentials): The credentials required for authentication.
            webhook_type (str): The type of webhook to register.
            resource (str): The resource associated with the webhook.
            events (list[str]): The list of events to subscribe to.
            ingress_url (str): The URL where the webhook will send data.
            secret (str): A secret for securing the webhook.

        Returns:
            tuple: A tuple containing an empty string and the webhook configuration as a dictionary.
        """
        # Ensure the credentials are of the correct type.
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("API key is required to register a webhook")

        # Prepare the headers for the request, including the API key.
        headers = {
            "api-key": credentials.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

        # Prepare the payload for the request. Note that the events list is not used.
        # This is just a fake example
        payload = {"endPoint": ingress_url}

        # Send a POST request to register the webhook.
        response = requests.post(
            f"{self.BASE_URL}/example/webhookSubscribe", headers=headers, json=payload
        )

        # Check if the response indicates a failure.
        if not response.ok:
            error = response.json().get("error", "Unknown error")
            raise RuntimeError(f"Failed to register webhook: {error}")

        # Prepare the webhook configuration to return.
        webhook_config = {
            "endpoint": ingress_url,
            "provider": self.PROVIDER_NAME,
            "events": ["example_event"],
            "type": webhook_type,
        }

        return "", webhook_config

    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None:
        """
        Deregister a webhook with the provider.

        Args:
            webhook (integrations.Webhook): The webhook object to deregister.
            credentials (Credentials): The credentials associated with the webhook.

        Raises:
            ValueError: If the webhook doesn't belong to the credentials or if deregistration fails.
        """
        if webhook.credentials_id != credentials.id:
            raise ValueError(
                f"Webhook #{webhook.id} does not belong to credentials {credentials.id}"
            )

        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("API key is required to deregister a webhook")

        headers = {
            "api-key": credentials.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

        # Construct the delete URL based on the webhook information
        delete_url = f"{self.BASE_URL}/example/webhooks/{webhook.provider_webhook_id}"

        response = requests.delete(delete_url, headers=headers)

        if response.status_code not in [204, 404]:
            # 204 means successful deletion, 404 means the webhook was already deleted
            error = response.json().get("error", "Unknown error")
            raise ValueError(f"Failed to delete webhook: {error}")

        # If we reach here, the webhook was successfully deleted or didn't exist
