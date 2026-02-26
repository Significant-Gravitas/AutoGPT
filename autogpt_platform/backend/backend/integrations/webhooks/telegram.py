"""
Telegram Bot API Webhooks Manager.

Handles webhook registration and validation for Telegram bots.
"""

import hmac
import logging

from fastapi import HTTPException, Request
from strenum import StrEnum

from backend.data import integrations
from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.providers import ProviderName
from backend.util.exceptions import MissingConfigError
from backend.util.request import Requests
from backend.util.settings import Config

from ._base import BaseWebhooksManager
from .utils import webhook_ingress_url

logger = logging.getLogger(__name__)


class TelegramWebhookType(StrEnum):
    BOT = "bot"


class TelegramWebhooksManager(BaseWebhooksManager):
    """
    Manages Telegram bot webhooks.

    Telegram webhooks are registered via the setWebhook API method.
    Incoming requests are validated using the secret_token header.
    """

    PROVIDER_NAME = ProviderName.TELEGRAM
    WebhookType = TelegramWebhookType

    TELEGRAM_API_BASE = "https://api.telegram.org"

    async def get_suitable_auto_webhook(
        self,
        user_id: str,
        credentials: Credentials,
        webhook_type: TelegramWebhookType,
        resource: str,
        events: list[str],
    ) -> integrations.Webhook:
        """
        Telegram only supports one webhook per bot. Instead of creating a new
        webhook object when events change (which causes the old one to be pruned
        and deregistered — removing the ONLY webhook for the bot), we find the
        existing webhook and update its events in place.
        """
        app_config = Config()
        if not app_config.platform_base_url:
            raise MissingConfigError(
                "PLATFORM_BASE_URL must be set to use Webhook functionality"
            )

        # Exact match — no re-registration needed
        if webhook := await integrations.find_webhook_by_credentials_and_props(
            user_id=user_id,
            credentials_id=credentials.id,
            webhook_type=webhook_type,
            resource=resource,
            events=events,
        ):
            return webhook

        # Find any existing webhook for the same bot, regardless of events
        if existing := await integrations.find_webhook_by_credentials_and_props(
            user_id=user_id,
            credentials_id=credentials.id,
            webhook_type=webhook_type,
            resource=resource,
            events=None,  # Ignore events for this lookup
        ):
            # Re-register with Telegram using the same URL but new allowed_updates
            ingress_url = webhook_ingress_url(self.PROVIDER_NAME, existing.id)
            _, config = await self._register_webhook(
                credentials,
                webhook_type,
                resource,
                events,
                ingress_url,
                existing.secret,
            )
            return await integrations.update_webhook(
                existing.id, events=events, config=config
            )

        # No existing webhook at all — create a new one
        return await self._create_webhook(
            user_id=user_id,
            webhook_type=webhook_type,
            events=events,
            resource=resource,
            credentials=credentials,
        )

    @classmethod
    async def validate_payload(
        cls,
        webhook: integrations.Webhook,
        request: Request,
        credentials: Credentials | None,
    ) -> tuple[dict, str]:
        """
        Validates incoming Telegram webhook request.

        Telegram sends X-Telegram-Bot-Api-Secret-Token header when secret_token
        was set in setWebhook call.

        Returns:
            tuple: (payload dict, event_type string)
        """
        # Verify secret token header
        secret_header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if not secret_header or not hmac.compare_digest(secret_header, webhook.secret):
            raise HTTPException(
                status_code=403,
                detail="Invalid or missing X-Telegram-Bot-Api-Secret-Token",
            )

        payload = await request.json()

        # Determine event type based on update content
        if "message" in payload:
            message = payload["message"]
            if "text" in message:
                event_type = "message.text"
            elif "photo" in message:
                event_type = "message.photo"
            elif "voice" in message:
                event_type = "message.voice"
            elif "audio" in message:
                event_type = "message.audio"
            elif "document" in message:
                event_type = "message.document"
            elif "video" in message:
                event_type = "message.video"
            else:
                logger.warning(
                    "Unknown Telegram webhook payload type; "
                    f"message.keys() = {message.keys()}"
                )
                event_type = "message.other"
        elif "edited_message" in payload:
            event_type = "message.edited_message"
        elif "message_reaction" in payload:
            event_type = "message_reaction"
        else:
            event_type = "unknown"

        return payload, event_type

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: TelegramWebhookType,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        """
        Register webhook with Telegram using setWebhook API.

        Args:
            credentials: Bot token credentials
            webhook_type: Type of webhook (always BOT for Telegram)
            resource: Resource identifier (unused for Telegram, bots are global)
            events: Events to subscribe to
            ingress_url: URL to receive webhook payloads
            secret: Secret token for request validation

        Returns:
            tuple: (provider_webhook_id, config dict)
        """
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("API key (bot token) is required for Telegram webhooks")

        token = credentials.api_key.get_secret_value()
        url = f"{self.TELEGRAM_API_BASE}/bot{token}/setWebhook"

        # Map event filter to Telegram allowed_updates
        if events:
            telegram_updates: set[str] = set()
            for event in events:
                telegram_updates.add(event.split(".")[0])
                # "message.edited_message" requires the "edited_message" update type
                if "edited_message" in event:
                    telegram_updates.add("edited_message")
            sorted_updates = sorted(telegram_updates)
        else:
            sorted_updates = ["message", "message_reaction"]

        webhook_data = {
            "url": ingress_url,
            "secret_token": secret,
            "allowed_updates": sorted_updates,
        }

        response = await Requests().post(url, json=webhook_data)
        result = response.json()

        if not result.get("ok"):
            error_desc = result.get("description", "Unknown error")
            raise ValueError(f"Failed to set Telegram webhook: {error_desc}")

        # Telegram doesn't return a webhook ID, use empty string
        config = {
            "url": ingress_url,
            "allowed_updates": webhook_data["allowed_updates"],
        }

        return "", config

    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None:
        """
        Deregister webhook by calling setWebhook with empty URL.

        This removes the webhook from Telegram's servers.
        """
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("API key (bot token) is required for Telegram webhooks")

        token = credentials.api_key.get_secret_value()
        url = f"{self.TELEGRAM_API_BASE}/bot{token}/setWebhook"

        # Setting empty URL removes the webhook
        response = await Requests().post(url, json={"url": ""})
        result = response.json()

        if not result.get("ok"):
            error_desc = result.get("description", "Unknown error")
            logger.warning(f"Failed to deregister Telegram webhook: {error_desc}")
