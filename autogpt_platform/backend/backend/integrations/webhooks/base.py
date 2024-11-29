import logging
import secrets
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar
from uuid import uuid4

from fastapi import Request
from strenum import StrEnum

from backend.data import integrations
from backend.data.model import Credentials
from backend.util.exceptions import MissingConfigError
from backend.util.settings import Config

logger = logging.getLogger(__name__)
app_config = Config()

WT = TypeVar("WT", bound=StrEnum)


class BaseWebhooksManager(ABC, Generic[WT]):
    # --8<-- [start:BaseWebhooksManager1]
    PROVIDER_NAME: ClassVar[str]
    # --8<-- [end:BaseWebhooksManager1]

    WebhookType: WT

    async def get_suitable_webhook(
        self,
        user_id: str,
        credentials: Credentials,
        webhook_type: WT,
        resource: str,
        events: list[str],
    ) -> integrations.Webhook:
        if not app_config.platform_base_url:
            raise MissingConfigError(
                "PLATFORM_BASE_URL must be set to use Webhook functionality"
            )

        if webhook := await integrations.find_webhook(
            credentials.id, webhook_type, resource, events
        ):
            return webhook
        return await self._create_webhook(
            user_id, credentials, webhook_type, resource, events
        )

    async def prune_webhook_if_dangling(
        self, webhook_id: str, credentials: Credentials
    ) -> bool:
        webhook = await integrations.get_webhook(webhook_id)
        if webhook.attached_nodes is None:
            raise ValueError("Error retrieving webhook including attached nodes")
        if webhook.attached_nodes:
            # Don't prune webhook if in use
            return False

        await self._deregister_webhook(webhook, credentials)
        await integrations.delete_webhook(webhook.id)
        return True

    # --8<-- [start:BaseWebhooksManager3]
    @classmethod
    @abstractmethod
    async def validate_payload(
        cls, webhook: integrations.Webhook, request: Request
    ) -> tuple[dict, str]:
        """
        Validates an incoming webhook request and returns its payload and type.

        Params:
            webhook: Object representing the configured webhook and its properties in our system.
            request: Incoming FastAPI `Request`

        Returns:
            dict: The validated payload
            str: The event type associated with the payload
        """

    # --8<-- [end:BaseWebhooksManager3]

    # --8<-- [start:BaseWebhooksManager5]
    async def trigger_ping(self, webhook: integrations.Webhook) -> None:
        """
        Triggers a ping to the given webhook.

        Raises:
            NotImplementedError: if the provider doesn't support pinging
        """
        # --8<-- [end:BaseWebhooksManager5]
        raise NotImplementedError(f"{self.__class__.__name__} doesn't support pinging")

    # --8<-- [start:BaseWebhooksManager2]
    @abstractmethod
    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: WT,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        """
        Registers a new webhook with the provider.

        Params:
            credentials: The credentials with which to create the webhook
            webhook_type: The provider-specific webhook type to create
            resource: The resource to receive events for
            events: The events to subscribe to
            ingress_url: The ingress URL for webhook payloads
            secret: Secret used to verify webhook payloads

        Returns:
            str: Webhook ID assigned by the provider
            config: Provider-specific configuration for the webhook
        """
        ...

    # --8<-- [end:BaseWebhooksManager2]

    # --8<-- [start:BaseWebhooksManager4]
    @abstractmethod
    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None: ...

    # --8<-- [end:BaseWebhooksManager4]

    async def _create_webhook(
        self,
        user_id: str,
        credentials: Credentials,
        webhook_type: WT,
        resource: str,
        events: list[str],
    ) -> integrations.Webhook:
        id = str(uuid4())
        secret = secrets.token_hex(32)
        provider_name = self.PROVIDER_NAME
        ingress_url = (
            f"{app_config.platform_base_url}/api/integrations/{provider_name}"
            f"/webhooks/{id}/ingress"
        )
        provider_webhook_id, config = await self._register_webhook(
            credentials, webhook_type, resource, events, ingress_url, secret
        )
        return await integrations.create_webhook(
            integrations.Webhook(
                id=id,
                user_id=user_id,
                provider=provider_name,
                credentials_id=credentials.id,
                webhook_type=webhook_type,
                resource=resource,
                events=events,
                provider_webhook_id=provider_webhook_id,
                config=config,
                secret=secret,
            )
        )
