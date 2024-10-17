import hashlib
import hmac
import logging
from enum import Enum

import requests
from autogpt_libs.supabase_integration_credentials_store import Credentials
from fastapi import HTTPException, Request

from backend.data import integrations
from backend.integrations.providers import ProviderName

from .base import BaseWebhooksManager

logger = logging.getLogger(__name__)


class GithubWebhookType(str, Enum):
    REPO = "repo"


class GithubWebhooksManager(BaseWebhooksManager):
    PROVIDER_NAME = ProviderName.GITHUB

    WebhookType = GithubWebhookType

    GITHUB_API_URL = "https://api.github.com"
    GITHUB_API_DEFAULT_HEADERS = {"Accept": "application/vnd.github.v3+json"}

    @classmethod
    async def validate_payload(
        cls, webhook: integrations.Webhook, request: Request
    ) -> dict:
        payload_body = await request.body()
        signature_header = request.headers.get("x-hub-signature-256")
        secret_token = webhook.secret

        if not signature_header:
            raise HTTPException(
                status_code=403, detail="x-hub-signature-256 header is missing!"
            )

        hash_object = hmac.new(
            secret_token.encode("utf-8"), msg=payload_body, digestmod=hashlib.sha256
        )
        expected_signature = "sha256=" + hash_object.hexdigest()

        if not hmac.compare_digest(expected_signature, signature_header):
            raise HTTPException(
                status_code=403, detail="Request signatures didn't match!"
            )

        return await request.json()

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: GithubWebhookType,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        if webhook_type == self.WebhookType.REPO and resource.count("/") > 1:
            raise ValueError("Invalid resource format: expected 'owner/repo'")

        headers = {
            **self.GITHUB_API_DEFAULT_HEADERS,
            "Authorization": credentials.bearer(),
        }
        webhook_data = {
            "name": "web",
            "active": True,
            "events": events,
            "config": {
                "url": ingress_url,
                "content_type": "json",
                "insecure_ssl": "0",
                "secret": secret,
            },
        }

        response = requests.post(
            f"{self.GITHUB_API_URL}/repos/{resource}/hooks",
            headers=headers,
            json=webhook_data,
        )

        if response.status_code != 201:
            raise Exception(f"Failed to create GitHub webhook: {response.text}")

        webhook_id = response.json()["id"]
        config = response.json()["config"]

        return str(webhook_id), config

    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None:
        webhook_type = self.WebhookType(webhook.webhook_type)
        if webhook.credentials_id != credentials.id:
            raise ValueError(
                f"Webhook #{webhook.id} does not belong to credentials {credentials.id}"
            )

        headers = {
            **self.GITHUB_API_DEFAULT_HEADERS,
            "Authorization": credentials.bearer(),
        }

        if webhook_type == self.WebhookType.REPO:
            repo = webhook.resource
            delete_url = f"{self.GITHUB_API_URL}/repos/{repo}/hooks/{webhook.provider_webhook_id}"  # noqa
        else:
            raise NotImplementedError(
                f"Unsupported webhook type '{webhook.webhook_type}'"
            )

        response = requests.delete(delete_url, headers=headers)

        if response.status_code not in [204, 404]:
            # 204 means successful deletion, 404 means the webhook was already deleted
            raise Exception(f"Failed to delete GitHub webhook: {response.text}")

        # If we reach here, the webhook was successfully deleted or didn't exist
