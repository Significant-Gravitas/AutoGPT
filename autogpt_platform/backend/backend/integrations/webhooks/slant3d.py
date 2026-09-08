import hashlib
import hmac
import logging
import time
from urllib.parse import quote

from fastapi import HTTPException, Request

from backend.data import integrations
from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks._base import BaseWebhooksManager
from backend.util.request import Requests

logger = logging.getLogger(__name__)


class Slant3DWebhooksManager(BaseWebhooksManager):
    PROVIDER_NAME = ProviderName.SLANT3D
    BASE_URL = "https://slant3dapi.com/v2/api"

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        if not resource:
            raise ValueError("Set platform_id to register a Slant3D v2 webhook")
        platform = await self._platform_request("GET", resource, credentials)
        if platform["data"].get("webhookURL") not in (None, "", ingress_url):
            raise ValueError(
                "This Slant3D platform already has a different webhook URL. "
                "Choose a dedicated platform or remove its existing webhook first."
            )
        await self._platform_request(
            "PATCH",
            resource,
            credentials,
            json={"webhookURL": ingress_url, "webhookSecret": secret},
        )
        return resource, {
            "endpoint": ingress_url,
            "api_version": 2,
            "provider": self.PROVIDER_NAME,
            "events": events,
            "type": webhook_type,
        }

    @classmethod
    async def verify_signature(
        cls, webhook: integrations.Webhook, request: Request
    ) -> None:
        if webhook.config.get("api_version") != 2:
            return
        timestamp = request.headers.get("X-Webhook-Timestamp", "")
        signature = request.headers.get("X-Webhook-Signature-256", "")
        try:
            age = abs(time.time() * 1000 - int(timestamp))
        except ValueError:
            raise HTTPException(403, "Invalid Slant3D webhook timestamp")
        if not webhook.secret or age > 5 * 60 * 1000:
            raise HTTPException(403, "Invalid or expired Slant3D webhook signature")
        signed_payload = timestamp.encode() + b"." + await request.body()
        digest = hmac.new(
            webhook.secret.encode(), signed_payload, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, f"sha256={digest}"):
            raise HTTPException(403, "Invalid Slant3D webhook signature")

    @classmethod
    async def validate_payload(
        cls,
        webhook: integrations.Webhook,
        request: Request,
        credentials: Credentials | None,
    ) -> tuple[dict, str]:
        payload = await request.json()
        if webhook.config.get("api_version") == 2:
            if payload.get("platform_id") != webhook.resource:
                raise ValueError(
                    "Slant3D webhook platform does not match the subscription"
                )
            if payload.get("dummy"):
                return payload, "dummy"
            event = payload["event_type"]
            if not event.startswith("order."):
                return payload, event
            order = payload["data"]["order"]
            return {
                **payload,
                "orderId": order["public_id"],
                "status": order.get("status", event.removeprefix("order.").upper()),
                "trackingNumber": order.get("tracking_number") or "",
                "carrierCode": order.get("carrier_code") or "",
            }, event
        required = ["orderId", "status", "trackingNumber", "carrierCode"]
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        return {
            field: payload[field] for field in required
        }, f"order.{payload['status'].lower()}"

    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None:
        if webhook.config.get("api_version") != 2:
            logger.warning(
                f"Manual deregistration required for legacy Slant3D webhook {webhook.id}"
            )
            return
        platform = await self._platform_request("GET", webhook.resource, credentials)
        if platform["data"].get("webhookURL") == webhook.config["endpoint"]:
            await self._platform_request(
                "PATCH", webhook.resource, credentials, json={"webhookURL": ""}
            )

    async def _platform_request(
        self, method: str, platform_id: str, credentials: Credentials, **kwargs
    ) -> dict:
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("API key is required for Slant3D webhooks")
        response = await Requests(raise_for_status=False, retry_max_attempts=1).request(
            method,
            f"{self.BASE_URL}/platforms/{quote(platform_id, safe='')}",
            headers={
                "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            **kwargs,
        )
        result = response.json()
        if not response.ok or result.get("success") is False:
            raise ValueError(
                f"Slant3D returned error: {result.get('error') or result.get('message')}"
            )
        return result
