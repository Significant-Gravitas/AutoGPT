import hashlib
import hmac
import logging
import time

from fastapi import HTTPException, Request
from strenum import StrEnum

from backend.data import integrations
from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.providers import ProviderName
from backend.util.request import Requests

from ._base import BaseWebhooksManager

logger = logging.getLogger(__name__)

STRIPE_API_URL = "https://api.stripe.com/v1"
# Tolerance window for Stripe timestamp verification (5 minutes)
STRIPE_TIMESTAMP_TOLERANCE = 300


class StripeWebhookType(StrEnum):
    ACCOUNT = "account"


class StripeWebhooksManager(BaseWebhooksManager[StripeWebhookType]):
    PROVIDER_NAME = ProviderName.STRIPE
    WebhookType = StripeWebhookType

    @classmethod
    async def verify_signature(
        cls,
        webhook: integrations.Webhook,
        request: Request,
    ) -> None:
        sig_header = request.headers.get("Stripe-Signature")
        if not sig_header:
            raise HTTPException(
                status_code=403, detail="Stripe-Signature header is missing"
            )

        # Stripe stores its signing secret in config, not the platform-generated secret
        signing_secret = webhook.config.get("signing_secret", "")
        if not signing_secret:
            raise HTTPException(
                status_code=500, detail="Stripe signing secret not configured"
            )

        # Parse Stripe-Signature header: t=timestamp,v1=signature
        parts = {k: v for k, v in (p.split("=", 1) for p in sig_header.split(","))}
        timestamp = parts.get("t")
        v1_sig = parts.get("v1")
        if not timestamp or not v1_sig:
            raise HTTPException(
                status_code=403, detail="Invalid Stripe-Signature format"
            )

        # Reject stale timestamps
        if abs(time.time() - int(timestamp)) > STRIPE_TIMESTAMP_TOLERANCE:
            raise HTTPException(
                status_code=403, detail="Stripe webhook timestamp is too old"
            )

        payload_body = await request.body()
        signed_payload = f"{timestamp}.{payload_body.decode('utf-8')}"
        expected = hmac.new(
            signing_secret.encode("utf-8"),
            msg=signed_payload.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(expected, v1_sig):
            raise HTTPException(
                status_code=403, detail="Stripe webhook signature mismatch"
            )

    @classmethod
    async def validate_payload(
        cls,
        webhook: integrations.Webhook,
        request: Request,
        credentials: Credentials | None,
    ) -> tuple[dict, str]:
        payload = await request.json()
        event_type = payload.get("type", "")
        if not event_type:
            raise HTTPException(
                status_code=400, detail="Stripe event type missing from payload"
            )
        return payload, event_type

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: StripeWebhookType,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("Stripe webhook registration requires an API key")

        api_key = credentials.api_key.get_secret_value()

        # Build form-encoded body for Stripe API (requires form encoding, not JSON)
        form_data = f"url={ingress_url}"
        for event in events:
            form_data += f"&enabled_events[]={event}"

        response = await Requests(raise_for_status=False).post(
            f"{STRIPE_API_URL}/webhook_endpoints",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=form_data,
        )

        if not response.ok:
            error = response.json().get("error", {})
            raise ValueError(
                f"Stripe webhook registration failed: {error.get('message', response.status)}"
            )

        data = response.json()
        # Store Stripe's signing secret in config — it's only returned on creation
        return data["id"], {"signing_secret": data["secret"]}

    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None:
        if not isinstance(credentials, APIKeyCredentials):
            logger.warning("Cannot deregister Stripe webhook: API key credentials required")
            return

        endpoint_id = webhook.provider_webhook_id
        if not endpoint_id:
            return

        api_key = credentials.api_key.get_secret_value()
        response = await Requests(raise_for_status=False).delete(
            f"{STRIPE_API_URL}/webhook_endpoints/{endpoint_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if response.status not in (200, 404):
            error = response.json().get("error", {})
            logger.warning(
                f"Failed to deregister Stripe webhook {endpoint_id}: "
                f"{error.get('message', response.status)}"
            )
