import hashlib
import hmac
import logging
import time
from urllib.parse import urlencode

from fastapi import HTTPException, Request
from strenum import StrEnum

from backend.data import integrations
from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.providers import ProviderName
from backend.util.request import Requests, Response

from ._base import BaseWebhooksManager

logger = logging.getLogger(__name__)

STRIPE_API_URL = "https://api.stripe.com/v1"
# Tolerance window for Stripe timestamp verification (5 minutes)
STRIPE_TIMESTAMP_TOLERANCE = 300
# `webhook.config` key holding the secret Stripe returns at endpoint creation.
# Shared by the writer (`_register_webhook`) and the reader (`verify_signature`).
SIGNING_SECRET_KEY = "signing_secret"  # pragma: allowlist secret


class StripeWebhookType(StrEnum):
    ACCOUNT = "account"


def _signatures_match(expected: str, candidate: str) -> bool:
    """Constant-time compare that treats a non-ASCII candidate as a mismatch."""
    try:
        return hmac.compare_digest(expected, candidate)
    except TypeError:
        return False


def _error_message(response: Response) -> str:
    """
    Pull Stripe's error message out of a failed response.

    Falls back to the status code: an error from a proxy in front of Stripe
    (or a 5xx HTML page) isn't necessarily JSON, and a JSONDecodeError here
    would mask the actual failure.
    """
    try:
        message = response.json().get("error", {}).get("message")
    except Exception:
        message = None
    return message or f"HTTP {response.status}"


class StripeWebhooksManager(BaseWebhooksManager):
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
        signing_secret = webhook.config.get(SIGNING_SECRET_KEY, "")
        if not signing_secret:
            raise HTTPException(
                status_code=500, detail="Stripe signing secret not configured"
            )

        # Parse Stripe-Signature header: t=timestamp,v1=signature. During a
        # signing-secret rotation Stripe signs each delivery with every active
        # secret, so there can be more than one v1 entry.
        timestamp = ""
        v1_sigs: list[str] = []
        for part in sig_header.split(","):
            key, _, value = part.partition("=")
            key = key.strip()
            if key == "t":
                timestamp = value.strip()
            elif key == "v1":
                v1_sigs.append(value.strip())
        if not timestamp or not v1_sigs:
            raise HTTPException(
                status_code=403, detail="Invalid Stripe-Signature format"
            )

        # Reject stale timestamps. `int()` parses arbitrarily large values, so
        # subtracting one from a float raises OverflowError rather than
        # ValueError — both are just a malformed header, not a server fault.
        try:
            age = abs(time.time() - int(timestamp))
        except (ValueError, OverflowError):
            raise HTTPException(
                status_code=403, detail="Invalid Stripe-Signature timestamp"
            )
        if age > STRIPE_TIMESTAMP_TOLERANCE:
            raise HTTPException(
                status_code=403, detail="Stripe webhook timestamp is too old"
            )

        # Sign the raw bytes: decoding and re-encoding the body would transcode
        # the whole payload for nothing, and blow up on a non-UTF-8 body.
        payload_body = await request.body()
        expected = hmac.new(
            signing_secret.encode("utf-8"),
            msg=f"{timestamp}.".encode("utf-8") + payload_body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        # `compare_digest` raises TypeError on a non-ASCII str, and Starlette
        # decodes headers as latin-1 — so an arbitrary `v1=<non-ascii>` would
        # turn a 403 into an unhandled 500 on a public endpoint.
        if not any(_signatures_match(expected, sig) for sig in v1_sigs):
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

        # Stripe's API takes form encoding, not JSON
        form_data = urlencode(
            [("url", ingress_url)] + [("enabled_events[]", event) for event in events]
        )

        response = await Requests(raise_for_status=False).post(
            f"{STRIPE_API_URL}/webhook_endpoints",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=form_data,
        )

        if not response.ok:
            raise ValueError(
                f"Stripe webhook registration failed: {_error_message(response)}"
            )

        data = response.json()
        # Store Stripe's signing secret in config — it's only returned on creation,
        # so a 200 without it leaves a webhook that can never verify a delivery.
        if not data.get("id") or not data.get("secret"):
            raise ValueError(
                "Stripe webhook registration returned no endpoint ID or signing secret"
            )
        return data["id"], {SIGNING_SECRET_KEY: data["secret"]}

    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None:
        if not isinstance(credentials, APIKeyCredentials):
            logger.warning(
                "Cannot deregister Stripe webhook: API key credentials required"
            )
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
            logger.warning(
                f"Failed to deregister Stripe webhook {endpoint_id}: "
                f"{_error_message(response)}"
            )
