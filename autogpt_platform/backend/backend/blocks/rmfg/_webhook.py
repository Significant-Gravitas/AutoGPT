"""Webhook manager for RMFG lifecycle events.

RMFG registers endpoints through its API and returns a signing secret once,
at creation. Deliveries carry ``X-RMFG-Timestamp`` and
``X-RMFG-Signature: v1=<hex HMAC-SHA256 of "<timestamp>.<raw body>">``.
"""

import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, Request
from pydantic import TypeAdapter, ValidationError
from strenum import StrEnum

from backend.data import integrations
from backend.sdk import APIKeyCredentials, BaseWebhooksManager, Credentials, Requests
from backend.util.request import Response

from ._http import ErrorEnvelope

logger = logging.getLogger(__name__)

RMFG_API_URL = "https://api.rmfg.com/v1"
SIGNATURE_HEADER = "X-RMFG-Signature"
TIMESTAMP_HEADER = "X-RMFG-Timestamp"
# How far a delivery's timestamp may drift before it is treated as a replay.
TIMESTAMP_TOLERANCE_SECONDS = 300
# ``webhook.config`` key holding the secret RMFG returns at endpoint creation.
SIGNING_SECRET_KEY = "signing_secret"  # pragma: allowlist secret

# Trigger-block event filter field -> RMFG event type. The platform matches
# deliveries against the filter's field names, so both directions are needed.
EVENT_TYPES: dict[str, str] = {
    "design_ready": "design.ready",
    "design_failed": "design.failed",
    "production_files_ready": "dfm_report.production_files.ready",
    "production_files_failed": "dfm_report.production_files.failed",
    "quote_ready": "quote.ready",
    "quote_failed": "quote.failed",
    "cart_checked_out": "cart.checked_out",
    "order_status_changed": "order.status_changed",
}
FILTER_KEYS: dict[str, str] = {event: key for key, event in EVENT_TYPES.items()}

_JSON_OBJECT = TypeAdapter(dict)


class RMFGWebhookType(StrEnum):
    ACCOUNT = "account"


class RMFGWebhooksManager(BaseWebhooksManager):
    WebhookType = RMFGWebhookType

    @classmethod
    async def verify_signature(
        cls, webhook: integrations.WebhookWithRelations, request: Request
    ) -> None:
        secret = webhook.config.get(SIGNING_SECRET_KEY, "")
        if not secret:
            raise HTTPException(
                status_code=500, detail="RMFG signing secret not configured"
            )

        signature_header = request.headers.get(SIGNATURE_HEADER)
        timestamp = request.headers.get(TIMESTAMP_HEADER)
        if not signature_header or not timestamp:
            raise HTTPException(
                status_code=403,
                detail=f"{SIGNATURE_HEADER} or {TIMESTAMP_HEADER} header is missing",
            )
        _reject_stale_timestamp(timestamp)

        body = await request.body()
        expected = hmac.new(
            secret.encode("utf-8"),
            msg=f"{timestamp}.".encode("utf-8") + body,
            digestmod=hashlib.sha256,
        ).hexdigest()
        if not any(
            _signatures_match(expected, sig) for sig in _v1_signatures(signature_header)
        ):
            raise HTTPException(status_code=403, detail="RMFG signature mismatch")

    @classmethod
    async def validate_payload(
        cls,
        webhook: integrations.Webhook,
        request: Request,
        credentials: Credentials | None = None,
    ) -> tuple[dict, str]:
        try:
            payload = _JSON_OBJECT.validate_python(await request.json())
        except (ValueError, ValidationError) as exc:
            raise HTTPException(
                status_code=400, detail="RMFG webhook body must be a JSON object"
            ) from exc
        event_type = str(payload.get("type") or "")
        if not event_type:
            raise HTTPException(
                status_code=400, detail="RMFG event type missing from payload"
            )
        # Report the filter key so the trigger block's event filter matches;
        # an event this integration does not know simply triggers nothing.
        return payload, FILTER_KEYS.get(event_type, event_type)

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: RMFGWebhookType,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        if not isinstance(credentials, APIKeyCredentials):
            raise ValueError("RMFG webhook registration requires an API key")
        unknown = [event for event in events if event not in EVENT_TYPES]
        if unknown:
            raise ValueError(f"Unknown RMFG events: {', '.join(unknown)}")

        response = await Requests(raise_for_status=False).post(
            f"{RMFG_API_URL}/webhook-endpoints",
            headers=_headers(credentials),
            json={
                "url": ingress_url,
                "events": [EVENT_TYPES[event] for event in events],
                "description": "AutoGPT Platform trigger",
            },
        )
        if not response.ok:
            raise ValueError(
                f"RMFG webhook registration failed: {_error_message(response)}"
            )
        data = response.json()
        # The secret is only returned on creation; without it no delivery
        # could ever be verified, so refuse to keep a half-registered hook.
        if not data.get("id") or not data.get("secret"):
            raise ValueError(
                "RMFG webhook registration returned no endpoint ID or signing secret"
            )
        return str(data["id"]), {SIGNING_SECRET_KEY: data["secret"]}

    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None:
        if not isinstance(credentials, APIKeyCredentials):
            logger.warning("Cannot deregister RMFG webhook: API key required")
            return
        if not webhook.provider_webhook_id:
            return
        response = await Requests(raise_for_status=False).delete(
            f"{RMFG_API_URL}/webhook-endpoints/{webhook.provider_webhook_id}",
            headers=_headers(credentials),
        )
        # 404 means the endpoint is already gone, which is the desired state.
        if response.status not in (200, 204, 404):
            logger.warning(
                f"Failed to deregister RMFG webhook {webhook.provider_webhook_id}: "
                f"{_error_message(response)}"
            )


def _headers(credentials: APIKeyCredentials) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }


def _v1_signatures(header: str) -> list[str]:
    """Extract every ``v1=`` value from a comma-separated signature header."""
    found: list[str] = []
    for part in header.split(","):
        key, _, value = part.partition("=")
        if key.strip() == "v1" and value.strip():
            found.append(value.strip())
    return found


def _signatures_match(expected: str, candidate: str) -> bool:
    """Constant-time compare that treats a non-ASCII candidate as a mismatch."""
    try:
        return hmac.compare_digest(expected, candidate)
    except TypeError:
        return False


def _reject_stale_timestamp(timestamp: str) -> None:
    sent_at = _parse_timestamp(timestamp)
    if sent_at is None:
        raise HTTPException(status_code=403, detail="Invalid RMFG timestamp")
    if abs(time.time() - sent_at) > TIMESTAMP_TOLERANCE_SECONDS:
        raise HTTPException(
            status_code=403, detail="RMFG webhook timestamp is outside the window"
        )


def _parse_timestamp(timestamp: str) -> Optional[float]:
    """Accept epoch seconds or an ISO-8601 instant; return epoch seconds."""
    candidate = timestamp.strip()
    try:
        return float(candidate)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _error_message(response: Response) -> str:
    try:
        message = ErrorEnvelope.model_validate(response.json()).error.message
    except (ValueError, ValidationError):
        message = ""
    return message or f"HTTP {response.status}"
