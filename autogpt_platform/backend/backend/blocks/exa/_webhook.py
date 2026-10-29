"""
Exa Webhook Manager implementation.

Two kinds of Exa webhook arrive through this manager:

* ``webset`` -- registered automatically through the Websets API
  (``POST https://api.exa.ai/websets/v0/webhooks``). Exa returns a signing
  secret on creation, which is stored in ``config["exa_secret"]``.
* ``search_monitor`` -- a manual webhook. The user pastes the block's URL into a
  Search Monitor's ``webhook.url``; Exa returns that monitor's ``webhookSecret``
  once, at creation, and the user pastes it into the trigger block's
  ``webhook_secret`` input, which is read here at verification time.

Both are signed the same way: an ``Exa-Signature: t=<unix_ts>,v1=<hex>[,v1=...]``
header carrying HMAC-SHA256 of ``"<t>.<raw body>"`` keyed with the secret.
See https://exa.ai/docs/websets/api/webhooks/verifying-signatures and
https://exa.ai/docs/monitors/quickstart (Receive webhook updates).
"""

import hashlib
import hmac
import logging
from enum import Enum

from fastapi import HTTPException, Request
from pydantic import SecretStr, TypeAdapter, ValidationError

from backend.data.integrations import WebhookWithRelations
from backend.data.model import Credentials
from backend.sdk import APIKeyCredentials, BaseWebhooksManager, ProviderName, Webhook

from ._exa_rest import WEBSETS_API_BASE, ExaApiError, exa_request, path_segment
from ._platform_key import require_own_exa_key

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "Exa-Signature"
# Input on the Search Monitor trigger block that holds the monitor's secret.
SEARCH_MONITOR_SECRET_INPUT = "webhook_secret"  # pragma: allowlist secret

_OPTIONAL_SECRET: TypeAdapter[SecretStr | None] = TypeAdapter(SecretStr | None)
_JSON_OBJECT = TypeAdapter(dict)


class ExaEventType(str, Enum):
    """Event types Exa can deliver (the Websets API ``EventType`` list)."""

    WEBSET_CREATED = "webset.created"
    WEBSET_DELETED = "webset.deleted"
    WEBSET_PAUSED = "webset.paused"
    WEBSET_IDLE = "webset.idle"
    WEBSET_SEARCH_CREATED = "webset.search.created"
    WEBSET_SEARCH_CANCELED = "webset.search.canceled"
    WEBSET_SEARCH_COMPLETED = "webset.search.completed"
    WEBSET_SEARCH_UPDATED = "webset.search.updated"
    IMPORT_CREATED = "import.created"
    IMPORT_COMPLETED = "import.completed"
    # Legacy: not in Exa's current EventType list and never delivered. Kept so
    # saved values stay valid; it is never sent to Exa.
    IMPORT_PROCESSING = "import.processing"
    WEBSET_ITEM_CREATED = "webset.item.created"
    WEBSET_ITEM_ENRICHED = "webset.item.enriched"
    WEBSET_EXPORT_CREATED = "webset.export.created"
    WEBSET_EXPORT_COMPLETED = "webset.export.completed"
    MONITOR_CREATED = "monitor.created"
    MONITOR_UPDATED = "monitor.updated"
    MONITOR_DELETED = "monitor.deleted"
    MONITOR_RUN_CREATED = "monitor.run.created"
    MONITOR_RUN_COMPLETED = "monitor.run.completed"


LEGACY_EVENT_TYPES = frozenset({ExaEventType.IMPORT_PROCESSING})

# Event filter field name on the Exa trigger blocks -> the Exa event it selects.
# The platform stores a webhook's events and matches deliveries by these field
# names, so registration translates them to Exa's names and `validate_payload`
# translates each delivery back.
FILTER_FIELD_EVENTS: dict[str, ExaEventType] = {
    "webset_created": ExaEventType.WEBSET_CREATED,
    "webset_deleted": ExaEventType.WEBSET_DELETED,
    "webset_paused": ExaEventType.WEBSET_PAUSED,
    "webset_idle": ExaEventType.WEBSET_IDLE,
    "search_created": ExaEventType.WEBSET_SEARCH_CREATED,
    "search_completed": ExaEventType.WEBSET_SEARCH_COMPLETED,
    "search_canceled": ExaEventType.WEBSET_SEARCH_CANCELED,
    "search_updated": ExaEventType.WEBSET_SEARCH_UPDATED,
    "item_created": ExaEventType.WEBSET_ITEM_CREATED,
    "item_enriched": ExaEventType.WEBSET_ITEM_ENRICHED,
    "export_created": ExaEventType.WEBSET_EXPORT_CREATED,
    "export_completed": ExaEventType.WEBSET_EXPORT_COMPLETED,
    "import_created": ExaEventType.IMPORT_CREATED,
    "import_completed": ExaEventType.IMPORT_COMPLETED,
    "import_processing": ExaEventType.IMPORT_PROCESSING,
    "monitor_created": ExaEventType.MONITOR_CREATED,
    "monitor_updated": ExaEventType.MONITOR_UPDATED,
    "monitor_deleted": ExaEventType.MONITOR_DELETED,
    "monitor_run_created": ExaEventType.MONITOR_RUN_CREATED,
    "monitor_run_completed": ExaEventType.MONITOR_RUN_COMPLETED,
}
_EVENT_FILTER_FIELDS = {
    event.value: name for name, event in FILTER_FIELD_EVENTS.items()
}


def exa_event_names(filter_fields: list[str]) -> list[str]:
    """Translate enabled filter fields to the Exa event names to subscribe to.

    Legacy events Exa no longer emits are dropped, since Exa rejects them.
    """
    unknown = [name for name in filter_fields if name not in FILTER_FIELD_EVENTS]
    if unknown:
        raise ValueError(f"Unknown Exa event filter fields: {', '.join(unknown)}")
    events = [FILTER_FIELD_EVENTS[name] for name in filter_fields]
    return [event.value for event in events if event not in LEGACY_EVENT_TYPES]


def filter_field_for(event_type: str) -> str:
    """Map an Exa event name to the filter field that selects it.

    Unknown event names are returned unchanged, so they match no filter.
    """
    return _EVENT_FILTER_FIELDS.get(event_type, event_type)


class ExaWebhookManager(BaseWebhooksManager):
    """Webhook manager for Exa Websets webhooks and Search Monitor deliveries."""

    PROVIDER_NAME = ProviderName("exa")

    class WebhookType(str, Enum):
        WEBSET = "webset"
        SEARCH_MONITOR = "search_monitor"

    @classmethod
    async def verify_signature(
        cls, webhook: WebhookWithRelations, request: Request
    ) -> None:
        if webhook.webhook_type == cls.WebhookType.SEARCH_MONITOR:
            secret = _search_monitor_secret(webhook)
            if not secret:
                # The secret only exists once the monitor has been created, so
                # until it is pasted into the block the URL is the only
                # credential, as for the platform's other manual webhooks.
                return
        else:
            secret = webhook.config.get("exa_secret")
            if not secret:
                raise HTTPException(
                    status_code=403,
                    detail="Webhook is missing its Exa signing secret; "
                    "re-register the webhook",
                )
        verify_exa_signature(
            secret, request.headers.get(SIGNATURE_HEADER), await request.body()
        )

    @classmethod
    async def validate_payload(
        cls, webhook: Webhook, request: Request, credentials: Credentials | None
    ) -> tuple[dict, str]:
        """Parse a delivery and return it with its event's filter field name."""
        try:
            payload = _JSON_OBJECT.validate_python(await request.json())
        except (ValueError, ValidationError) as exc:
            raise HTTPException(
                status_code=400, detail="Exa webhook body must be a JSON object"
            ) from exc
        event_type = str(payload.get("type") or payload.get("eventType") or "unknown")
        return payload, filter_field_for(event_type)

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        """Create a Websets webhook pointing at our ingress URL."""
        if not isinstance(credentials, APIKeyCredentials):
            raise TypeError("Exa webhooks require API key credentials")
        # A Websets webhook receives every event in the Exa team, which on the
        # shared platform key would include other users' websets.
        require_own_exa_key(credentials, "The Exa Webset trigger")
        exa_events = exa_event_names(events)
        if not exa_events:
            raise ValueError(
                "Select at least one event Exa still sends; "
                "'import processing' is legacy and is never delivered"
            )

        webhook_data = await exa_request(
            credentials,
            "POST",
            f"{WEBSETS_API_BASE}/webhooks",
            body={
                "url": ingress_url,
                "events": exa_events,
                "metadata": {"resource": resource, "webhook_type": webhook_type},
            },
        )
        exa_secret = webhook_data.get("secret")
        if not exa_secret:
            raise ValueError("Exa did not return a signing secret for the webhook")
        return webhook_data["id"], {
            "events": exa_events,
            "resource": resource,
            "exa_secret": exa_secret,
        }

    async def _deregister_webhook(
        self, webhook: Webhook, credentials: Credentials
    ) -> None:
        """Delete the Websets webhook; one already gone counts as deleted."""
        if not webhook.provider_webhook_id:
            # Search Monitor webhooks are manual: nothing was registered at Exa.
            return
        if not isinstance(credentials, APIKeyCredentials):
            raise TypeError("Exa webhooks require API key credentials")
        try:
            await exa_request(
                credentials,
                "DELETE",
                f"{WEBSETS_API_BASE}/webhooks/{path_segment(webhook.provider_webhook_id)}",
            )
        except ExaApiError as exc:
            if exc.status != 404:
                raise


def verify_exa_signature(
    secret: str, signature_header: str | None, body: bytes
) -> None:
    """Raise 403 unless ``signature_header`` signs ``body`` with ``secret``."""
    if not signature_header:
        raise HTTPException(
            status_code=403, detail=f"Missing {SIGNATURE_HEADER} header"
        )

    timestamp: str | None = None
    provided_signatures: list[str] = []
    for part in signature_header.split(","):
        key, _, value = part.strip().partition("=")
        if key == "t":
            timestamp = value
        elif key == "v1":
            provided_signatures.append(value)

    if not timestamp or not provided_signatures:
        raise HTTPException(
            status_code=403, detail=f"Malformed {SIGNATURE_HEADER} header"
        )

    signed_payload = f"{timestamp}.".encode() + body
    expected = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    # Compare as bytes: compare_digest raises TypeError on non-ASCII str input,
    # which would surface as a 500 instead of a 403.
    if not any(
        hmac.compare_digest(expected.encode(), sig.encode("utf-8", "ignore"))
        for sig in provided_signatures
    ):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")


def _search_monitor_secret(webhook: WebhookWithRelations) -> str | None:
    """Return the monitor secret set on the nodes/presets using this webhook."""
    sources = [node.input_default for node in webhook.triggered_nodes] + [
        preset.inputs for preset in webhook.triggered_presets
    ]
    found = {
        secret
        for secret in (
            _read_secret(source.get(SEARCH_MONITOR_SECRET_INPUT)) for source in sources
        )
        if secret
    }
    if len(found) > 1:
        # Only one signature can be checked; picking one would make
        # verification depend on node order, so refuse instead.
        logger.warning(
            "Exa webhook %s has %d distinct webhook secrets across its targets; "
            "refusing the delivery.",
            webhook.id,
            len(found),
        )
        raise HTTPException(
            status_code=403,
            detail="All Exa Search Monitor triggers sharing this webhook must "
            "use the same webhook secret",
        )
    return next(iter(found), None)


def _read_secret(raw: object) -> str | None:
    """Coerce a stored secret input (str or SecretStr) to a plain string."""
    try:
        secret = _OPTIONAL_SECRET.validate_python(raw)
    except ValidationError as exc:
        # A secret is set but unreadable. Treating that as "no secret" would
        # silently accept unsigned deliveries, so fail closed.
        raise HTTPException(
            status_code=403,
            detail="The Exa webhook secret on this trigger could not be read",
        ) from exc
    if secret is None:
        return None
    return secret.get_secret_value().strip() or None
