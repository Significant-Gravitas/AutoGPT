"""Webhook manager for All Quiet outbound integrations.

All Quiet can optionally sign each delivery. The signature is
``HMAC-SHA256(secret, "<timestamp>:<body>")``, base64-encoded, and All Quiet
sends it under one of two header pairs depending on the format chosen on the
outbound integration:

* All Quiet — ``x-aq-signature`` / ``x-aq-timestamp``
* AWS       — ``x-amzn-event-signature`` / ``x-amzn-event-timestamp``

Both are accepted here so either format works without reconfiguring the block.
"""

import base64
import hashlib
import hmac
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, Request
from pydantic import SecretStr, TypeAdapter, ValidationError
from strenum import StrEnum

from backend.data.integrations import Webhook, WebhookWithRelations
from backend.sdk import Credentials, ManualWebhookManagerBase

logger = logging.getLogger(__name__)

# Header pairs All Quiet may sign with, in (signature, timestamp) order.
SIGNATURE_HEADER_PAIRS = (
    ("x-aq-signature", "x-aq-timestamp"),
    ("x-amzn-event-signature", "x-amzn-event-timestamp"),
)

# How far a signed delivery's timestamp may drift before it is treated as a
# replay. Only enforced when the timestamp parses; the signature itself is
# always checked.
MAX_TIMESTAMP_SKEW = timedelta(minutes=5)

# Epoch seconds or milliseconds, optionally signed or fractional.
_EPOCH_ONLY = re.compile(r"[+-]?\d+(?:\.\d+)?")

_OPTIONAL_SECRET = TypeAdapter(Optional[SecretStr])
_JSON_OBJECT = TypeAdapter(dict)


class AllQuietWebhookType(StrEnum):
    INCIDENT = "incident"


class AllQuietWebhooksManager(ManualWebhookManagerBase):
    WebhookType = AllQuietWebhookType

    # Name of the input field on the trigger block carrying the signing secret.
    # Read at verification time rather than snapshotted at registration, so
    # rotating the secret on the block takes effect immediately.
    SIGNING_SECRET_INPUT = "signing_secret"  # pragma: allowlist secret

    @classmethod
    async def validate_payload(
        cls,
        webhook: Webhook,
        request: Request,
        credentials: Credentials | None = None,
    ) -> tuple[dict, str]:
        """Parse the delivery body.

        All Quiet's outbound webhook body is whatever Handlebars template the
        user configured, so the shape is not fixed — the trigger block reads the
        well-known keys and passes the rest through. Only one event type exists;
        callers filter on the payload's own status/intent instead.
        """
        try:
            payload = _JSON_OBJECT.validate_python(await request.json())
        except (ValueError, ValidationError) as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "All Quiet webhook body must be a JSON object. Check the "
                    "body template on the outbound integration."
                ),
            ) from exc

        return payload, AllQuietWebhookType.INCIDENT

    @classmethod
    async def verify_signature(
        cls, webhook: WebhookWithRelations, request: Request
    ) -> None:
        secret = cls._configured_secret(webhook)
        if not secret:
            # Signing is opt-in on the All Quiet side. With no secret
            # configured the webhook URL is the only credential, matching the
            # platform's other manual webhooks.
            return

        signature, timestamp = cls._signed_headers(request)
        body = await request.body()
        expected = base64.b64encode(
            hmac.new(
                secret.encode("utf-8"),
                msg=f"{timestamp}:".encode("utf-8") + body,
                digestmod=hashlib.sha256,
            ).digest()
        )

        # Compare as bytes: hmac.compare_digest raises TypeError on a str
        # containing non-ASCII, which would escape as a 500 instead of a 403.
        if not hmac.compare_digest(expected, signature.encode("utf-8", "ignore")):
            raise HTTPException(status_code=403, detail="Invalid webhook signature")

        cls._reject_stale_timestamp(timestamp)

    @classmethod
    def _signed_headers(cls, request: Request) -> tuple[str, str]:
        """Return the (signature, timestamp) pair All Quiet signed this with."""
        for signature_header, timestamp_header in SIGNATURE_HEADER_PAIRS:
            signature = request.headers.get(signature_header)
            timestamp = request.headers.get(timestamp_header)
            if signature and timestamp:
                return signature, timestamp

        accepted = ", ".join(pair[0] for pair in SIGNATURE_HEADER_PAIRS)
        raise HTTPException(
            status_code=403,
            detail=(
                "Webhook is configured with a signing secret but the request "
                f"carries no signature. Expected one of: {accepted} (with its "
                "matching timestamp header)."
            ),
        )

    @classmethod
    def _reject_stale_timestamp(cls, timestamp: str) -> None:
        """Reject replays of an otherwise validly signed delivery.

        Fails closed: a signing secret is configured, so a timestamp we cannot
        place in time is treated as a failed check rather than waved through.
        Letting it pass would leave the replay window permanently open for any
        sender that varies its timestamp format.
        """
        sent_at = _parse_timestamp(timestamp)
        if sent_at is None:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Webhook timestamp is not in a recognized format, so the "
                    "delivery cannot be checked for replay."
                ),
            )

        if abs(datetime.now(timezone.utc) - sent_at) > MAX_TIMESTAMP_SKEW:
            raise HTTPException(
                status_code=403,
                detail="Webhook timestamp is outside the accepted window",
            )

    @classmethod
    def _configured_secret(cls, webhook: WebhookWithRelations) -> str | None:
        """Find the signing secret set on any node or preset using this webhook."""
        sources = [node.input_default for node in webhook.triggered_nodes] + [
            preset.inputs for preset in webhook.triggered_presets
        ]

        found: list[str] = []
        for source in sources:
            raw = source.get(cls.SIGNING_SECRET_INPUT)
            # Stored values arrive as plain strings or SecretStr depending on
            # the serialization path; coerce both to a plain string.
            try:
                secret = _OPTIONAL_SECRET.validate_python(raw)
            except ValidationError as exc:
                # A secret is configured but unreadable. Treating that as "no
                # secret" would silently downgrade to accepting any delivery,
                # so fail closed instead.
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "A signing secret is configured on this webhook but "
                        "could not be read, so the delivery cannot be verified."
                    ),
                ) from exc
            if secret and secret.get_secret_value().strip():
                found.append(secret.get_secret_value())

        if not found:
            return None

        # Compute the distinct count before logging so no secret-derived value
        # flows into the logger call args.
        distinct_count = len(set(found))
        if distinct_count > 1:
            # Only one signature can be checked, so picking a winner would make
            # verification depend on node ordering. Refuse instead of silently
            # enforcing one target's secret against every delivery.
            logger.warning(
                "Webhook %s has %d distinct signing_secret values across "
                "attached targets; refusing the delivery.",
                webhook.id,
                distinct_count,
            )
            raise HTTPException(
                status_code=403,
                detail=(
                    "This webhook is attached to targets configured with "
                    "different signing secrets. All targets sharing a webhook "
                    "must use the same secret."
                ),
            )
        return found[0]


def _parse_timestamp(timestamp: str) -> Optional[datetime]:
    """Parse the timestamp formats All Quiet signs with, as an aware UTC datetime.

    Covers the ISO-8601 spellings used by the All Quiet header pair
    (``2023-12-17T11:51:08.844Z``, and the ``...T11:51:08.000Z`` form the AWS
    pair documents) plus epoch seconds/milliseconds, which AWS-style senders
    commonly use. Returns None when the value matches none of them.
    """
    candidate = timestamp.strip()
    if not candidate:
        return None

    # An all-digit value is an epoch, and must not be offered to
    # `fromisoformat` first. Python 3.11+ accepts ISO 8601 *basic* format, so a
    # 13-digit millisecond value whose leading digits happen to spell a valid
    # YYYYMMDD parses as a date centuries in the past — 1787121651526 becomes
    # 1787-12-16 — which then fails the replay-window check. Whether a given
    # millisecond value does that depends on the wall clock, so the symptom is
    # an intermittent 403 on valid deliveries.
    if not _EPOCH_ONLY.fullmatch(candidate):
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    try:
        epoch = float(candidate)
    except ValueError:
        return None

    # Heuristic: values this large can only be milliseconds. 1e11 seconds is
    # year 5138, while 1e11 ms is 1973, so the split is unambiguous in practice.
    if abs(epoch) >= 1e11:
        epoch /= 1000.0
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
