"""MailerLite's webhook: an unsubscribe there is a refusal of marketing here.

Someone who clicks "unsubscribe" in a MailerLite email has refused marketing as
surely as someone who opted out at signup, so the account is marked opted out
(`marketingOptOutSource = "email_unsubscribe"`) and the MailerLite gate keeps
it out from then on (`notifications/consent.py`).

Authenticated by MailerLite's signature, not by a user: the `Signature` header
is the hex HMAC-SHA256 of the raw body under the webhook's secret
(https://developers.mailerlite.com/docs/webhooks.html). With no secret
configured every call is refused, since an empty key would let anyone sign.

MailerLite retries anything but a 2xx, so a call that changes nothing (an
unknown address, another event, a repeat) is still a 200.
"""

import hashlib
import hmac
import logging

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, TypeAdapter, ValidationError

from backend.data.user import (
    MARKETING_OPT_OUT_SOURCE_EMAIL_UNSUBSCRIBE,
    record_marketing_opt_out_by_email,
)
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# No auth dependency: MailerLite has no user, and the signature is the check.
router = APIRouter()

UNSUBSCRIBED_EVENT = "subscriber.unsubscribed"


class MailerLiteSubscriber(BaseModel):
    email: str


class MailerLiteEvent(BaseModel):
    """One event. A single delivery carries the subscriber itself, named by
    `event`; a batched one wraps it as `subscriber`, named by `type`."""

    event: str | None = None
    type: str | None = None
    email: str | None = None
    subscriber: MailerLiteSubscriber | None = None

    @property
    def name(self) -> str | None:
        return self.event or self.type

    @property
    def address(self) -> str | None:
        return self.subscriber.email if self.subscriber else self.email

    def unsubscribed_addresses(self) -> list[str]:
        if self.name != UNSUBSCRIBED_EVENT or not self.address:
            return []
        return [self.address]


class MailerLiteBatch(BaseModel):
    events: list[MailerLiteEvent]

    def unsubscribed_addresses(self) -> list[str]:
        return [a for event in self.events for a in event.unsubscribed_addresses()]


_PAYLOAD = TypeAdapter(MailerLiteBatch | MailerLiteEvent)


@router.post("/mailerlite/webhook", summary="Handle MailerLite webhooks")
async def mailerlite_webhook(request: Request) -> Response:
    secret = settings.secrets.mailerlite_webhook_secret
    if not secret:
        logger.error(
            "mailerlite_webhook: MAILERLITE_WEBHOOK_SECRET is not configured; "
            "rejecting the request"
        )
        raise HTTPException(status_code=503, detail="Webhook not configured")

    body = await request.body()
    if not signature_matches(body, request.headers.get("signature"), secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = _PAYLOAD.validate_json(body)
    except ValidationError:
        raise HTTPException(status_code=400, detail="Invalid payload")

    for address in payload.unsubscribed_addresses():
        user_id = await record_marketing_opt_out_by_email(
            address, MARKETING_OPT_OUT_SOURCE_EMAIL_UNSUBSCRIBE
        )
        if user_id:
            logger.info(
                f"mailerlite_webhook: user {user_id} unsubscribed in MailerLite; "
                "recorded the marketing opt-out"
            )
    return Response(status_code=200)


def signature_matches(body: bytes, signature: str | None, secret: str) -> bool:
    if not signature:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    # Bytes: compare_digest raises on a non-ASCII str, and headers arrive as
    # latin-1, so a junk header would otherwise be a 500, not a refusal.
    return hmac.compare_digest(expected.encode(), signature.strip().lower().encode())
