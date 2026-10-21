import logging
from types import TracebackType
from typing import Annotated, Self

from fastapi import APIRouter, Body, Query, Security
from fastapi.responses import JSONResponse

from backend.api.utils.api_key_auth import APIKeyAuthenticator
from backend.data.user import (
    get_user_by_email,
    set_user_email_verification,
    unsubscribe_user_by_token,
)
from backend.util.posthog_events import PostHogEvent
from backend.util.settings import Settings

from .analytics import safe_link, track_email_event
from .models import (
    PostmarkBounceEnum,
    PostmarkBounceWebhook,
    PostmarkClickWebhook,
    PostmarkDeliveryWebhook,
    PostmarkOpenWebhook,
    PostmarkSpamComplaintWebhook,
    PostmarkSubscriptionChangeWebhook,
    PostmarkWebhook,
)

logger = logging.getLogger(__name__)
settings = Settings()

router = APIRouter()

postmark_api_key_auth = APIKeyAuthenticator(
    "X-Postmark-Webhook-Token",
    settings.secrets.postmark_webhook_token,
)


@router.post("/unsubscribe", summary="One Click Email Unsubscribe")
async def unsubscribe_via_one_click(token: Annotated[str, Query()]):
    logger.info("Received unsubscribe request from One Click Unsubscribe")
    await unsubscribe_user_by_token(token)
    return JSONResponse(status_code=200, content={"status": "ok"})


@router.post(
    "/",
    dependencies=[Security(postmark_api_key_auth)],
    summary="Handle Postmark Email Webhooks",
)
async def postmark_webhook_handler(
    webhook: Annotated[
        PostmarkWebhook,
        Body(discriminator="RecordType"),
    ]
):
    # The payload holds the recipient's address; keep it out of the logs.
    logger.info(
        "Received %s webhook from Postmark for message %s",
        webhook.RecordType,
        webhook.MessageID,
    )
    with _RetryOnFailure(webhook) as boundary:
        await _handle(webhook)
    if boundary.failed:
        return JSONResponse(status_code=503, content={"status": "retry"})


class _RetryOnFailure:
    """Turns any failure while handling a webhook into a retry, without the
    failure reaching Sentry.

    Neither re-raised nor logged with its traceback: Sentry would send every
    frame's locals, and the webhook in them holds the recipient's address.
    Only the record type, message id and error type are logged; Postmark
    retries anything but a 200. Cancellation still propagates.
    """

    def __init__(self, webhook: PostmarkWebhook):
        self.webhook = webhook
        self.failed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is None or not issubclass(exc_type, Exception):
            return False
        logger.error(
            "Postmark %s webhook for message %s failed: %s",
            self.webhook.RecordType,
            self.webhook.MessageID,
            exc_type.__name__,
        )
        self.failed = True
        return True


async def _handle(webhook: PostmarkWebhook) -> None:
    match webhook:
        case PostmarkDeliveryWebhook():
            await delivery_handler(webhook)
        case PostmarkBounceWebhook():
            await bounce_handler(webhook)
        case PostmarkSpamComplaintWebhook():
            spam_handler(webhook)
        case PostmarkOpenWebhook():
            await open_handler(webhook)
        case PostmarkClickWebhook():
            await click_handler(webhook)
        case PostmarkSubscriptionChangeWebhook():
            subscription_handler(webhook)
        case _:
            logger.warning(
                "Unhandled Postmark webhook type %s. Update handler mappings.",
                type(webhook),
            )


async def _user_id_for(recipient: str, message_id: str) -> str | None:
    """The user an email went to, or None for an address with no account,
    such as the internal Ops inbox."""
    user = await get_user_by_email(recipient)
    if not user:
        logger.info("Postmark message %s went to no known user", message_id)
        return None
    return user.id


async def bounce_handler(event: PostmarkBounceWebhook):
    user_id = await _user_id_for(event.Email, event.MessageID)
    if not user_id:
        return
    track_email_event(
        user_id,
        PostHogEvent.EMAIL_BOUNCED,
        event,
        occurred_at=event.BouncedAt,
        properties={"bounce_type": event.Type},
    )
    if event.TypeCode in [
        PostmarkBounceEnum.Transient,
        PostmarkBounceEnum.SoftBounce,
        PostmarkBounceEnum.DnsError,
    ]:
        logger.info(
            "Softish bounce %s for user %s, not setting email verification to false",
            event.TypeCode,
            user_id,
        )
        return
    await set_user_email_verification(user_id, False)
    logger.debug(f"Setting email verification to false for user: {user_id}")


def spam_handler(event: PostmarkSpamComplaintWebhook):
    logger.info("Spam handler")


async def delivery_handler(event: PostmarkDeliveryWebhook):
    if user_id := await _user_id_for(event.Recipient, event.MessageID):
        track_email_event(
            user_id,
            PostHogEvent.EMAIL_DELIVERED,
            event,
            occurred_at=event.DeliveredAt,
        )


async def open_handler(event: PostmarkOpenWebhook):
    if user_id := await _user_id_for(event.Recipient, event.MessageID):
        track_email_event(
            user_id,
            PostHogEvent.EMAIL_OPENED,
            event,
            occurred_at=event.ReceivedAt,
            properties={"first_open": event.FirstOpen},
        )


async def click_handler(event: PostmarkClickWebhook):
    if user_id := await _user_id_for(event.Recipient, event.MessageID):
        track_email_event(
            user_id,
            PostHogEvent.EMAIL_CLICKED,
            event,
            occurred_at=event.ReceivedAt,
            properties={"link": safe_link(event.OriginalLink)},
        )


def subscription_handler(event: PostmarkSubscriptionChangeWebhook):
    logger.info("Subscription handler")
