import logging
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Query, Security
from fastapi.responses import JSONResponse

from backend.data.user import (
    get_user_by_email,
    set_user_email_verification,
    unsubscribe_user_by_token,
)
from backend.server.routers.postmark.models import (
    PostmarkBounceEnum,
    PostmarkBounceWebhook,
    PostmarkClickWebhook,
    PostmarkDeliveryWebhook,
    PostmarkOpenWebhook,
    PostmarkSpamComplaintWebhook,
    PostmarkSubscriptionChangeWebhook,
    PostmarkWebhook,
)
from backend.server.utils.api_key_auth import APIKeyAuthenticator
from backend.util.settings import Settings

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
    try:
        await unsubscribe_user_by_token(token)
    except Exception as e:
        logger.exception("Unsubscribe failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"message": str(e), "hint": "Verify Postmark token settings."},
        )
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
    logger.info(f"Received webhook from Postmark: {webhook}")
    match webhook:
        case PostmarkDeliveryWebhook():
            delivery_handler(webhook)
        case PostmarkBounceWebhook():
            await bounce_handler(webhook)
        case PostmarkSpamComplaintWebhook():
            spam_handler(webhook)
        case PostmarkOpenWebhook():
            open_handler(webhook)
        case PostmarkClickWebhook():
            click_handler(webhook)
        case PostmarkSubscriptionChangeWebhook():
            subscription_handler(webhook)
        case _:
            logger.warning(
                "Unhandled Postmark webhook type %s. Update handler mappings.",
                type(webhook),
            )
            return


async def bounce_handler(event: PostmarkBounceWebhook):
    logger.info(f"Bounce handler {event=}")
    if event.TypeCode in [
        PostmarkBounceEnum.Transient,
        PostmarkBounceEnum.SoftBounce,
        PostmarkBounceEnum.DnsError,
    ]:
        logger.info(
            f"Softish bounce: {event.TypeCode} for {event.Email}, not setting email verification to false"
        )
        return
    logger.info(f"{event.Email=}")
    user = await get_user_by_email(event.Email)
    if not user:
        logger.warning(
            "Received bounce for unknown email %s. Ensure user records are current.",
            event.Email,
        )
        return
    await set_user_email_verification(user.id, False)
    logger.debug(f"Setting email verification to false for user: {user.id}")


def spam_handler(event: PostmarkSpamComplaintWebhook):
    logger.info("Spam handler")
    pass


def delivery_handler(event: PostmarkDeliveryWebhook):
    logger.info("Delivery handler")
    pass


def open_handler(event: PostmarkOpenWebhook):
    logger.info("Open handler")
    pass


def click_handler(event: PostmarkClickWebhook):
    logger.info("Click handler")
    pass


def subscription_handler(event: PostmarkSubscriptionChangeWebhook):
    logger.info("Subscription handler")
    pass
