import logging
from typing import Annotated

from autogpt_libs.auth.middleware import APIKeyValidator
from fastapi import APIRouter, Body, Depends

from backend.server.v2.system_webhooks.models import (
    PostmarkBounceWebhook,
    PostmarkClickWebhook,
    PostmarkDeliveryWebhook,
    PostmarkOpenWebhook,
    PostmarkSpamComplaintWebhook,
    PostmarkSubscriptionChangeWebhook,
    PostmarkWebhook,
)
from backend.util.settings import Settings

settings = Settings()
postmark_validator = APIKeyValidator(
    "X-Postmark-Webhook-Token",
    settings.secrets.postmark_webhook_token,
)

router = APIRouter(dependencies=[Depends(postmark_validator.get_dependency())])


logger = logging.getLogger(__name__)


@router.post("/")
def postmark_webhook_handler(
    webhook: Annotated[
        PostmarkWebhook,
        Body(discriminator="RecordType"),
    ]
):
    logger.info(webhook)
    logger.info(type(webhook))
    match webhook:
        case PostmarkDeliveryWebhook():
            delivery_handler(webhook)
        case PostmarkBounceWebhook():
            bounce_handler(webhook)
        case PostmarkSpamComplaintWebhook():
            spam_handler(webhook)
        case PostmarkOpenWebhook():
            open_handler(webhook)
        case PostmarkClickWebhook():
            click_handler(webhook)
        case PostmarkSubscriptionChangeWebhook():
            subscription_handler(webhook)
        case _:
            logger.warning(f"Unknown webhook type: {type(webhook)}")
            return


def bounce_handler(event: PostmarkBounceWebhook):
    logger.info(f"Bounce handler {event=}")
    pass


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
