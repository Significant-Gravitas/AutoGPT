"""Recover a paid Checkout's owned and origin-validated return destination."""

import logging
from urllib.parse import urlsplit

import stripe

from backend.data.credit import CHECKOUT_SESSION_ID_PLACEHOLDER
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_activation_models import ActivationPreviewRequest
from backend.data.subscription_activation_stripe import BillingSubscription
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


async def activation_return_to(subscription: BillingSubscription) -> str:
    destination = subscription.metadata.get("pro_activation_return_to")
    # Older subscriptions stored the success URL template; the Session has it filled.
    if destination and CHECKOUT_SESSION_ID_PLACEHOLDER not in destination:
        return ActivationPreviewRequest(return_to=destination).return_to
    try:
        sessions = await stripe_call(
            stripe.checkout.Session.list_async,
            subscription=subscription.id,
            customer=subscription.customer,
            limit=100,
        )
        async for session in stripe_list_items(sessions):
            if (
                session.customer != subscription.customer
                or session.subscription != subscription.id
                or session.mode != "subscription"
                or not session.success_url
            ):
                continue
            return checkout_destination(
                session.success_url.replace(CHECKOUT_SESSION_ID_PLACEHOLDER, session.id)
            )
    except stripe.StripeError:
        # The payment status is read after this; a lookup failure must not block it.
        logger.warning(f"Checkout Session lookup failed for {subscription.id}")
    return "/settings/billing"


def checkout_destination(success_url: str) -> str:
    config = Settings().config
    origin = config.frontend_base_url or config.platform_base_url
    expected = urlsplit(origin or "")
    parsed = urlsplit(success_url)
    if (
        not origin
        or parsed.scheme not in ("http", "https")
        or parsed.scheme != expected.scheme
        or parsed.netloc != expected.netloc
        or "@" in parsed.netloc
        or "\\" in success_url
        or any(ord(char) < 32 for char in success_url)
    ):
        return "/settings/billing"
    path = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
    path += f"#{parsed.fragment}" if parsed.fragment else ""
    return ActivationPreviewRequest(return_to=path).return_to
