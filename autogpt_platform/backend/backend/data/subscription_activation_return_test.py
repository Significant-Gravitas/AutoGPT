from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import stripe

from backend.data import subscription_activation_return as destination
from backend.data.subscription_activation_stripe import BillingSubscription


def subscription(metadata=None):
    return BillingSubscription.model_validate(
        {
            "id": "sub_1",
            "customer": "cus_1",
            "status": "active",
            "metadata": metadata or {},
            "items": {"data": []},
        }
    )


@pytest.mark.asyncio
async def test_short_metadata_destination_needs_no_checkout_fetch(monkeypatch):
    fetch = AsyncMock()
    monkeypatch.setattr(destination, "stripe_call", fetch)
    assert (
        await destination.activation_return_to(
            subscription(
                {
                    "pro_activation_return_to": "/chat/original?continue=true#latest",
                }
            )
        )
        == "/chat/original?continue=true#latest"
    )
    fetch.assert_not_awaited()


LONG = f"/chat/original?context={'a' * 700}#latest"
TEMPLATE = (
    "/onboarding?step=preparing&subscription=success&session_id={CHECKOUT_SESSION_ID}"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata, success_path, return_to",
    [
        ({}, LONG, LONG),
        (
            {"pro_activation_return_to": TEMPLATE},
            TEMPLATE,
            TEMPLATE.replace("{CHECKOUT_SESSION_ID}", "cs_1"),
        ),
    ],
    ids=["too-long-to-store", "stored-template"],
)
async def test_owned_checkout_restores_destination_stripe_returned_to(
    monkeypatch, metadata, success_path, return_to
):
    session = SimpleNamespace(
        id="cs_1",
        customer="cus_1",
        subscription="sub_1",
        mode="subscription",
        success_url=f"https://app.test{success_path}",
    )
    monkeypatch.setattr(
        destination,
        "stripe_call",
        AsyncMock(return_value=SimpleNamespace(data=[session], has_more=False)),
    )
    monkeypatch.setattr(
        destination,
        "Settings",
        lambda: SimpleNamespace(
            config=SimpleNamespace(
                frontend_base_url="https://app.test", platform_base_url=None
            )
        ),
    )
    assert await destination.activation_return_to(subscription(metadata)) == return_to


@pytest.mark.asyncio
async def test_a_failed_checkout_lookup_falls_back_instead_of_raising(monkeypatch):
    monkeypatch.setattr(
        destination,
        "stripe_call",
        AsyncMock(side_effect=stripe.APIConnectionError("Stripe unreachable")),
    )
    assert (
        await destination.activation_return_to(
            subscription({"pro_activation_return_to": TEMPLATE})
        )
        == "/settings/billing"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.test/chat",
        "//app.test/chat",
        "https://app.test@attacker.test/chat",
        "https://app.test/\\evil",
    ],
)
def test_checkout_return_rejects_untrusted_origin(monkeypatch, url):
    monkeypatch.setattr(
        destination,
        "Settings",
        lambda: SimpleNamespace(
            config=SimpleNamespace(
                frontend_base_url="https://app.test", platform_base_url=None
            )
        ),
    )
    assert destination.checkout_destination(url) == "/settings/billing"
