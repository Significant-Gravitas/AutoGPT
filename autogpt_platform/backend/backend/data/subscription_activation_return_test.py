from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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


@pytest.mark.asyncio
async def test_owned_checkout_restores_long_destination(monkeypatch):
    return_to = f"/chat/original?context={'a' * 700}#latest"
    session = SimpleNamespace(
        id="cs_1",
        customer="cus_1",
        subscription="sub_1",
        mode="subscription",
        success_url=f"https://app.test{return_to}",
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
    assert await destination.activation_return_to(subscription()) == return_to


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
