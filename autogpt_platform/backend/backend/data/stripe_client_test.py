"""Tests for the single-door Stripe helper."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import stripe
from prometheus_client import REGISTRY

from backend.data.stripe_client import (
    stripe_call,
    stripe_call_timeout,
    stripe_list_items,
)


def _n(resource: str, method: str, outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "autogpt_stripe_requests_total",
            {"resource": resource, "method": method, "outcome": outcome},
        )
        or 0.0
    )


class _Fake:
    """Stands in for a stripe resource class; __qualname__ drives the labels."""

    @staticmethod
    async def list_async(*, customer: str):
        return {"customer": customer}

    @staticmethod
    async def create_async(**_):
        raise stripe.RateLimitError("slow down")

    @staticmethod
    async def modify_async(*_a, **_k):
        raise stripe.InvalidRequestError("bad", "param")

    @staticmethod
    async def retrieve_async(*_a, **_k):
        await asyncio.sleep(10)


for _f in (
    _Fake.list_async,
    _Fake.create_async,
    _Fake.modify_async,
    _Fake.retrieve_async,
):
    _f.__qualname__ = f"Subscription.{_f.__name__}"


@pytest.mark.asyncio
async def test_ok_returns_result_and_counts():
    before = _n("Subscription", "list", "ok")
    assert await stripe_call(_Fake.list_async, customer="cus_1") == {
        "customer": "cus_1"
    }
    assert _n("Subscription", "list", "ok") == before + 1


@pytest.mark.asyncio
async def test_rate_limit_is_its_own_outcome_and_reraised_unchanged():
    before = _n("Subscription", "create", "rate_limited")
    with pytest.raises(stripe.RateLimitError):
        await stripe_call(_Fake.create_async, customer="cus_1")
    assert _n("Subscription", "create", "rate_limited") == before + 1


@pytest.mark.asyncio
async def test_other_stripe_errors_keep_their_type():
    """Callers branch on stripe.InvalidRequestError; the wrapper must not mask it."""
    before = _n("Subscription", "modify", "api_error")
    with pytest.raises(stripe.InvalidRequestError):
        await stripe_call(_Fake.modify_async, "sub_1", cancel_at_period_end=True)
    assert _n("Subscription", "modify", "api_error") == before + 1


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_counted():
    before = _n("Subscription", "retrieve", "timeout")
    with pytest.raises(stripe.APIConnectionError) as info:
        await stripe_call_timeout(0.05, _Fake.retrieve_async, "sub_1")
    assert isinstance(info.value, stripe.StripeError)
    assert "exceeded 0.05s" in str(info.value)
    assert info.value.should_retry is False
    assert _n("Subscription", "retrieve", "timeout") == before + 1


@pytest.mark.asyncio
async def test_pagination_preserves_all_items():
    first = stripe.ListObject.construct_from(
        {"data": [{"id": "first"}], "has_more": True}, "test-key"
    )
    second = stripe.ListObject.construct_from(
        {"data": [{"id": "second"}], "has_more": False}, "test-key"
    )
    with patch.object(
        stripe.ListObject, "next_page_async", AsyncMock(return_value=second)
    ) as next_page:
        assert [item.id async for item in stripe_list_items(first)] == [
            "first",
            "second",
        ]
    next_page.assert_awaited_once()


@pytest.mark.asyncio
async def test_incomplete_empty_page_is_not_treated_as_exhaustive_history():
    page = stripe.ListObject.construct_from({"data": [], "has_more": True}, "test-key")
    with pytest.raises(ValueError, match="empty page"):
        _ = [item async for item in stripe_list_items(page)]
