"""Tests for StripeSubscriptionTriggerBlock.run() payload parsing.

The bundled ``test_input``/``test_output`` (exercised by the standard block
runner) only covers ``customer.subscription.created``. These cover the other
two subscribable events, the older top-level ``plan`` payload shape, the
nickname fallback, and the malformed-payload path.
"""

import copy

import pytest

from backend.blocks.stripe._auth import TEST_CREDENTIALS_INPUT
from backend.blocks.stripe.triggers import (
    StripeSubscriptionTriggerBlock,
    load_example_payload,
)


async def run_block(payload: dict) -> dict:
    """Run the block over ``payload`` and collect its outputs by name."""
    block = StripeSubscriptionTriggerBlock()
    input_data = StripeSubscriptionTriggerBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "payload": payload}
    )
    return {name: value async for name, value in block.run(input_data)}


@pytest.mark.asyncio
async def test_updated_event_reports_the_new_plan():
    """An upgrade must surface the price the subscription moved *to*."""
    outputs = await run_block(load_example_payload("customer.subscription.updated"))

    assert outputs["event_type"] == "customer.subscription.updated"
    assert outputs["subscription_id"] == "sub_1OxK2fLkdIwHu7ixABCDEFGH"
    assert outputs["customer_id"] == "cus_Pq1234ABCDEF"
    assert outputs["status"] == "active"
    assert outputs["plan_name"] == "Pro Yearly"
    assert outputs["plan_interval"] == "year"
    assert outputs["amount_cents"] == 19200
    assert outputs["currency"] == "usd"
    assert outputs["livemode"] is False


@pytest.mark.asyncio
async def test_deleted_event_surfaces_cancellation_context():
    """A cancellation must be distinguishable from a scheduled end."""
    outputs = await run_block(load_example_payload("customer.subscription.deleted"))

    assert outputs["event_type"] == "customer.subscription.deleted"
    assert outputs["status"] == "canceled"
    assert outputs["canceled_at"] == 1714678400
    assert outputs["cancel_at_period_end"] is False


@pytest.mark.asyncio
async def test_scheduled_cancellation_is_not_reported_as_canceled():
    """``cancel_at_period_end`` is the only signal for a not-yet-ended cancel."""
    payload = copy.deepcopy(load_example_payload("customer.subscription.updated"))
    payload["data"]["object"]["cancel_at_period_end"] = True

    outputs = await run_block(payload)

    assert outputs["cancel_at_period_end"] is True
    assert outputs["canceled_at"] == 0
    assert outputs["status"] == "active"


@pytest.mark.asyncio
async def test_falls_back_to_top_level_plan_on_older_api_versions():
    """Pre-2019 API versions send no ``items``, only a flat ``plan``."""
    payload = copy.deepcopy(load_example_payload("customer.subscription.created"))
    del payload["data"]["object"]["items"]

    outputs = await run_block(payload)

    assert outputs["plan_name"] == "Pro Monthly"
    assert outputs["plan_interval"] == "month"
    assert outputs["amount_cents"] == 2000


@pytest.mark.asyncio
async def test_plan_name_falls_back_to_price_id_without_a_nickname():
    """Most prices have no nickname; the raw ID is better than an empty string."""
    payload = copy.deepcopy(load_example_payload("customer.subscription.created"))
    payload["data"]["object"]["items"]["data"][0]["price"]["nickname"] = None

    outputs = await run_block(payload)

    assert outputs["plan_name"] == "price_1OxABCLkdIwHu7ixMonthlyPro"


@pytest.mark.asyncio
async def test_missing_unit_amount_falls_back_to_zero():
    """Metered prices carry no `unit_amount`; the output is typed `int`."""
    payload = copy.deepcopy(load_example_payload("customer.subscription.created"))
    payload["data"]["object"]["items"]["data"][0]["price"]["unit_amount"] = None

    outputs = await run_block(payload)

    assert outputs["amount_cents"] == 0


@pytest.mark.parametrize(
    "events, expected",
    [
        (
            {"created": True, "updated": True, "deleted": False},
            ["customer.subscription.created", "customer.subscription.updated"],
        ),
        (
            {"created": False, "updated": False, "deleted": True},
            ["customer.subscription.deleted"],
        ),
        (
            {"created": True, "updated": True, "deleted": True},
            [
                "customer.subscription.created",
                "customer.subscription.updated",
                "customer.subscription.deleted",
            ],
        ),
    ],
)
def test_event_filter_maps_to_stripe_event_names(events, expected):
    """The toggles must expand to the exact names registered with Stripe.

    A typo in `event_format`, or a toggle that stops reaching the filter, would
    silently register the wrong events — the webhook would then be live but
    never deliver.
    """
    config = StripeSubscriptionTriggerBlock().webhook_config
    assert config is not None
    derived = [
        config.event_format.format(event=name)
        for name, enabled in events.items()
        if enabled
    ]
    assert derived == expected


@pytest.mark.parametrize(
    "event_type, enabled",
    [
        ("customer.subscription.created", True),
        ("customer.subscription.deleted", False),
        ("customer.subscription.trial_will_end", False),
    ],
)
def test_block_only_triggers_on_enabled_events(event_type, enabled):
    """The same mapping is used on the way back in, to route a delivery."""
    block = StripeSubscriptionTriggerBlock()
    trigger_config = {"events": {"created": True, "updated": True, "deleted": False}}
    assert block.is_triggered_by_event_type(trigger_config, event_type) is enabled


@pytest.mark.asyncio
async def test_updated_event_exposes_previous_attributes():
    """`updated` fires on any mutation; this is what separates an upgrade."""
    payload = copy.deepcopy(load_example_payload("customer.subscription.updated"))
    payload["data"]["previous_attributes"] = {"items": {"data": [{"price": "old"}]}}

    outputs = await run_block(payload)

    assert outputs["previous_attributes"] == {"items": {"data": [{"price": "old"}]}}


@pytest.mark.asyncio
async def test_previous_attributes_defaults_to_empty_mapping():
    """Absent on created/deleted — must be {} rather than None for a dict output."""
    outputs = await run_block(load_example_payload("customer.subscription.created"))

    assert outputs["previous_attributes"] == {}


@pytest.mark.asyncio
async def test_malformed_payload_emits_no_partial_output():
    """Yielding "error" aborts the block, so nothing may be emitted before it."""
    outputs = await run_block({"type": "customer.subscription.created"})

    assert list(outputs) == ["error"]
    assert "Failed to parse Stripe subscription payload" in outputs["error"]
