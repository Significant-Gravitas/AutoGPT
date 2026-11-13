from copy import deepcopy

import pytest

from backend.data import subscription_activation_evidence as evidence


def invoice():
    return {
        "id": "in_paid",
        "status": "paid",
        "amount_remaining": 0,
        "amount_paid": 5000,
        "created": 200,
        "status_transitions": {"paid_at": 200},
        "customer": "cus",
        "subscription": "sub",
        "billing_reason": "subscription_cycle",
        "lines": {
            "data": [
                {
                    "type": "subscription",
                    "subscription": "sub",
                    "amount": 5000,
                    "quantity": 1,
                    "price": {"id": "price"},
                    "period": {"start": 200, "end": 300},
                }
            ],
            "has_more": False,
        },
    }


def qualifies(value, trial_end=200):
    return evidence.qualifying_invoice(
        value,
        customer_id="cus",
        subscription_id="sub",
        price_id="price",
        trial_end=trial_end,
    )


def test_zero_dollar_trial_setup_is_not_paid_activation():
    value = invoice()
    value.update(billing_reason="subscription_create", created=100, amount_paid=0)
    value["lines"]["data"][0].update(amount=0, period={"start": 100, "end": 200})
    assert not qualifies(value)


@pytest.mark.parametrize("amount_paid", [0, 2000, 5000])
def test_settled_credits_and_discounts_are_valid(amount_paid):
    value = invoice()
    value["amount_paid"] = amount_paid
    assert qualifies(value)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "open"},
        {"status": "draft"},
        {"status": "void"},
        {"amount_remaining": 1},
        {"customer": "other"},
        {"subscription": "other"},
        {"status_transitions": {"paid_at": None}},
        {"created": 199},
        {"billing_reason": "manual"},
    ],
)
def test_unsettled_or_unowned_invoice_never_activates(change):
    value = invoice()
    value.update(change)
    assert not qualifies(value)


def test_initial_signup_requires_creation_invoice():
    value = invoice()
    assert not qualifies(value, trial_end=None)
    value["billing_reason"] = "subscription_create"
    assert qualifies(value, trial_end=None)


def test_paying_early_starts_the_paid_period_just_after_trial_end():
    # Ending a trial early puts the invoice and its paid period one second
    # after trial_end.
    value = invoice()
    value.update(billing_reason="subscription_update", created=201)
    value["status_transitions"] = {"paid_at": 201}
    value["lines"]["data"][0]["period"] = {"start": 201, "end": 301}
    assert qualifies(value, trial_end=200)


@pytest.mark.parametrize(
    "billing_reason", ["subscription_cycle", "subscription_update"]
)
@pytest.mark.parametrize(
    "gap, accepted",
    [
        (-1, False),
        (0, True),
        (1, True),
        (60, True),
        (61, False),
        (30 * 86400, False),
    ],
)
def test_only_a_period_starting_just_after_trial_end_is_the_first(
    billing_reason, gap, accepted
):
    start = 200 + gap
    value = invoice()
    value.update(billing_reason=billing_reason, created=max(200, start))
    value["status_transitions"] = {"paid_at": max(200, start)}
    value["lines"]["data"][0]["period"] = {"start": start, "end": start + 100}
    assert qualifies(value) is accepted


def test_paying_late_keeps_the_first_period_while_trial_end_stands():
    value = invoice()
    value.update(billing_reason="subscription_update", created=201)
    value["status_transitions"] = {"paid_at": 201 + 3600}
    value["lines"]["data"][0]["period"] = {"start": 201, "end": 301}
    assert qualifies(value)


def test_later_trial_renewal_cannot_be_initial_activation():
    value = invoice()
    value["lines"]["data"][0]["period"] = {"start": 300, "end": 400}
    assert not qualifies(value)


def test_basil_invoice_and_line_identity():
    value = invoice()
    del value["subscription"]
    value["parent"] = {"subscription_details": {"subscription": "sub"}}
    line = value["lines"]["data"][0]
    del line["price"]
    line["pricing"] = {"price_details": {"price": "price"}}
    assert qualifies(value)


@pytest.mark.parametrize(
    "change",
    [
        {"quantity": 2},
        {"price": {"id": "other"}},
        {"proration": True},
        {"period": {"start": 199, "end": 300}},
    ],
)
def test_foreign_plan_or_period_is_not_proof(change):
    value = deepcopy(invoice())
    value["lines"]["data"][0].update(change)
    assert not qualifies(value)


def test_trial_with_one_time_fee_is_not_paid_recurring_service():
    value = invoice()
    value["lines"]["data"][0]["amount"] = 0
    value["lines"]["data"].append(
        {"amount": 1000, "type": "invoiceitem", "period": {"start": 200, "end": 300}}
    )
    assert not evidence.settled_recurring_invoice(value)
    assert not qualifies(value)


def test_discounted_net_zero_recurring_line_counts_as_settled():
    value = invoice()
    value["amount_paid"] = 0
    value["lines"]["data"][0].update(amount=0, discount_amounts=[{"amount": 5000}])
    assert qualifies(value)


def test_modern_subscription_line_binding():
    value = invoice()
    line = value["lines"]["data"][0]
    del line["type"]
    del line["subscription"]
    line["parent"] = {
        "type": "subscription_item_details",
        "subscription_item_details": {"subscription": "sub", "proration": False},
    }
    assert qualifies(value)
    line["parent"]["subscription_item_details"]["subscription"] = "other"
    assert not qualifies(value)
