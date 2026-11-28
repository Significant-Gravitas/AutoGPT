"""Which paid invoices the read-only audit lists as paid late."""

from scripts.list_billing_restart_accounts import _paid_late

START = 1772539200
HOUR = 3600


def _invoice(paid_at: int, attempt_count: int = 1) -> dict:
    return {
        "id": "in_1",
        "billing_reason": "subscription_cycle",
        "attempt_count": attempt_count,
        "status_transitions": {"paid_at": paid_at},
        "lines": {
            "data": [
                {"type": "subscription", "proration": False, "period": {"start": START}}
            ]
        },
    }


def test_a_manual_payment_soon_after_a_recorded_pause_is_listed():
    """Stripe counts only its own automatic attempts, so a payment made by
    hand an hour after the failed one leaves attempt_count at 1."""
    invoice = _invoice(paid_at=START + 2 * HOUR)

    assert _paid_late(invoice, paused=set()) is None
    assert _paid_late(invoice, paused={"in_1"}) == (START, START + 2 * HOUR)


def test_without_a_record_stripe_alone_decides():
    assert _paid_late(_invoice(paid_at=START + 4 * HOUR), set()) == (
        START,
        START + 4 * HOUR,
    )
    assert _paid_late(_invoice(paid_at=START + 2 * HOUR, attempt_count=2), set())
    assert _paid_late(_invoice(paid_at=START + HOUR), set()) is None
