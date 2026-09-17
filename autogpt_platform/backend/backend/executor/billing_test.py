"""The low-balance alert's runway forecast."""

from types import SimpleNamespace

from backend.executor.billing import _low_balance_cause


def test_a_low_balance_alert_never_forecasts_from_a_single_charge():
    """The rate must be a rate.

    Using the threshold-crossing transaction's own cost meant a 1-credit charge
    against a 5-credit balance forecast 500 days of runway — inside an email
    whose subject says the credits are about to run out. Both the date and the
    "per day" figure reached the customer.
    """
    client = SimpleNamespace(
        get_recent_daily_spend=lambda _uid: 250.0,  # 2.50 credits/day
        count_scheduled_agents=lambda _uid: 2,
    )
    cause = _low_balance_cause(
        current_balance=1000, transaction_cost=1, db_client=client, user_id="u1"
    )
    # 1000 cents at 250 cents/day is 4 days, not 1000.
    assert cause.days_left == 4
    assert "2.50 credits" in (cause.daily_rate_display or "")


def test_no_spend_history_means_no_invented_date():
    """A wrong run-out date in this email is worse than no date."""
    client = SimpleNamespace(
        get_recent_daily_spend=lambda _uid: 0.0,
        count_scheduled_agents=lambda _uid: 2,
    )
    cause = _low_balance_cause(
        current_balance=1000, transaction_cost=1, db_client=client, user_id="u1"
    )
    assert cause.days_left is None
    assert cause.runs_out_label is None
    assert cause.headline == "Your credits are running low"
    assert "2 scheduled agents would stop" in cause.body
    assert cause.tag == "low balance"
