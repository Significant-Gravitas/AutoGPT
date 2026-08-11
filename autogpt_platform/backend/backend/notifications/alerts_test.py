"""The rules that keep the Alert channel worth reading.

Debounce and coalesce, never twice in 24 hours, two per day, and cancel
anything that solved itself — with everything that can't be sent folded into
the Briefing rather than dropped.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import AlertCause, AlertConditionStatus

from backend.data.alerts import ALERT_DEDUPE_WINDOW, AlertConditionDTO
from backend.notifications.alert_causes import (
    AuthExpiredCause,
    LowBalanceCause,
    ZeroBalanceCause,
)
from backend.notifications.alerts import (
    ALERT_DEBOUNCE,
    MAX_ALERT_EMAILS_PER_DAY,
    build_alert_email,
    raise_alert,
    resolve_alert,
)

USER = "user-1"


def _condition(cause: AlertCause, payload) -> AlertConditionDTO:
    return AlertConditionDTO(
        id=f"c-{cause.value}",
        user_id=USER,
        cause=cause,
        cause_key=f"{cause.value.lower()}:x",
        data=payload.model_dump(mode="json"),
        status=AlertConditionStatus.PENDING,
        created_at=datetime.now(tz=timezone.utc) - ALERT_DEBOUNCE,
        sent_at=None,
        briefed_at=None,
    )


AUTH = AuthExpiredCause(
    cta_path="/integrations/gmail/reconnect",
    agent="Invoice Chaser",
    provider="Gmail",
    expired_at_label="9:14",
    runs_skipped=2,
    next_try_label="16:00",
)
ZERO = ZeroBalanceCause(
    cta_path="/settings/billing", agent="Price Watch", shortfall_display="4.00"
)
LOW = LowBalanceCause(
    cta_path="/settings/billing",
    days_left=3,
    daily_rate_display="2.00 credits",
    balance_display="6.00 credits",
    runs_out_label="9 Aug",
    scheduled_agents=4,
)


@pytest.mark.asyncio
async def test_pending_alerts_coalesce_into_one_email():
    pending = [_condition(AlertCause.LOW_BALANCE, LOW), _condition(AlertCause.ZERO_BALANCE, ZERO)]
    with patch(
        "backend.notifications.alerts.alerts_db.get_pending_conditions",
        AsyncMock(return_value=pending),
    ), patch(
        "backend.notifications.alerts.alerts_db.count_alerts_sent_since",
        AsyncMock(return_value=0),
    ):
        built = await build_alert_email(USER, alerts_enabled=True)

    assert built is not None
    # Severity decides which condition leads: being out of credits outranks a
    # forecast of running low.
    assert built.data.primary.headline == ZERO.headline
    assert [item.agent for item in built.data.also] == ["Your balance"]
    assert built.data.also_label == "Also waiting"
    assert sorted(built.condition_ids) == sorted(c.id for c in pending)


@pytest.mark.asyncio
async def test_the_cta_verb_is_the_fix_not_view_dashboard():
    with patch(
        "backend.notifications.alerts.alerts_db.get_pending_conditions",
        AsyncMock(return_value=[_condition(AlertCause.AUTH_EXPIRED, AUTH)]),
    ), patch(
        "backend.notifications.alerts.alerts_db.count_alerts_sent_since",
        AsyncMock(return_value=0),
    ):
        built = await build_alert_email(USER, alerts_enabled=True)

    assert built is not None
    assert built.data.primary.cta_label == "Reconnect Gmail"
    assert (
        built.data.primary.subject
        == "Invoice Chaser is stuck — Gmail needs a reconnect"
    )


@pytest.mark.asyncio
async def test_overflow_past_the_daily_cap_folds_into_the_briefing():
    pending = [_condition(AlertCause.AUTH_EXPIRED, AUTH)]
    with patch(
        "backend.notifications.alerts.alerts_db.get_pending_conditions",
        AsyncMock(return_value=pending),
    ), patch(
        "backend.notifications.alerts.alerts_db.count_alerts_sent_since",
        AsyncMock(return_value=MAX_ALERT_EMAILS_PER_DAY),
    ), patch(
        "backend.notifications.alerts.alerts_db.mark_deferred", AsyncMock()
    ) as mark_deferred, patch(
        "backend.notifications.alerts.alerts_db.mark_sent", AsyncMock()
    ) as mark_sent:
        built = await build_alert_email(USER, alerts_enabled=True)

    assert built is None
    mark_deferred.assert_awaited_once_with([pending[0].id])
    # Nothing is marked sent, so the 24h dedupe window never opens on an email
    # that was never sent.
    mark_sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_alerts_switched_off_still_reach_the_briefing():
    pending = [_condition(AlertCause.AUTH_EXPIRED, AUTH)]
    with patch(
        "backend.notifications.alerts.alerts_db.get_pending_conditions",
        AsyncMock(return_value=pending),
    ), patch(
        "backend.notifications.alerts.alerts_db.mark_deferred", AsyncMock()
    ) as mark_deferred:
        built = await build_alert_email(USER, alerts_enabled=False)

    assert built is None
    mark_deferred.assert_awaited_once_with([pending[0].id])


@pytest.mark.asyncio
async def test_nothing_pending_sends_nothing():
    with patch(
        "backend.notifications.alerts.alerts_db.get_pending_conditions",
        AsyncMock(return_value=[]),
    ):
        assert await build_alert_email(USER, alerts_enabled=True) is None


@pytest.mark.asyncio
async def test_raising_a_condition_carries_its_cause_and_payload():
    with patch(
        "backend.notifications.alerts.alerts_db.raise_condition", AsyncMock()
    ) as raise_condition:
        await raise_alert(USER, "auth_expired:gmail:g1", AUTH)

    kwargs = raise_condition.await_args.kwargs
    assert kwargs["cause"] is AlertCause.AUTH_EXPIRED
    assert kwargs["cause_key"] == "auth_expired:gmail:g1"
    assert kwargs["data"]["provider"] == "Gmail"


@pytest.mark.asyncio
async def test_a_condition_that_clears_cancels_the_send():
    with patch(
        "backend.notifications.alerts.alerts_db.resolve_condition",
        AsyncMock(return_value=True),
    ) as resolve_condition:
        await resolve_alert(USER, "auth_expired:gmail:g1")

    resolve_condition.assert_awaited_once_with(USER, "auth_expired:gmail:g1")


def test_the_debounce_and_dedupe_windows_match_the_design():
    assert ALERT_DEBOUNCE == timedelta(minutes=10)
    assert ALERT_DEDUPE_WINDOW == timedelta(hours=24)
    assert MAX_ALERT_EMAILS_PER_DAY == 2
