"""delete_account: the order of its steps, and what a failure at each one leaves.

Every step is mocked where the service uses it; routes_test.py runs the same
flow against the real database.
"""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import stripe
from prisma.enums import AccountDeletionSource, SubscriptionTier
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.service import (
    EmailMismatchError,
    PlanCancellationError,
    delete_account,
)
from backend.data.model import AutoTopUpConfig
from backend.util.posthog_events import PostHogEvent

MODULE = "backend.api.features.account_deletion.service"
USER_ID = "user-1"
EMAIL = "Ada.Lovelace@Example.com"
TYPED = "  ada.lovelace@example.COM "
ERASE_AFTER = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)

LOAD = ["get_user_by_id", "get_deletion_request"]
BEFORE_CLOSE = [
    "cancel_all_subscriptions_now",
    "set_auto_top_up",
    "revoke_connected_apps",
    "revoke_issued_access",
    "stop_user_work",
    "stop_account_emails",
]
CLOSE = [
    "close_account",
    "remember_account_closed",
    "forget_refusal",
    "forget_creator_page",
]
AFTER_CLOSE = [
    "send_account_deleted_email",
    "mark_confirmation_sent",
    "capture_anonymous",
]
STEPS = LOAD + BEFORE_CLOSE + CLOSE + AFTER_CLOSE
# A confirmation that did not go out is left unmarked for the erasure routine.
UNSENT = [name for name in STEPS if name != "mark_confirmation_sent"]
SYNC_STEPS = {"remember_account_closed", "forget_refusal", "capture_anonymous"}


def _user() -> MagicMock:
    return MagicMock(
        id=USER_ID,
        email=EMAIL,
        subscription_tier=SubscriptionTier.PRO,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def steps(mocker: MockerFixture) -> MagicMock:
    """Every step, attached to one parent so the order of the calls is kept."""
    parent = MagicMock()
    for name in STEPS:
        step = mocker.patch(
            f"{MODULE}.{name}",
            new_callable=MagicMock if name in SYNC_STEPS else AsyncMock,
        )
        parent.attach_mock(step, name)
    parent.get_user_by_id.return_value = _user()
    parent.get_deletion_request.return_value = None
    parent.cancel_all_subscriptions_now.return_value = 1
    parent.close_account.return_value = MagicMock(eraseAfter=ERASE_AFTER)
    return parent


def _called(steps: MagicMock) -> list[str]:
    return [name for name, _, _ in steps.mock_calls]


async def _delete(source: AccountDeletionSource = AccountDeletionSource.SETTINGS):
    return await delete_account(user_id=USER_ID, confirm_email=TYPED, source=source)


@pytest.mark.asyncio
async def test_every_step_runs_once_in_order(steps: MagicMock):
    assert await _delete() == ERASE_AFTER

    assert _called(steps) == STEPS
    steps.get_user_by_id.assert_awaited_once_with(USER_ID)
    steps.get_deletion_request.assert_awaited_once_with(USER_ID)
    for name in BEFORE_CLOSE:
        if name != "set_auto_top_up":
            getattr(steps, name).assert_awaited_once_with(USER_ID)
    steps.close_account.assert_awaited_once_with(
        USER_ID, AccountDeletionSource.SETTINGS
    )
    steps.remember_account_closed.assert_called_once_with(USER_ID)
    steps.forget_refusal.assert_called_once_with(USER_ID)
    steps.forget_creator_page.assert_awaited_once_with(USER_ID)
    steps.send_account_deleted_email.assert_awaited_once_with(
        EMAIL, AccountDeletionSource.SETTINGS, ERASE_AFTER
    )
    steps.mark_confirmation_sent.assert_awaited_once_with(USER_ID)
    steps.capture_anonymous.assert_called_once_with(
        PostHogEvent.ACCOUNT_DELETION_COMPLETED,
        {"plan_state": "pro", "surface": "settings"},
    )


@pytest.mark.asyncio
async def test_the_paywall_is_recorded_as_the_surface(steps: MagicMock):
    await _delete(AccountDeletionSource.PAYWALL)

    _, properties = steps.capture_anonymous.call_args.args
    assert properties == {"plan_state": "pro", "surface": "paywall"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source", [AccountDeletionSource.SETTINGS, AccountDeletionSource.PAYWALL]
)
async def test_the_confirmation_email_is_told_the_source(
    steps: MagicMock, source: AccountDeletionSource
):
    await _delete(source)

    steps.send_account_deleted_email.assert_awaited_once_with(
        EMAIL, source, ERASE_AFTER
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "typed",
    ["someone.else@example.com", "ada.lovelace@example.co", "", "   "],
)
async def test_a_different_email_changes_nothing(steps: MagicMock, typed: str):
    with pytest.raises(EmailMismatchError):
        await delete_account(
            user_id=USER_ID, confirm_email=typed, source=AccountDeletionSource.SETTINGS
        )

    assert _called(steps) == ["get_user_by_id"]


@pytest.mark.asyncio
async def test_a_blank_email_never_matches(steps: MagicMock):
    """Not even an account whose own email is blank."""
    steps.get_user_by_id.return_value.email = ""

    with pytest.raises(EmailMismatchError):
        await delete_account(
            user_id=USER_ID, confirm_email="  ", source=AccountDeletionSource.SETTINGS
        )

    assert _called(steps) == ["get_user_by_id"]


@pytest.mark.asyncio
async def test_an_account_already_closed_returns_at_once(steps: MagicMock):
    """A repeat call: no Stripe call, no second email, no second count, and
    the erasure date the first call set."""
    steps.get_deletion_request.return_value = MagicMock(eraseAfter=ERASE_AFTER)

    assert await _delete() == ERASE_AFTER

    assert _called(steps) == LOAD


@pytest.mark.asyncio
async def test_the_plan_is_read_before_stripe_is_touched(steps: MagicMock):
    """The cancel's webhooks move the tier to NO_TIER; the count keeps the
    plan the account had."""
    user = steps.get_user_by_id.return_value

    async def cancel_moves_the_tier(user_id: str) -> int:
        user.subscription_tier = SubscriptionTier.NO_TIER
        return 1

    steps.cancel_all_subscriptions_now.side_effect = cancel_moves_the_tier

    await _delete()

    _, properties = steps.capture_anonymous.call_args.args
    assert properties["plan_state"] == "pro"


@pytest.mark.asyncio
async def test_a_stripe_error_stops_before_anything_changes(steps: MagicMock):
    refused = stripe.APIConnectionError("Stripe is unreachable")
    steps.cancel_all_subscriptions_now.side_effect = refused

    with pytest.raises(PlanCancellationError) as raised:
        await _delete()

    assert raised.value.__cause__ is refused
    assert _called(steps) == LOAD + ["cancel_all_subscriptions_now"]


@pytest.mark.asyncio
async def test_any_other_cancel_error_is_not_reported_as_a_plan_error(
    steps: MagicMock,
):
    steps.cancel_all_subscriptions_now.side_effect = RuntimeError("database down")

    with pytest.raises(RuntimeError, match="database down"):
        await _delete()

    assert "close_account" not in _called(steps)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", BEFORE_CLOSE[1:])
async def test_a_failure_before_the_close_leaves_the_account_open(
    steps: MagicMock, failing: str
):
    getattr(steps, failing).side_effect = RuntimeError(f"{failing} failed")

    with pytest.raises(RuntimeError, match=f"{failing} failed"):
        await _delete()

    called = _called(steps)
    assert called == STEPS[: STEPS.index(failing) + 1]
    assert not set(CLOSE + AFTER_CLOSE) & set(called)


@pytest.mark.asyncio
async def test_a_retry_after_a_failure_before_the_close_runs_every_step(
    steps: MagicMock,
):
    steps.revoke_issued_access.side_effect = [RuntimeError("database down"), None]
    with pytest.raises(RuntimeError):
        await _delete()
    steps.reset_mock()

    await _delete()

    assert _called(steps) == STEPS


@pytest.mark.asyncio
async def test_a_failed_confirmation_email_still_succeeds(
    steps: MagicMock, caplog: pytest.LogCaptureFixture
):
    steps.send_account_deleted_email.side_effect = RuntimeError(
        f"Postmark refused {EMAIL}"
    )

    with caplog.at_level(logging.DEBUG):
        await _delete()

    assert _called(steps) == UNSENT
    assert USER_ID in caplog.text
    assert EMAIL.lower() not in caplog.text.lower()


@pytest.mark.asyncio
async def test_a_failure_to_mark_the_confirmation_sent_still_succeeds(
    steps: MagicMock, caplog: pytest.LogCaptureFixture
):
    """The email went out; at worst the erasure routine sends a second one."""
    steps.mark_confirmation_sent.side_effect = RuntimeError("db down")

    with caplog.at_level(logging.ERROR):
        await _delete()

    assert _called(steps) == STEPS
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_losing_the_close_race_sends_nothing(steps: MagicMock):
    """A concurrent call closed the account first; it sends the email and
    records the count, so this one must not do either again."""
    steps.close_account.return_value = None
    steps.get_deletion_request.side_effect = [None, MagicMock(eraseAfter=ERASE_AFTER)]

    assert await _delete() == ERASE_AFTER

    assert _called(steps) == LOAD + BEFORE_CLOSE + CLOSE + ["get_deletion_request"]


@pytest.mark.asyncio
async def test_neither_email_is_ever_logged(
    steps: MagicMock, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.DEBUG):
        await _delete()
        with pytest.raises(EmailMismatchError):
            await delete_account(
                user_id=USER_ID,
                confirm_email="typo@example.com",
                source=AccountDeletionSource.SETTINGS,
            )
        steps.cancel_all_subscriptions_now.side_effect = stripe.CardError(
            "declined", param=None, code="card_declined"
        )
        with pytest.raises(PlanCancellationError):
            await _delete()

    logged = caplog.text.lower()
    assert "typo@example.com" not in logged
    assert EMAIL.lower() not in logged


@pytest.mark.asyncio
async def test_auto_top_up_is_switched_off_before_the_close(steps: MagicMock):
    """Work that outlives the deletion must not charge the saved card; the
    order of the steps is checked with the others above."""
    await _delete()

    steps.set_auto_top_up.assert_awaited_once_with(
        USER_ID, AutoTopUpConfig(threshold=0, amount=0)
    )


@pytest.mark.asyncio
async def test_a_stalled_confirmation_email_does_not_hold_the_request(
    steps: MagicMock, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    mocker.patch(f"{MODULE}.CONFIRMATION_EMAIL_TIMEOUT_SECONDS", 0.05)

    async def stall(
        email: str, source: AccountDeletionSource, erase_after: datetime
    ) -> None:
        await asyncio.Event().wait()

    steps.send_account_deleted_email.side_effect = stall

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(_delete(), timeout=5)

    assert _called(steps) == UNSENT
    assert "TimeoutError" in caplog.text
