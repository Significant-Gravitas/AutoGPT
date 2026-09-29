"""The MailerLite trial group holds exactly the customers currently on a trial:
join on start and on un-cancel, leave on cancel, conversion, failed conversion
and end. A conversion also joins the paying audience, as a first checkout
would."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.notifications import (
    AudienceAction,
    AudienceEventModel,
    NotificationResult,
    SubscriberField,
    SubscriptionStatus,
)
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer
from backend.notifications import mailerlite, subscriber_fields
from backend.notifications import trial as notices
from backend.notifications import trial_audience
from backend.notifications.notifications import NotificationManager

EMAIL = "sam@example.com"


@pytest.fixture
def trial() -> TrialState:
    now = datetime.now(UTC)
    return TrialState(
        id="trial-1",
        user_id="user-1",
        customer_id="cus_1",
        offer=AcceptedTrialOffer(
            version="offer-v1",
            new_users_from=now - timedelta(days=1),
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=250_000,
            weekly_cost_limit=1_000_000,
            total_cost_limit=1_000_000,
            onboarding_credit_amount=300,
            price_id="price_pro",
            unit_amount=2000,
            currency="usd",
        ),
        checkout_session_id="cs_1",
        subscription_id="sub_1",
        checkout_attempt=0,
        success_url="https://example.com",
        cancel_url="https://example.com",
        checkout_metadata={},
        status="trialing",
        card_verified_at=now,
        started_at=now,
        ends_at=now + timedelta(days=7),
        consumed_at=now,
        converted_at=None,
        cancel_at_period_end=False,
        cost_microdollars=0,
    )


@pytest.fixture
def trial_group(monkeypatch):
    fake = SimpleNamespace(
        config=SimpleNamespace(mailerlite_trial_group_id="grp_trial")
    )
    monkeypatch.setattr(trial_audience, "settings", fake)
    return fake


def _state(trial: TrialState, kind: str) -> tuple[TrialState, dict]:
    """A trial and the live Stripe subscription at the moment `kind` applies."""
    trial_end = int(trial.ends_at.timestamp())
    raw = {
        "id": trial.subscription_id,
        "customer": trial.customer_id,
        "status": "trialing",
        "trial_end": trial_end,
        "cancel_at_period_end": False,
        "metadata": {
            "trial_enrollment_id": trial.id,
            "user_id": trial.user_id,
            "trial_checkout_attempt": str(trial.checkout_attempt),
        },
    }
    if kind == "canceled":
        raw["cancel_at_period_end"] = True
    elif kind == "converted":
        raw["status"] = "active"
        trial = trial.model_copy(
            update={
                "converted_at": datetime.now(UTC),
                "conversion_invoice_id": "in_1",
            }
        )
    elif kind == "ended":
        raw["status"] = "canceled"
    elif kind == "payment_failed":
        raw["status"] = "past_due"
    return trial, raw


async def _notify(
    trial,
    raw,
    kind,
    *,
    welcomed=False,
    claimed=True,
    audience=None,
    raises=False,
    email=EMAIL,
    claim_welcome=None,
):
    audience = audience or AsyncMock(return_value=NotificationResult(success=True))
    notice = AsyncMock(return_value=NotificationResult(success=True))
    user = SimpleNamespace(name="Sam", email=email)
    users = MagicMock(
        get_user_by_id=AsyncMock(return_value=user),
        claim_welcome_email=claim_welcome or AsyncMock(return_value=not welcomed),
    )
    with (
        patch.object(notices, "stripe_call", AsyncMock(return_value=raw)),
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(
                get_subscription_trial=AsyncMock(return_value=trial)
            ),
        ),
        patch.object(notices, "user_db", return_value=users),
        patch.object(trial_audience, "user_db", return_value=users),
        patch.object(notices, "claim_once", AsyncMock(return_value=claimed)),
        patch.object(notices, "release_claim", AsyncMock()) as release,
        patch.object(notices, "queue_notification_async", notice),
        patch.object(trial_audience, "queue_audience_change", audience),
        patch.object(notices, "_track_billing_event"),
    ):
        if raises:
            with pytest.raises(RuntimeError):
                await notices.notify_trial(raw, kind)
        else:
            await notices.notify_trial(raw, kind)
    actions = [c.args[0].action for c in audience.await_args_list]
    return actions, notice, release


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, actions",
    [
        ("started", [AudienceAction.ADD_TRIAL]),
        ("canceled", [AudienceAction.REMOVE_TRIAL]),
        ("resumed", [AudienceAction.ADD_TRIAL]),
        ("ended", [AudienceAction.REMOVE_TRIAL]),
        ("payment_failed", [AudienceAction.REMOVE_TRIAL]),
        ("ending", []),
    ],
)
async def test_each_trial_transition_moves_the_trial_group(
    trial, trial_group, kind, actions
):
    trial, raw = _state(trial, kind)
    if kind == "ending":
        raw["trial_end"] = int((datetime.now(UTC) + timedelta(days=1)).timestamp())
    got, notice, _ = await _notify(trial, raw, kind)
    assert got == actions
    notice.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "welcomed, paying",
    [(False, AudienceAction.ENROLL_TOUR), (True, AudienceAction.ADD_CHANGELOG)],
)
async def test_conversion_leaves_the_trial_and_joins_the_paying_audience(
    trial, trial_group, welcomed, paying
):
    trial, raw = _state(trial, "converted")
    got, notice, _ = await _notify(trial, raw, "converted", welcomed=welcomed)
    assert got == [AudienceAction.REMOVE_TRIAL, paying]
    notice.assert_awaited_once()


@pytest.mark.asyncio
async def test_conversion_joins_the_paying_audience_without_a_trial_group(
    trial, monkeypatch
):
    monkeypatch.setattr(
        trial_audience,
        "settings",
        SimpleNamespace(config=SimpleNamespace(mailerlite_trial_group_id="")),
    )
    trial, raw = _state(trial, "converted")
    got, _, _ = await _notify(trial, raw, "converted")
    assert got == [AudienceAction.ENROLL_TOUR]


@pytest.mark.asyncio
async def test_no_trial_group_means_no_trial_group_changes(trial, monkeypatch):
    monkeypatch.setattr(
        trial_audience,
        "settings",
        SimpleNamespace(config=SimpleNamespace(mailerlite_trial_group_id="")),
    )
    trial, raw = _state(trial, "started")
    got, notice, _ = await _notify(trial, raw, "started")
    assert got == []
    notice.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_replayed_event_changes_nothing(trial, trial_group):
    trial, raw = _state(trial, "canceled")
    got, notice, _ = await _notify(trial, raw, "canceled", claimed=False)
    assert got == []
    notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_group_change_releases_the_claim_so_stripe_retries(
    trial, trial_group
):
    trial, raw = _state(trial, "canceled")
    failing = AsyncMock(return_value=NotificationResult(success=False, message="down"))
    _, notice, release = await _notify(
        trial, raw, "canceled", audience=failing, raises=True
    )
    release.assert_awaited_once_with(notices.trial_notice_key(trial, "canceled"))
    notice.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["started", "converted"])
async def test_an_email_mailerlite_cannot_take_never_blocks_the_notice(
    trial, trial_group, kind
):
    """A reserved-domain address fails the audience event's email validation.
    That must cost the MailerLite change, not the customer's trial notice."""
    trial, raw = _state(trial, kind)
    got, notice, release = await _notify(trial, raw, kind, email="sam@site.test")
    assert got == []
    notice.assert_awaited_once()
    release.assert_not_awaited()


@pytest.mark.asyncio
async def test_joining_the_paying_audience_never_fails_the_sent_conversion(
    trial, trial_group
):
    trial, raw = _state(trial, "converted")
    broken = AsyncMock(side_effect=RuntimeError("database unavailable"))
    got, notice, release = await _notify(trial, raw, "converted", claim_welcome=broken)
    assert got == [AudienceAction.REMOVE_TRIAL]
    notice.assert_awaited_once()
    release.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action, handler",
    [
        (AudienceAction.ADD_TRIAL, "add_to_trial"),
        (AudienceAction.REMOVE_TRIAL, "remove_from_trial"),
    ],
)
async def test_the_consumer_routes_trial_changes_to_mailerlite(action, handler):
    event = AudienceEventModel(action=action, email=EMAIL, user_id="user-1")
    with patch.object(mailerlite, handler, AsyncMock()) as called:
        assert await NotificationManager._process_audience_change(
            MagicMock(), event.model_dump_json()
        )
    called.assert_awaited_once_with(EMAIL, None)


@pytest.fixture
def mailerlite_configured(monkeypatch):
    fake = SimpleNamespace(
        config=SimpleNamespace(mailerlite_trial_group_id="grp_trial"),
        secrets=SimpleNamespace(mailerlite_api_token="token"),
    )
    monkeypatch.setattr(mailerlite, "settings", fake)
    return fake


def _response(status: int, body: dict | None = None) -> MagicMock:
    response = MagicMock(status=status)
    response.json.return_value = body or {}
    return response


@pytest.mark.asyncio
async def test_add_to_trial_upserts_into_the_trial_group(mailerlite_configured):
    client = MagicMock(post=AsyncMock(return_value=_response(201)))
    with patch.object(mailerlite, "_client", return_value=client):
        await mailerlite.add_to_trial(EMAIL)
    assert client.post.await_args.kwargs["json"] == {
        "email": EMAIL,
        "groups": ["grp_trial"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [204, 404])
async def test_remove_from_trial_unassigns_and_treats_gone_as_done(
    mailerlite_configured, status
):
    client = MagicMock(
        get=AsyncMock(return_value=_response(200, {"data": {"id": "ml_1"}})),
        delete=AsyncMock(return_value=_response(status)),
    )
    with patch.object(mailerlite, "_client", return_value=client):
        await mailerlite.remove_from_trial(EMAIL)
    assert client.delete.await_args.args[0].endswith(
        "/subscribers/ml_1/groups/grp_trial"
    )


@pytest.mark.asyncio
async def test_remove_from_trial_fails_loudly_so_the_job_retries(mailerlite_configured):
    client = MagicMock(
        get=AsyncMock(return_value=_response(200, {"data": {"id": "ml_1"}})),
        delete=AsyncMock(return_value=_response(500)),
    )
    with (
        patch.object(mailerlite, "_client", return_value=client),
        pytest.raises(mailerlite.MailerLiteError) as raised,
    ):
        await mailerlite.remove_from_trial(EMAIL)
    assert EMAIL not in str(raised.value)


# ── subscriber fields ──────────────────────────────────────────────────────


@pytest.fixture
def fields_on(monkeypatch):
    monkeypatch.setattr(
        subscriber_fields,
        "settings",
        SimpleNamespace(secrets=SimpleNamespace(mailerlite_api_token="token")),
    )


def _day(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%d")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, status",
    [
        ("started", SubscriptionStatus.IN_TRIAL),
        ("resumed", SubscriptionStatus.IN_TRIAL),
        ("canceled", SubscriptionStatus.TRIAL_CANCELED),
        # Neither ever became a paid subscription, so neither is churn.
        ("ended", SubscriptionStatus.TRIAL_CANCELED),
        ("payment_failed", SubscriptionStatus.TRIAL_CANCELED),
        ("converted", SubscriptionStatus.SUBSCRIBED),
    ],
)
async def test_each_trial_transition_sets_the_status_on_its_group_change(
    trial, trial_group, fields_on, kind, status
):
    trial, raw = _state(trial, kind)
    audience = AsyncMock(return_value=NotificationResult(success=True))
    await _notify(trial, raw, kind, audience=audience)
    event = audience.await_args_list[0].args[0]
    assert event.action in (AudienceAction.ADD_TRIAL, AudienceAction.REMOVE_TRIAL)
    assert event.fields[SubscriberField.STATUS] == status.value


@pytest.mark.asyncio
async def test_a_trial_start_carries_stripes_trial_start_date(
    trial, trial_group, fields_on
):
    trial, raw = _state(trial, "started")
    raw["trial_start"] = 1788000000
    audience = AsyncMock(return_value=NotificationResult(success=True))
    await _notify(trial, raw, "started", audience=audience)
    fields = audience.await_args.args[0].fields
    assert fields[SubscriberField.TRIAL_STARTED] == _day(1788000000)


@pytest.mark.asyncio
async def test_a_conversion_starts_the_subscription_when_the_trial_ended(
    trial, trial_group, fields_on
):
    trial, raw = _state(trial, "converted")
    audience = AsyncMock(return_value=NotificationResult(success=True))
    await _notify(trial, raw, "converted", audience=audience)
    fields = audience.await_args_list[0].args[0].fields
    assert fields[SubscriberField.SUBSCRIPTION_STARTED] == _day(raw["trial_end"])
    assert fields[SubscriberField.SUBSCRIPTION_ENDED] is None


@pytest.mark.asyncio
async def test_without_a_trial_group_the_status_still_goes_on_its_own(
    trial, monkeypatch, fields_on
):
    monkeypatch.setattr(
        trial_audience,
        "settings",
        SimpleNamespace(config=SimpleNamespace(mailerlite_trial_group_id="")),
    )
    trial, raw = _state(trial, "canceled")
    audience = AsyncMock(return_value=NotificationResult(success=True))
    got, notice, _ = await _notify(trial, raw, "canceled", audience=audience)
    assert got == [AudienceAction.UPDATE_FIELDS]
    assert audience.await_args.args[0].fields == {
        SubscriberField.STATUS: SubscriptionStatus.TRIAL_CANCELED.value
    }
    notice.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_reminder_changes_no_fields(trial, trial_group, fields_on):
    trial, raw = _state(trial, "started")
    raw["trial_end"] = int((datetime.now(UTC) + timedelta(days=1)).timestamp())
    got, notice, _ = await _notify(trial, raw, "ending")
    assert got == []
    notice.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_consumer_writes_fields_with_no_group_change():
    event = AudienceEventModel(
        action=AudienceAction.UPDATE_FIELDS,
        email=EMAIL,
        user_id="user-1",
        fields={SubscriberField.STATUS: "signed"},
    )
    with patch.object(mailerlite, "update_fields", AsyncMock()) as called:
        assert await NotificationManager._process_audience_change(
            MagicMock(), event.model_dump_json()
        )
    called.assert_awaited_once_with(EMAIL, {SubscriberField.STATUS: "signed"})
