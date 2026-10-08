"""The role backfill takes only reliable picks, fills gaps without replacing a
newer pick on either side, never creates a MailerLite subscriber, and keeps
anyone MailerLite may not hold out of it."""

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.onboarding_role import OnboardingRole
from backend.notifications import role_backfill
from backend.notifications.mailerlite import pseudonym
from backend.notifications.role_backfill import RoleAssignment, RoleRecord

OPTED_OUT = datetime(2026, 10, 2, tzinfo=UTC)


def _record(email: str, **fields) -> RoleRecord:
    return RoleRecord(user_id=f"u-{email}", email=email, **fields)


def test_only_reliable_picks_are_planned():
    plan = role_backfill.plan(
        [
            _record("kept@x.io", choice="Other", other="Dentist"),
            _record("exact@x.io", understanding_role="Sales/BD"),
            # The kept pick wins over whatever the understanding says now.
            _record("both@x.io", choice="Marketing", understanding_role="CMO"),
            _record("typed@x.io", understanding_role="Dentist"),
            _record("rewritten@x.io", understanding_role="decision maker"),
        ],
        current={},
    )
    assert (plan.accounts, plan.kept, plan.exact, plan.skipped) == (5, 2, 1, 2)
    assert plan.roles == {"Other": 1, "Sales / BD": 1, "Marketing": 1}
    assert {a.email: a.role for a in plan.posthog} == {
        "kept@x.io": OnboardingRole(choice="Other", other="Dentist"),
        "exact@x.io": OnboardingRole(choice="Sales/BD"),
        "both@x.io": OnboardingRole(choice="Marketing"),
    }


def test_mailerlite_only_fills_in_its_own_subscribers_without_a_role():
    plan = role_backfill.plan(
        [
            _record("empty@x.io", understanding_role="Marketing"),
            _record("set@x.io", understanding_role="Marketing"),
            _record("absent@x.io", understanding_role="Marketing"),
        ],
        current={
            "empty@x.io": {"role": None, "subscription_status": "signed"},
            "set@x.io": {"role": "Engineering"},
        },
    )
    assert [a.email for a in plan.mailerlite] == ["empty@x.io"]
    assert plan.already_set == 1
    assert plan.not_subscribers == 1
    assert len(plan.posthog) == 3


def test_anyone_mailerlite_may_not_hold_still_reaches_posthog():
    """Marketing consent and the Iran and Russia rule are MailerLite's; the
    role is analytics too."""
    records = [
        _record(
            "out@x.io", understanding_role="Marketing", marketing_opt_out_at=OPTED_OUT
        ),
        _record(
            "moscow@x.io", understanding_role="Marketing", timezone="Europe/Moscow"
        ),
        _record("sam@firma.ir", understanding_role="Marketing"),
        _record("billed@x.io", understanding_role="Marketing", billing_country="RU"),
        _record("seen@x.io", understanding_role="Marketing", excluded_country="IR"),
    ]
    current = {r.email: {} for r in records}
    plan = role_backfill.plan(records, current)
    assert plan.mailerlite == []
    assert (plan.opted_out, plan.excluded_country) == (1, 4)
    assert len(plan.posthog) == 5


def _assignment(email: str = "sam@x.io") -> RoleAssignment:
    return RoleAssignment(
        user_id="u-1", email=email, role=OnboardingRole(choice="Other", other="CFO")
    )


def _response(status: int, body: dict | None = None) -> MagicMock:
    response = MagicMock(status=status)
    response.json.return_value = body or {}
    return response


@pytest.fixture
def mailerlite(monkeypatch):
    monkeypatch.setattr(role_backfill.asyncio, "sleep", AsyncMock())
    client = MagicMock(put=AsyncMock(return_value=_response(200)))
    monkeypatch.setattr(role_backfill, "_client", lambda: client)
    monkeypatch.setattr(role_backfill, "_headers", lambda: {})
    lookup = AsyncMock(return_value={"id": "ml_1", "fields": {"role": None}})
    monkeypatch.setattr(role_backfill, "_find_subscriber", lookup)
    return client, lookup


@pytest.mark.asyncio
async def test_each_subscriber_is_updated_by_id_never_upserted(mailerlite):
    client, lookup = mailerlite
    assert await role_backfill.apply([_assignment()]) == (1, 0, 0)
    lookup.assert_awaited_once_with("sam@x.io")
    url = client.put.await_args.args[0]
    assert url.endswith("/subscribers/ml_1")
    assert client.put.await_args.kwargs["json"] == {
        "fields": {"role": "Other", "role_other": "CFO"}
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subscriber",
    [None, {"id": "ml_1", "fields": {"role": "Engineering"}}],
    ids=["gone", "picked-meanwhile"],
)
async def test_someone_who_needs_nothing_any_more_is_skipped(mailerlite, subscriber):
    """The live code keeps writing while a run lasts: a role it wrote
    meanwhile is the newer pick."""
    client, lookup = mailerlite
    lookup.return_value = subscriber
    assert await role_backfill.apply([_assignment()]) == (0, 0, 1)
    client.put.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_subscriber_deleted_before_the_write_is_skipped(mailerlite):
    client, _ = mailerlite
    client.put.return_value = _response(404)
    assert await role_backfill.apply([_assignment()]) == (0, 0, 1)


@pytest.mark.asyncio
async def test_a_failure_is_counted_by_pseudonym_and_the_run_goes_on(
    mailerlite, caplog
):
    client, _ = mailerlite
    client.put.side_effect = [
        _response(422, {"message": "nope"}),
        RuntimeError("network"),
        _response(200),
    ]
    changes = [_assignment(f"{n}@x.io") for n in ("a", "b", "c")]
    with caplog.at_level(logging.WARNING, logger=role_backfill.__name__):
        assert await role_backfill.apply(changes) == (1, 2, 0)
    assert pseudonym("a@x.io") in caplog.text
    assert "a@x.io" not in caplog.text


@pytest.mark.asyncio
async def test_posthog_gets_a_set_once_per_person_and_a_flush(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(role_backfill, "get_posthog_client", lambda: client)
    sent = MagicMock()
    monkeypatch.setattr(role_backfill, "set_onboarding_role", sent)
    monkeypatch.setattr(role_backfill, "POSTHOG_FLUSH_EVERY", 2)

    await role_backfill.send_to_posthog([_assignment() for _ in range(3)])

    assert sent.call_count == 3
    assert all(call.kwargs["keep_existing"] for call in sent.call_args_list)
    assert client.flush.call_count == 2


@pytest.mark.asyncio
async def test_posthog_must_be_configured(monkeypatch):
    monkeypatch.setattr(role_backfill, "get_posthog_client", lambda: None)
    with pytest.raises(RuntimeError, match="PostHog is not configured"):
        await role_backfill.send_to_posthog([_assignment()])
