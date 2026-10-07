"""The backfill must put each existing customer where the live handlers would
have, and never fight MailerLite's own tour → changelog handoff."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.notifications import mailerlite, mailerlite_backfill
from backend.notifications.mailerlite_backfill import (
    Audience,
    Customer,
    Decision,
    Standing,
    Subscription,
)

TOUR, CHANGELOG, TRIAL = "grp_tour", "grp_changelog", "grp_trial"
OPTED_OUT = datetime(2026, 10, 2, tzinfo=UTC)


@pytest.fixture
def configured(monkeypatch):
    fake = SimpleNamespace(
        config=SimpleNamespace(
            mailerlite_onboarding_group_id=TOUR,
            mailerlite_changelog_group_id=CHANGELOG,
            mailerlite_trial_group_id=TRIAL,
        ),
        secrets=SimpleNamespace(mailerlite_api_token="token"),
    )
    monkeypatch.setattr(mailerlite, "settings", fake)
    monkeypatch.setattr(mailerlite_backfill, "settings", fake)
    return fake


@pytest.fixture
def no_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(mailerlite_backfill.asyncio, "sleep", sleep)
    return sleep


def _sub(status: str) -> Subscription:
    """`trialing+cancel` is a trial set to cancel; `past_due+trial` is past due
    on a subscription that started as a trial."""
    name, _, flag = status.partition("+")
    return Subscription(
        status=name, cancel_at_period_end=flag == "cancel", from_trial=flag == "trial"
    )


def _customer(email: str, *statuses: str) -> Customer:
    return Customer(
        user_id=f"u-{email}", email=email, subscriptions=[_sub(s) for s in statuses]
    )


def _audience(tour=(), changelog=(), trial=()) -> Audience:
    return Audience(
        tour={e: f"sub-{e}" for e in tour},
        changelog={e: f"sub-{e}" for e in changelog},
        trial={e: f"sub-{e}" for e in trial},
    )


def _response(status: int, body: dict) -> MagicMock:
    response = MagicMock(status=status)
    response.json.return_value = body
    return response


def _batch_ok(requests: list[dict], code: int = 201) -> MagicMock:
    return _response(200, {"responses": [{"code": code, "body": {}} for _ in requests]})


@pytest.mark.parametrize(
    "statuses, standing",
    [
        (["active"], Standing.PAYING),
        (["past_due"], Standing.PAYING),
        (["canceled", "active"], Standing.PAYING),
        (["trialing"], Standing.TRIALING),
        (["canceled", "trialing"], Standing.TRIALING),
        (["trialing+cancel"], Standing.TRIAL_CANCELLING),
        (["trialing+cancel", "trialing"], Standing.TRIALING),
        (["past_due+trial"], Standing.UNSETTLED),
        (["past_due+trial", "active"], Standing.PAYING),
        (["canceled"], Standing.CHURNED),
        (["canceled", "incomplete_expired"], Standing.CHURNED),
        (["unpaid"], Standing.UNSETTLED),
        (["incomplete"], Standing.UNSETTLED),
        (["canceled", "paused"], Standing.UNSETTLED),
        ([], Standing.UNSETTLED),
    ],
)
def test_classify(statuses, standing):
    assert mailerlite_backfill.classify([_sub(s) for s in statuses]) is standing


@pytest.mark.parametrize(
    "statuses, audience, decisions",
    [
        (["active"], _audience(), [Decision.ADD_CHANGELOG]),
        # Mid-tour, or finished and already moved across by MailerLite.
        (["active"], _audience(tour=["a@x.io"]), [Decision.SKIP_IN_TOUR]),
        (["active"], _audience(changelog=["a@x.io"]), [Decision.ALREADY_CORRECT]),
        (
            ["active"],
            _audience(trial=["a@x.io"]),
            [Decision.REMOVE_TRIAL, Decision.ADD_CHANGELOG],
        ),
        (["trialing"], _audience(), [Decision.ADD_TRIAL]),
        (["trialing"], _audience(trial=["a@x.io"]), [Decision.ALREADY_CORRECT]),
        # The trial group is "currently on a trial": cancelling means leaving.
        (["trialing+cancel"], _audience(trial=["a@x.io"]), [Decision.REMOVE_TRIAL]),
        (["trialing+cancel"], _audience(), [Decision.ALREADY_CORRECT]),
        # A trial cancel only leaves the trial group live; the changelog is
        # for churn, and a trialist has not churned.
        (
            ["trialing+cancel"],
            _audience(changelog=["a@x.io"]),
            [Decision.ALREADY_CORRECT],
        ),
        (
            ["trialing+cancel"],
            _audience(changelog=["a@x.io"], trial=["a@x.io"]),
            [Decision.REMOVE_TRIAL],
        ),
        # A failed conversion is neither on a trial nor paying.
        (
            ["past_due+trial"],
            _audience(trial=["a@x.io"]),
            [Decision.REMOVE_TRIAL, Decision.SKIP_UNSETTLED],
        ),
        (["canceled"], _audience(changelog=["a@x.io"]), [Decision.REMOVE_CHANGELOG]),
        (
            ["canceled"],
            _audience(changelog=["a@x.io"], trial=["a@x.io"]),
            [Decision.REMOVE_TRIAL, Decision.REMOVE_CHANGELOG],
        ),
        (["canceled"], _audience(), [Decision.ALREADY_CORRECT]),
        (["unpaid"], _audience(changelog=["a@x.io"]), [Decision.SKIP_UNSETTLED]),
    ],
)
def test_decide(statuses, audience, decisions):
    change = mailerlite_backfill.decide(
        _customer("a@x.io", *statuses), audience, trial_enabled=True
    )
    assert change.decisions == decisions


def test_trialing_is_skipped_without_a_trial_group():
    change = mailerlite_backfill.decide(
        _customer("a@x.io", "trialing"), _audience(), trial_enabled=False
    )
    assert change.decisions == [Decision.SKIP_NO_TRIAL_GROUP]


def test_membership_matches_case_insensitively():
    change = mailerlite_backfill.decide(
        _customer(" A@X.io ", "active"),
        _audience(changelog=["a@x.io"]),
        trial_enabled=True,
    )
    assert change.decisions == [Decision.ALREADY_CORRECT]


def test_plan_reads_the_trial_setting(configured):
    configured.config.mailerlite_trial_group_id = ""
    [change] = mailerlite_backfill.plan([_customer("a@x.io", "trialing")], _audience())
    assert change.decisions == [Decision.SKIP_NO_TRIAL_GROUP]


@pytest.mark.asyncio
async def test_apply_sends_the_live_handlers_calls(configured, no_sleep, monkeypatch):
    client = MagicMock()
    client.post = AsyncMock(
        side_effect=lambda url, **kw: _batch_ok(kw["json"]["requests"])
    )
    monkeypatch.setattr(mailerlite_backfill, "_client", lambda: client)
    audience = _audience(changelog=["gone@x.io"], trial=["paid@x.io"])
    changes = mailerlite_backfill.plan(
        [
            _customer("new@x.io", "active"),
            _customer("trial@x.io", "trialing"),
            _customer("gone@x.io", "canceled"),
            _customer("paid@x.io", "active"),
        ],
        audience,
    )

    result = await mailerlite_backfill.apply(changes, audience)

    sent = [call.kwargs["json"]["requests"] for call in client.post.await_args_list]
    assert all(
        c.args[0] == f"{mailerlite.API_BASE}/batch" for c in client.post.await_args_list
    )
    assert sent == [
        [
            {
                "method": "POST",
                "path": "api/subscribers",
                "body": {"email": "new@x.io", "groups": [CHANGELOG]},
            },
            {
                "method": "POST",
                "path": "api/subscribers",
                "body": {"email": "paid@x.io", "groups": [CHANGELOG]},
            },
        ],
        [
            {
                "method": "POST",
                "path": "api/subscribers",
                "body": {"email": "trial@x.io", "groups": [TRIAL]},
            }
        ],
        [
            {
                "method": "DELETE",
                "path": f"api/subscribers/sub-gone@x.io/groups/{CHANGELOG}",
            }
        ],
        [{"method": "DELETE", "path": f"api/subscribers/sub-paid@x.io/groups/{TRIAL}"}],
    ]
    assert result.succeeded == {
        Decision.ADD_CHANGELOG: 2,
        Decision.ADD_TRIAL: 1,
        Decision.REMOVE_CHANGELOG: 1,
        Decision.REMOVE_TRIAL: 1,
    }
    assert sum(result.failed.values()) == 0


@pytest.mark.asyncio
async def test_apply_paces_batches_for_the_import_limit(
    configured, no_sleep, monkeypatch
):
    client = MagicMock()
    client.post = AsyncMock(
        side_effect=lambda url, **kw: _batch_ok(kw["json"]["requests"])
    )
    monkeypatch.setattr(mailerlite_backfill, "_client", lambda: client)
    customers = [_customer(f"c{i}@x.io", "active") for i in range(120)]
    changes = mailerlite_backfill.plan(customers, _audience())

    await mailerlite_backfill.apply(changes, _audience())

    sizes = [len(c.kwargs["json"]["requests"]) for c in client.post.await_args_list]
    assert sizes == [50, 50, 20]
    assert [c.args[0] for c in no_sleep.await_args_list] == [
        mailerlite_backfill.UPSERT_BATCH_INTERVAL_SECONDS
    ] * 2


# MailerLite's validation error, echoing the address back in another case.
_REFUSAL = {
    "message": "The given data was invalid.",
    "errors": {"email": ["Bad@x.io is not a deliverable address."]},
}


@pytest.mark.asyncio
async def test_apply_counts_failures_and_treats_gone_as_removed(
    configured, no_sleep, monkeypatch, caplog
):
    client = MagicMock()
    client.post = AsyncMock(
        side_effect=[
            _response(200, {"responses": [{"code": 422, "body": _REFUSAL}]}),
            _response(200, {"responses": [{"code": 404}]}),
        ]
    )
    monkeypatch.setattr(mailerlite_backfill, "_client", lambda: client)
    audience = _audience(changelog=["gone@x.io"])
    changes = mailerlite_backfill.plan(
        [_customer("bad@x.io", "active"), _customer("gone@x.io", "canceled")], audience
    )

    result = await mailerlite_backfill.apply(changes, audience)

    assert result.failed[Decision.ADD_CHANGELOG] == 1
    assert result.succeeded[Decision.REMOVE_CHANGELOG] == 1
    assert "bad@x.io" not in caplog.text.lower()
    assert f"{mailerlite.pseudonym('bad@x.io')} at .io with 422" in caplog.text
    assert "The given data was invalid." in caplog.text
    assert "is not a deliverable address" in caplog.text


def test_failure_reason_without_a_body():
    assert mailerlite_backfill._failure_reason(None, "a@x.io") == "no reason given"
    assert mailerlite_backfill._failure_reason({}, "a@x.io") == "no reason given"


@pytest.mark.parametrize(
    "email, expected",
    [
        ("a@example.com", ".com"),
        ("a@mail.example.co", ".co"),
        ("a@example.fart", ".fart"),
        ("a@example.co.uk", ".co.uk"),
        ("a@Example.COM.AU ", ".com.au"),
        ("a@co.uk", ".uk"),
        ("a@localhost", "no TLD"),
        ("a@example.", "no TLD"),
    ],
)
def test_top_level_domain(email, expected):
    assert mailerlite_backfill._top_level_domain(email) == expected


@pytest.mark.asyncio
async def test_a_short_batch_answer_is_an_error(configured, no_sleep, monkeypatch):
    client = MagicMock()
    client.post = AsyncMock(return_value=_response(200, {"responses": []}))
    monkeypatch.setattr(mailerlite_backfill, "_client", lambda: client)
    changes = mailerlite_backfill.plan([_customer("a@x.io", "active")], _audience())

    with pytest.raises(mailerlite.MailerLiteError):
        await mailerlite_backfill.apply(changes, _audience())


@pytest.mark.asyncio
async def test_read_audience_pages_every_status(configured, monkeypatch):
    pages = {
        (TOUR, "active", None): {
            "data": [{"id": "1", "email": "T@x.io"}],
            "meta": {"next_cursor": "c2"},
        },
        (TOUR, "active", "c2"): {"data": [{"id": "2", "email": "t2@x.io"}], "meta": {}},
        (CHANGELOG, "unsubscribed", None): {
            "data": [{"id": "3", "email": "u@x.io"}],
            "meta": {},
        },
    }

    async def get(url, **kw):
        path, _, query = url.partition("?")
        params = dict(p.split("=", 1) for p in query.split("&"))
        group = path.split("/")[-2]
        key = (group, params["filter%5Bstatus%5D"], params.get("cursor"))
        return _response(200, pages.get(key, {"data": [], "meta": {}}))

    client = MagicMock()
    client.get = AsyncMock(side_effect=get)
    monkeypatch.setattr(mailerlite_backfill, "_client", lambda: client)

    audience = await mailerlite_backfill.read_audience()

    assert audience.tour == {"t@x.io": "1", "t2@x.io": "2"}
    # Unsubscribed is still a member, so it is not re-added.
    assert audience.changelog == {"u@x.io": "3"}
    assert audience.trial == {}


@pytest.mark.asyncio
async def test_read_audience_needs_the_groups(configured):
    configured.config.mailerlite_changelog_group_id = ""
    with pytest.raises(mailerlite.MailerLiteNotConfigured):
        await mailerlite_backfill.read_audience()


def test_a_second_run_finds_nothing_to_do():
    customers = [
        _customer("new@x.io", "active"),
        _customer("trial@x.io", "trialing"),
        _customer("gone@x.io", "canceled"),
    ]
    after = _audience(changelog=["new@x.io"], trial=["trial@x.io"])
    changes = [
        mailerlite_backfill.decide(c, after, trial_enabled=True) for c in customers
    ]
    assert all(c.decisions == [Decision.ALREADY_CORRECT] for c in changes)


@pytest.mark.asyncio
async def test_read_audience_stops_on_a_repeated_cursor(configured, monkeypatch):
    """A cursor MailerLite already handed out would page forever. It is an
    error, not the end, or the plan would be made from a partial read."""
    page = {"data": [{"id": "1", "email": "t@x.io"}], "meta": {"next_cursor": "c2"}}
    client = MagicMock()
    client.get = AsyncMock(side_effect=_paging_forever(page))
    monkeypatch.setattr(mailerlite_backfill, "_client", lambda: client)

    with pytest.raises(mailerlite.MailerLiteError, match="repeated"):
        await mailerlite_backfill.read_audience()
    assert client.get.await_count == 2


def _paging_forever(page: dict):
    """The same page every time. A mock never yields to the event loop, so a
    reader that does not stop would hang the test rather than time out."""
    calls = 0

    async def get(url, **kw):
        nonlocal calls
        calls += 1
        if calls > 10:
            raise AssertionError("the reader requested the same page forever")
        return _response(200, page)

    return get


# ── opted out of marketing ─────────────────────────────────────────────────


def _opted_out(email: str, *statuses: str) -> Customer:
    return _customer(email, *statuses).model_copy(
        update={"marketing_opt_out_at": OPTED_OUT}
    )


@pytest.mark.parametrize(
    "statuses, audience",
    [
        (["active"], _audience()),
        (["active"], _audience(trial=["a@x.io"])),
        (["trialing"], _audience()),
        (["trialing+cancel"], _audience(trial=["a@x.io"])),
        (["canceled"], _audience(changelog=["a@x.io"])),
        (["unpaid"], _audience()),
    ],
)
def test_an_opted_out_customer_is_only_ever_skipped(statuses, audience):
    """Not even a removal: a customer who refused marketing gets no MailerLite
    call of any kind."""
    change = mailerlite_backfill.decide(
        _opted_out("a@x.io", *statuses), audience, trial_enabled=True
    )
    assert change.decisions == [Decision.SKIP_OPTED_OUT]
    assert Decision.SKIP_OPTED_OUT not in mailerlite_backfill.CHANGES


@pytest.mark.asyncio
async def test_apply_never_writes_an_opted_out_customer(
    configured, no_sleep, monkeypatch
):
    client = MagicMock()
    client.post = AsyncMock(
        side_effect=lambda url, **kw: _batch_ok(kw["json"]["requests"])
    )
    monkeypatch.setattr(mailerlite_backfill, "_client", lambda: client)
    audience = _audience(changelog=["gone@x.io"], trial=["trial@x.io"])
    changes = mailerlite_backfill.plan(
        [
            _opted_out("new@x.io", "active"),
            _opted_out("gone@x.io", "canceled"),
            _opted_out("trial@x.io", "active"),
            _customer("paid@x.io", "active"),
        ],
        audience,
    )

    result = await mailerlite_backfill.apply(changes, audience)

    sent = [call.kwargs["json"]["requests"] for call in client.post.await_args_list]
    assert sent == [
        [
            {
                "method": "POST",
                "path": "api/subscribers",
                "body": {"email": "paid@x.io", "groups": [CHANGELOG]},
            }
        ]
    ]
    assert sum(result.succeeded.values()) == 1
    assert sum(result.failed.values()) == 0
