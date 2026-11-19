from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.hidden_rows import is_hidden_from_model
from backend.copilot.scheduled_chats.notices import (
    BILLING_PATH,
    INTEGRATIONS_PATH,
    NOTICE_METADATA_KIND,
    notice_message_id,
    post_notice,
    schedule_title,
    skip_notice,
)
from backend.copilot.turn_queue import TurnRefusal

_NOW = datetime(2026, 10, 11, 14, 9, tzinfo=timezone.utc)
_MIDNIGHT = datetime(2026, 10, 12, tzinfo=timezone.utc)
_MONDAY = datetime(2026, 10, 12, tzinfo=timezone.utc)


def _usage(window: str, resets_at: datetime | None = _MIDNIGHT) -> TurnRefusal:
    return TurnRefusal(
        reason="rate-limited", kind="usage", window=window, resets_at=resets_at
    )


def test_a_daily_limit_says_when_the_schedule_runs_again():
    notice = skip_notice(_usage("daily"), title="Briefing", now=_NOW, run="recurring")

    assert notice is not None
    assert notice.headline == "Skipped: daily usage limit reached"
    assert notice.body == (
        "“Briefing” didn't run. Scheduled chats run again when the limit resets "
        "at 00:00 UTC."
    )
    assert notice.resumes_at == _MIDNIGHT
    assert [(a.label, a.href) for a in notice.actions] == [("Upgrade", BILLING_PATH)]


def test_a_schedule_that_could_fit_the_limit_is_offered_to_run_less_often():
    notice = skip_notice(
        _usage("daily"),
        title="Discord monitor",
        now=_NOW,
        run="recurring",
        user_timezone="America/Chicago",
        crons=["* * * * *"],
    )

    assert notice is not None
    assert [action.model_dump() for action in notice.actions] == [
        {"kind": "link", "label": "Upgrade", "href": BILLING_PATH},
        {"kind": "run_less_often", "label": "Run less often"},
    ]


@pytest.mark.parametrize(
    ("refusal", "run", "crons"),
    [
        (_usage("daily"), "recurring", ["0 9 * * mon"]),
        (_usage("weekly", _MONDAY), "recurring", None),
        (_usage("daily"), "waits_for_reset", ["* * * * *"]),
        (_usage("daily"), "dropped", ["* * * * *"]),
        (_usage("trial"), "recurring", ["* * * * *"]),
        (TurnRefusal(reason="paywalled", kind="no_plan"), "recurring", ["* * * * *"]),
    ],
    ids=[
        "weekly-already",
        "no-crons",
        "one-shot-waits",
        "one-shot-gone",
        "trial",
        "no-plan",
    ],
)
def test_run_less_often_is_offered_only_where_it_would_help(refusal, run, crons):
    """Once a week is the bottom of the ladder; a one-shot runs once anyway;
    no cadence brings a used-up trial or a missing plan back."""
    notice = skip_notice(refusal, title="Briefing", now=_NOW, run=run, crons=crons)

    assert notice is not None
    assert "run_less_often" not in [action.kind for action in notice.actions]


def test_a_weekly_limit_names_the_day_it_resets():
    notice = skip_notice(
        _usage("weekly", _MONDAY), title="Briefing", now=_NOW, run="recurring"
    )

    assert notice is not None
    assert notice.headline == "Skipped: weekly usage limit reached"
    assert "resets on Monday at 00:00 UTC" in notice.body


def test_the_reset_is_on_the_owners_clock():
    """The schedule's timezone is on the job: nobody should have to work out
    what 00:00 UTC is where they live."""
    notice = skip_notice(
        _usage("daily"),
        title="Briefing",
        now=_NOW,
        run="recurring",
        user_timezone="America/Chicago",
    )

    assert notice is not None
    assert notice.body.endswith("when the limit resets at 19:00 CDT.")
    # The window it names is still the UTC one, so the dedupe is unchanged.
    assert notice.resumes_at == _MIDNIGHT


def test_a_weekly_reset_names_the_owners_day():
    """Monday 00:00 UTC is Sunday evening in Chicago."""
    notice = skip_notice(
        _usage("weekly", _MONDAY),
        title="Briefing",
        now=_NOW,
        run="recurring",
        user_timezone="America/Chicago",
    )

    assert notice is not None
    assert "resets on Sunday at 19:00 CDT" in notice.body


def test_an_unknown_timezone_falls_back_to_utc():
    notice = skip_notice(
        _usage("daily"),
        title="Briefing",
        now=_NOW,
        run="recurring",
        user_timezone="Mars/Olympus_Mons",
    )

    assert notice is not None
    assert notice.body.endswith("at 00:00 UTC.")


def test_a_dropped_one_shot_on_a_lost_chatgpt_plan_says_it_did_not_run():
    notice = skip_notice(
        TurnRefusal(reason="lacks Codex entitlement", kind="codex_access"),
        title="Check CI",
        now=_NOW,
        run="dropped",
    )

    assert notice is not None
    assert notice.body.startswith("“Check CI” didn't run")
    assert notice.body.endswith("then ask for it again.")


def test_a_one_shot_that_waits_says_when_it_will_run():
    notice = skip_notice(
        _usage("daily"), title="Check CI", now=_NOW, run="waits_for_reset"
    )

    assert notice is not None
    assert notice.body == "“Check CI” will run when the limit resets at 00:00 UTC."


def test_a_dropped_one_shot_promises_nothing():
    notice = skip_notice(_usage("daily"), title="Check CI", now=_NOW, run="dropped")

    assert notice is not None
    assert notice.body == "“Check CI” didn't run."
    assert notice.resumes_at is None


@pytest.mark.parametrize(
    ("refusal", "headline"),
    [
        (
            TurnRefusal(reason="paywalled", kind="no_plan"),
            "Skipped: scheduled chats need a plan",
        ),
        (
            _usage("trial", _NOW),
            "Skipped: your trial's usage limit is used up",
        ),
    ],
    ids=["no-plan", "trial-used-up"],
)
def test_no_plan_offers_a_plan(refusal, headline):
    notice = skip_notice(refusal, title="Briefing", now=_NOW, run="recurring")

    assert notice is not None
    assert notice.reason == "no_plan"
    assert notice.headline == headline
    assert notice.body == (
        "“Briefing” didn't run. Scheduled chats start again once you choose a plan."
    )
    assert [(a.label, a.href) for a in notice.actions] == [
        ("Choose a plan", BILLING_PATH)
    ]


def test_a_lost_chatgpt_connection_points_at_settings():
    notice = skip_notice(
        TurnRefusal(reason="lacks Codex entitlement", kind="codex_access"),
        title="Briefing",
        now=_NOW,
        run="recurring",
    )

    assert notice is not None
    assert notice.reason == "codex_access"
    assert [(a.label, a.href) for a in notice.actions] == [
        ("Open settings", INTEGRATIONS_PATH)
    ]


def test_an_outage_on_our_side_says_nothing():
    refusal = TurnRefusal(reason="unreadable", kind="outage", transient=True)

    assert skip_notice(refusal, title="Briefing", now=_NOW, run="recurring") is None


def test_a_banned_account_is_told_nothing():
    refusal = TurnRefusal(reason="is banned", kind="banned")

    assert skip_notice(refusal, title="Briefing", now=_NOW, run="recurring") is None


def test_a_schedule_says_it_once_per_window():
    """The message id is the dedupe: the same window maps to the same id, and
    the next window to a new one."""
    first = skip_notice(_usage("daily"), title="Briefing", now=_NOW, run="recurring")
    again = skip_notice(_usage("daily"), title="Briefing", now=_NOW, run="recurring")
    tomorrow = skip_notice(
        _usage("daily", datetime(2026, 10, 13, tzinfo=timezone.utc)),
        title="Briefing",
        now=_NOW,
        run="recurring",
    )
    assert first and again and tomorrow

    def message_id(notice):
        return notice_message_id(
            user_id="user-1", schedule_key="routine-1", window_key=notice.window_key
        )

    assert message_id(first) == message_id(again)
    assert message_id(first) != message_id(tomorrow)


def test_no_plan_says_it_at_most_weekly():
    """_NOW is a Sunday: the Saturday before is the same ISO week, the Monday
    after the next one."""
    refusal = TurnRefusal(reason="paywalled", kind="no_plan")
    sunday = skip_notice(refusal, title="B", now=_NOW, run="recurring")
    saturday = skip_notice(
        refusal,
        title="B",
        now=datetime(2026, 10, 10, tzinfo=timezone.utc),
        run="recurring",
    )
    monday = skip_notice(
        refusal,
        title="B",
        now=datetime(2026, 10, 12, tzinfo=timezone.utc),
        run="recurring",
    )
    assert sunday and saturday and monday
    assert sunday.window_key == saturday.window_key
    assert sunday.window_key != monday.window_key


def test_the_card_is_kept_from_the_model_and_names_its_schedule():
    notice = skip_notice(_usage("daily"), title="Briefing", now=_NOW, run="recurring")
    assert notice is not None

    metadata = notice.metadata(schedule_key="routine-1", title="Briefing")

    assert metadata["kind"] == NOTICE_METADATA_KIND
    assert is_hidden_from_model(metadata)
    assert metadata["schedule_key"] == "routine-1"
    assert metadata["resumes_at"] == _MIDNIGHT.isoformat()
    assert metadata["actions"] == [
        {"kind": "link", "label": "Upgrade", "href": BILLING_PATH}
    ]


@pytest.mark.parametrize(
    ("routine_title", "message", "title"),
    [
        ("Morning briefing", "Do the thing", "Morning briefing"),
        (None, "\n  Check the deploy  \nthen report", "Check the deploy"),
        (None, "x" * 80, "x" * 59 + "…"),
        (None, "   ", "Scheduled chat"),
    ],
    ids=["routine", "follow-up", "long-follow-up", "blank"],
)
def test_schedule_title(routine_title, message, title):
    assert schedule_title(routine_title=routine_title, message=message) == title


def _db() -> MagicMock:
    return MagicMock(
        append_session_message=AsyncMock(return_value=True),
        append_expert_run_message=AsyncMock(return_value="expert-thread"),
        append_plain_session_message=AsyncMock(return_value="new-chat"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("session_id", "expert_id", "called"),
    [
        ("session-1", "expert-1", "append_session_message"),
        (None, "expert-1", "append_expert_run_message"),
        (None, None, "append_plain_session_message"),
    ],
    ids=["own-thread", "expert-thread", "new-chat"],
)
async def test_the_note_goes_to_the_schedules_own_chat_first(
    session_id, expert_id, called
):
    notice = skip_notice(_usage("daily"), title="Briefing", now=_NOW, run="recurring")
    assert notice is not None
    db = _db()

    with patch("backend.copilot.scheduled_chats.notices.chat_db", return_value=db):
        posted = await post_notice(
            notice,
            user_id="user-1",
            schedule_key="routine-1",
            title="Briefing",
            session_id=session_id,
            expert_id=expert_id,
        )

    assert posted is True
    for name in (
        "append_session_message",
        "append_expert_run_message",
        "append_plain_session_message",
    ):
        method = getattr(db, name)
        assert method.await_count == int(name == called)
    kwargs = getattr(db, called).await_args.kwargs
    assert kwargs["content"] == notice.content
    assert kwargs["message_id"] == notice_message_id(
        user_id="user-1", schedule_key="routine-1", window_key=notice.window_key
    )
    assert kwargs["metadata"]["hidden_from_model"] is True
    if called == "append_plain_session_message":
        # The chat it opens is named for the schedule, as its run's would be.
        assert kwargs["title"] == "Briefing"


@pytest.mark.asyncio
async def test_a_note_already_posted_this_window_reports_nothing_new():
    notice = skip_notice(_usage("daily"), title="Briefing", now=_NOW, run="recurring")
    assert notice is not None
    db = _db()
    db.append_plain_session_message.return_value = None

    with patch("backend.copilot.scheduled_chats.notices.chat_db", return_value=db):
        posted = await post_notice(
            notice,
            user_id="user-1",
            schedule_key="routine-1",
            title="Briefing",
            session_id=None,
            expert_id=None,
        )

    assert posted is False
