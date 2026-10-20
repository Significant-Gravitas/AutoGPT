from backend.copilot.hidden_rows import is_hidden_from_model
from backend.copilot.scheduled_chats.failure_notices import (
    BOTS_PATH,
    failing_notice,
    pause_summary,
    paused_notice,
    provider_name,
    resumed_notice,
)
from backend.copilot.scheduled_chats.notices import BILLING_PATH, INTEGRATIONS_PATH
from backend.copilot.scheduled_chats.outcomes import RunFailure

_TWITTER = RunFailure(key="connection:twitter", kind="connection", provider="twitter")
_DISCORD_LINK = RunFailure(
    key="chat_link:discord", kind="chat_link", provider="discord"
)
_BUDGET = RunFailure(key="turn:max_budget_exhausted", kind="budget")
_STEPS = RunFailure(key="turn:max_turns_exhausted", kind="budget")
_ERROR = RunFailure(key="tool:run_agent:abc", kind="error", tool="run_agent")


def _actions(notice):
    return [action.model_dump() for action in notice.actions]


def test_a_paused_connection_offers_the_reconnect_and_resume():
    notice = paused_notice(_TWITTER, title="Briefing", failures=3, streak_id="s1")

    assert notice.headline == "Paused: “Briefing” can't use Twitter"
    assert notice.body.startswith("Its last 3 runs failed")
    assert _actions(notice) == [
        {"kind": "reconnect", "label": "Reconnect Twitter", "provider": "twitter"},
        {"kind": "resume", "label": "Resume"},
    ]
    assert notice.window_key == "paused:s1"


def test_a_paused_chat_platform_points_at_the_bot_settings():
    notice = paused_notice(_DISCORD_LINK, title="Monitor", failures=3, streak_id="s1")

    assert notice.headline == "Paused: “Monitor” can't post to Discord"
    assert _actions(notice) == [
        {"kind": "link", "label": "Open bot settings", "href": BOTS_PATH},
        {"kind": "resume", "label": "Resume"},
    ]


def test_a_chat_over_its_spending_limit_offers_an_upgrade():
    notice = paused_notice(_BUDGET, title="Digest", failures=3, streak_id="s1")

    assert notice.headline == "Paused: “Digest” keeps hitting its spending limit"
    assert _actions(notice)[0] == {
        "kind": "link",
        "label": "Upgrade",
        "href": BILLING_PATH,
    }


def test_a_chat_out_of_steps_says_step_limit_and_offers_only_resume():
    notice = paused_notice(_STEPS, title="Digest", failures=3, streak_id="s1")

    assert "step limit" in notice.headline
    assert _actions(notice) == [{"kind": "resume", "label": "Resume"}]


def test_any_other_error_points_at_the_chat():
    notice = paused_notice(_ERROR, title="Weekly", failures=3, streak_id="s1")

    assert notice.headline == "Paused: “Weekly” keeps failing"
    assert "Open the chat to see the error" in notice.body


def test_one_that_could_not_be_paused_says_so_without_claiming_it():
    notice = paused_notice(
        _TWITTER, title="Briefing", failures=3, streak_id="s1", paused=False
    )

    assert notice.headline == "“Briefing” can't use Twitter"
    assert notice.body == (
        "Its last 3 runs failed because the Twitter connection is missing or "
        "was refused. Reconnect it."
    )
    # Nothing to resume: it is still firing.
    assert _actions(notice) == [
        {"kind": "reconnect", "label": "Reconnect Twitter", "provider": "twitter"}
    ]
    assert notice.window_key == "failing:s1"


def test_a_failing_step_is_told_without_a_resume():
    """Nothing was paused, so there is nothing to resume."""
    notice = failing_notice(_TWITTER, title="Briefing", failures=3, streak_id="s1")

    assert notice.headline == "“Briefing” can't use Twitter"
    assert "went ahead without Twitter" in notice.body
    assert [a["kind"] for a in _actions(notice)] == ["reconnect"]
    assert notice.window_key == "failing:s1"


def test_every_note_is_hidden_from_the_model():
    for notice in (
        paused_notice(_TWITTER, title="B", failures=3, streak_id="s"),
        failing_notice(_TWITTER, title="B", failures=3, streak_id="s"),
        resumed_notice(title="B", streak_id="s"),
    ):
        assert is_hidden_from_model(notice.metadata(schedule_key="k", title="B"))


def test_provider_names_read_as_people_write_them():
    assert provider_name("twitter") == "Twitter"
    assert provider_name("github") == "GitHub"
    assert provider_name("google_drive") == "Google Drive"
    assert provider_name(None) == "a connected service"


def test_a_connection_its_tool_did_not_name_points_at_settings():
    """A tool that needs several accounts doesn't say which one failed: no
    Reconnect button to offer, and no provider to put in the sentence."""
    unnamed = RunFailure(key="connection:run_block", kind="connection")

    paused = paused_notice(unnamed, title="Briefing", failures=3, streak_id="s1")
    failing = failing_notice(unnamed, title="Briefing", failures=3, streak_id="s1")

    assert paused.headline == "Paused: “Briefing” can't use one of its connections"
    assert paused.body == (
        "Its last 3 runs failed because a connection it needs is missing or was "
        "refused. Reconnect it in Settings, then resume."
    )
    assert _actions(paused) == [
        {"kind": "link", "label": "Open settings", "href": INTEGRATIONS_PATH},
        {"kind": "resume", "label": "Resume"},
    ]
    assert failing.body == (
        "The last 3 runs went ahead without a connection they need, which is "
        "missing or was refused. Reconnect it in Settings to bring it back."
    )
    assert _actions(failing) == [
        {"kind": "link", "label": "Open settings", "href": INTEGRATIONS_PATH}
    ]
    assert pause_summary(unnamed) == "Paused: can't use one of its connections"


def test_a_chat_platform_its_tool_did_not_name_reads_without_one():
    unnamed = RunFailure(key="chat_link:post", kind="chat_link")

    paused = paused_notice(unnamed, title="Monitor", failures=3, streak_id="s1")
    failing = failing_notice(unnamed, title="Monitor", failures=3, streak_id="s1")

    assert paused.headline == "Paused: “Monitor” can't post to a chat platform"
    assert "because no server or chat is linked" in paused.body
    assert "because no server or chat is linked" in failing.body
    assert pause_summary(unnamed) == "Paused: can't post to a chat platform"


def test_pause_summary():
    assert pause_summary(_TWITTER) == "Paused: can't use Twitter"
    assert pause_summary(_DISCORD_LINK) == "Paused: can't post to Discord"
    assert pause_summary(_BUDGET) == "Paused: keeps hitting its spending limit"
    assert pause_summary(_STEPS) == "Paused: keeps hitting its step limit"
    assert pause_summary(_ERROR) == "Paused: keeps failing"
