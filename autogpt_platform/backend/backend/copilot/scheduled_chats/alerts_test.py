from prisma.enums import AlertCause

from backend.copilot.scheduled_chats.alerts import (
    FOLLOWUPS_PATH,
    alert_key,
    paused_cause,
    step_label,
)
from backend.copilot.scheduled_chats.outcomes import RunFailure

_TWITTER = RunFailure(key="connection:twitter", kind="connection", provider="twitter")


def test_a_paused_chat_reads_as_paused_itself_with_the_failing_step():
    cause = paused_cause(
        _TWITTER, title="Morning briefing", failures=3, session_id="session 1"
    )

    assert cause.cause == AlertCause.PAUSED_FAILURES
    assert cause.headline == "Morning briefing paused itself"
    assert cause.preheader == (
        "The step “Twitter connection” failed on 3 consecutive runs."
    )
    # The button opens the chat, where the card has Reconnect and Resume.
    assert cause.cta_path == "/copilot?sessionId=session%201"


def test_without_a_chat_the_button_opens_the_scheduled_page():
    cause = paused_cause(_TWITTER, title="B", failures=3, session_id=None)

    assert cause.cta_path == FOLLOWUPS_PATH


def test_step_labels():
    assert step_label(_TWITTER) == "Twitter connection"
    assert (
        step_label(
            RunFailure(key="chat_link:discord", kind="chat_link", provider="discord")
        )
        == "Posting to Discord"
    )
    assert (
        step_label(RunFailure(key="turn:max_budget_exhausted", kind="budget"))
        == "Spending limit"
    )
    assert (
        step_label(RunFailure(key="turn:max_turns_exhausted", kind="budget"))
        == "Step limit"
    )
    assert (
        step_label(RunFailure(key="tool:run_agent:x", kind="error", tool="run_agent"))
        == "Run agent"
    )


def test_a_step_with_no_provider_named_still_reads_as_a_step():
    """The email reads "The step “…” failed", so a connection or chat
    platform it couldn't name gets a label of its own."""
    assert (
        step_label(
            RunFailure(key="connection:run_block", kind="connection", tool="run_block")
        )
        == "Connected account"
    )
    assert (
        step_label(RunFailure(key="chat_link:post", kind="chat_link", tool="post"))
        == "Posting to chat"
    )


def test_one_alert_per_scheduled_chat():
    assert alert_key("routine-1") == "scheduled_chat:routine-1"
