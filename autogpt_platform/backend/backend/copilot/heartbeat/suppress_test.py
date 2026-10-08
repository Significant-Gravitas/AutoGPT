from types import SimpleNamespace

import pytest

from backend.copilot.heartbeat.suppress import (
    ACK_MAX_CHARS,
    ExplicitResponse,
    decide,
    explicit_from_tool_calls,
    strip_tokens,
)

_LONG = "The nightly invoice sync failed three times since 6am. " * 8


@pytest.mark.parametrize(
    "reply", ["NO_REPLY", "no_reply", "**NO_REPLY**", " NO_REPLY. ", "HEARTBEAT_OK"]
)
def test_a_silent_token_alone_says_nothing(reply):
    verdict = decide(reply, None)
    assert not verdict.deliver
    assert verdict.reason == "silent"


def test_tokens_are_stripped_wherever_they_sit():
    assert strip_tokens("NO_REPLY all quiet HEARTBEAT_OK") == "all quiet"


def test_a_short_reply_that_is_not_an_alert_is_dropped():
    verdict = decide("Checked everything, all good. NO_REPLY", None)
    assert not verdict.deliver
    assert verdict.reason == "short_reply"


def test_a_long_reply_is_delivered_without_its_token():
    assert len(_LONG) >= ACK_MAX_CHARS
    verdict = decide(f"{_LONG} HEARTBEAT_OK", None)
    assert verdict.deliver
    assert verdict.reason == "long_reply"
    assert "HEARTBEAT_OK" not in verdict.text


def test_an_explicit_alert_is_delivered_however_short():
    verdict = decide(
        "NO_REPLY",
        ExplicitResponse(notify=True, notification_text="Your sync agent failed."),
    )
    assert verdict.deliver
    assert verdict.text == "Your sync agent failed."
    assert verdict.reason == "explicit_alert"


def test_an_explicit_no_outranks_a_long_reply():
    verdict = decide(_LONG, ExplicitResponse(notify=False))
    assert not verdict.deliver
    assert verdict.reason == "declined"


def test_an_explicit_alert_with_no_text_says_nothing():
    verdict = decide("", ExplicitResponse(notify=True, notification_text="NO_REPLY"))
    assert not verdict.deliver


@pytest.mark.parametrize(
    "call",
    [
        {
            "tool_name": "heartbeat_respond",
            "input": {"notify": True, "notification_text": "A"},
        },
        {
            "tool_name": "mcp__copilot__heartbeat_respond",
            "input": {"notify": True, "notification_text": "A"},
        },
        SimpleNamespace(
            tool_name="run_capability",
            input={
                "id": "tool:heartbeat_respond",
                "input": {"notify": True, "notification_text": "A"},
            },
        ),
    ],
)
def test_the_explicit_answer_is_read_however_it_was_called(call):
    found = explicit_from_tool_calls([call])
    assert found == ExplicitResponse(notify=True, notification_text="A")


def test_other_tool_calls_are_not_an_answer():
    calls = [
        {"tool_name": "memory_search", "input": {"query": "alerts"}},
        {
            "tool_name": "run_capability",
            "input": {"id": "tool:web_search", "input": {}},
        },
    ]
    assert explicit_from_tool_calls(calls) is None


def test_the_last_answer_wins():
    calls = [
        {
            "tool_name": "heartbeat_respond",
            "input": {"notify": True, "notification_text": "A"},
        },
        {"tool_name": "heartbeat_respond", "input": {"notify": False}},
    ]
    assert explicit_from_tool_calls(calls) == ExplicitResponse(notify=False)
