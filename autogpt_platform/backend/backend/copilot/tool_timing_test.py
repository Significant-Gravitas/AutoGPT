"""A tool call's duration waits from the tool returning to its result being saved."""

from backend.copilot import tool_timing


def test_a_duration_is_handed_over_once_and_rounded():
    tool_timing.note("call-1", 4231.6)

    assert tool_timing.take("call-1") == 4232
    assert tool_timing.take("call-1") is None


def test_an_untimed_or_unnamed_call_has_no_duration():
    tool_timing.note("", 10)

    assert tool_timing.take("") is None
    assert tool_timing.take(None) is None
    assert tool_timing.take("never-ran") is None


def test_only_the_newest_durations_are_kept():
    for n in range(tool_timing._MAX_REMEMBERED + 5):
        tool_timing.note(f"old-{n}", n)

    assert tool_timing.take("old-0") is None
    assert tool_timing.take(f"old-{tool_timing._MAX_REMEMBERED + 4}") is not None
