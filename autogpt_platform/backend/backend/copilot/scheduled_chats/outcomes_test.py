import pytest

from backend.copilot.scheduled_chats.outcomes import (
    UNCLASSIFIED,
    ToolFailureDetail,
    classify_run,
    tool_failure_detail,
)
from backend.copilot.tools.models import ErrorResponse


def _run(**overrides):
    base = dict(
        cancelled=False,
        turn_error_code=None,
        turn_errored=False,
        tool_failures=[],
        tool_successes=0,
    )
    base.update(overrides)
    return classify_run(**base)


def _rejected(provider: str = "twitter") -> ToolFailureDetail:
    return ToolFailureDetail(error_type="credential_rejected", provider=provider)


def test_a_run_whose_tools_worked_is_fine():
    outcome = _run(tool_successes=3)

    assert outcome.failure is None
    assert not outcome.idle
    assert not outcome.neutral


def test_a_run_that_called_no_tool_is_idle():
    assert _run().idle


def test_a_refused_connection_with_other_steps_working_fails_one_step():
    """The Twitter briefing: it still went out, without Twitter."""
    outcome = _run(tool_failures=[("run_block", _rejected())], tool_successes=4)

    assert outcome.failure is not None
    assert outcome.failure.key == "connection:twitter"
    assert outcome.failure.kind == "connection"
    assert outcome.failure.provider == "twitter"
    assert not outcome.whole_run


def test_a_run_whose_only_step_failed_fails_outright():
    outcome = _run(tool_failures=[("run_block", _rejected("discord"))])

    assert outcome.whole_run
    assert outcome.failure is not None
    assert outcome.failure.key == "connection:discord"


def test_a_connection_the_owner_can_fix_explains_the_run_first():
    other = ToolFailureDetail(error_type=UNCLASSIFIED, text_hash="abc")
    outcome = _run(
        tool_failures=[("web_fetch", other), ("run_block", _rejected())],
        tool_successes=1,
    )

    assert outcome.failure is not None
    assert outcome.failure.kind == "connection"


@pytest.mark.parametrize(
    "failures",
    [
        [("run_block", _rejected("twitter")), ("run_block", _rejected("discord"))],
        [
            ("web_fetch", ToolFailureDetail(error_type=UNCLASSIFIED, text_hash="b")),
            ("run_agent", ToolFailureDetail(error_type=UNCLASSIFIED, text_hash="a")),
        ],
    ],
    ids=["two-connections", "two-errors"],
)
def test_a_run_failing_the_same_ways_fails_the_same_way_in_any_order(failures):
    """Tools called together finish in any order: whichever came back first
    must not decide, or the streak would start over every run."""
    first = _run(tool_failures=failures)
    reversed_ = _run(tool_failures=failures[::-1])

    assert first.failure and reversed_.failure
    assert first.failure.key == reversed_.failure.key


def test_an_unlinked_chat_platform_is_its_own_kind():
    detail = ToolFailureDetail(error_type="no_linked_servers", provider="discord")
    outcome = _run(tool_failures=[("post_to_chat_platform", detail)])

    assert outcome.failure is not None
    assert outcome.failure.key == "chat_link:discord"
    assert outcome.failure.kind == "chat_link"


def test_an_uncoded_error_is_compared_by_its_hash():
    a = ToolFailureDetail(error_type=UNCLASSIFIED, text_hash="aaa")
    b = ToolFailureDetail(error_type=UNCLASSIFIED, text_hash="bbb")

    first = _run(tool_failures=[("run_agent", a)])
    same = _run(tool_failures=[("run_agent", a)])
    other = _run(tool_failures=[("run_agent", b)])

    assert first.failure and same.failure and other.failure
    assert first.failure.key == same.failure.key == "tool:run_agent:aaa"
    assert first.failure.key != other.failure.key


@pytest.mark.parametrize("code", ["max_budget_exhausted", "max_turns_exhausted"])
def test_a_run_over_its_own_limit_fails_outright(code):
    outcome = _run(turn_errored=True, turn_error_code=code, tool_successes=2)

    assert outcome.whole_run
    assert outcome.failure is not None
    assert outcome.failure.key == f"turn:{code}"
    assert outcome.failure.kind == "budget"


@pytest.mark.parametrize(
    "code", ["transient_api_error", "idle_timeout", "sdk_stream_error", None]
)
def test_the_platforms_weather_neither_counts_nor_clears(code):
    assert _run(turn_errored=True, turn_error_code=code).neutral


def test_a_cancelled_run_neither_counts_nor_clears():
    outcome = _run(
        cancelled=True, turn_errored=True, tool_failures=[("x", _rejected())]
    )

    assert outcome.neutral


def test_tool_failure_detail_reads_the_code_and_provider():
    payload = ErrorResponse(
        message="Twitter rejected the saved credential (HTTP 401).",
        error="credential_rejected",
        details={"provider": "twitter"},
    ).model_dump_json()

    detail = tool_failure_detail(payload)

    assert detail == ToolFailureDetail(
        error_type="credential_rejected", provider="twitter"
    )


def test_tool_failure_detail_hashes_free_text_instead_of_keeping_it():
    payload = ErrorResponse(
        message="Failed to execute block",
        error="Unauthorized (401) for user nick@example.com",
    ).model_dump()

    detail = tool_failure_detail(payload)

    assert detail is not None
    assert detail.error_type == UNCLASSIFIED
    assert detail.text_hash is not None
    assert "nick" not in detail.model_dump_json()


def test_an_uncoded_error_differing_only_in_its_numbers_fails_the_same_way():
    """Ids, times and durations change every run; the failure doesn't."""

    def hashed(error: str) -> str | None:
        detail = tool_failure_detail(
            ErrorResponse(message="x", error=error).model_dump()
        )
        assert detail is not None
        return detail.text_hash

    first = hashed("Timed out after 30.2s (request 8f3a9c, 2026-10-19T08:00:00Z)")
    again = hashed("Timed out after 31.7s (request 77b1e0, 2026-10-20T08:00:00Z)")
    other = hashed("Connection refused (request 8f3a9c)")

    assert first == again
    assert first != other


def test_tool_failure_detail_ignores_anything_but_an_error_payload():
    assert tool_failure_detail('{"type": "block_output"}') is None
    assert tool_failure_detail("not json") is None
