"""A turn that lands at a limit closes as a pause, not as an error."""

from datetime import UTC, datetime

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)

from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamTextDelta,
    StreamTurnPaused,
)
from backend.copilot.turn_budget import TurnBudget

from .governor import TurnGovernor
from .response_adapter import SDKResponseAdapter
from .tool_adapter import MCP_TOOL_PREFIX

RESETS = datetime(2026, 10, 15, tzinfo=UTC)


def _adapter(*, landed: bool) -> SDKResponseAdapter:
    # A $0 budget is spent before the first call, so one refusal lands it.
    budget = TurnBudget(usd=0.0 if landed else 5.0, limiting="daily", resets_at=RESETS)
    governor = TurnGovernor(budget=budget, step_checkin=450)
    governor.refusal()
    adapter = SDKResponseAdapter(message_id="m1", session_id="s1")
    adapter.governor = governor
    adapter.convert_message(SystemMessage(subtype="init", data={}))
    return adapter


def _result(subtype: str, *, output_tokens: int = 10) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=100,
        duration_api_ms=50,
        is_error=subtype != "success",
        num_turns=3,
        session_id="s1",
        result="",
        usage={"output_tokens": output_tokens},
    )


def _paused(events: list[StreamBaseResponse]) -> dict:
    parts = [e for e in events if isinstance(e, StreamTurnPaused)]
    assert len(parts) == 1, events
    assert isinstance(events[-1], StreamFinish)
    assert parts[0] is events[-2]
    assert not any(isinstance(e, StreamError) for e in events)
    return parts[0].pause


def test_landed_turn_with_a_checkpoint_pauses_after_the_reply():
    adapter = _adapter(landed=True)
    reply = adapter.convert_message(
        AssistantMessage(
            content=[TextBlock(text="Where I got to: sent 5 of 12.")], model="test"
        )
    )
    assert any(isinstance(e, StreamTextDelta) for e in reply)
    pause = _paused(adapter.convert_message(_result("success")))
    assert pause["reason"] == "daily"
    assert pause["checkpoint_saved"] is True
    assert pause["resets_at"].startswith("2026-10-15")


def test_budget_backstop_pauses_instead_of_erroring():
    pause = _paused(
        _adapter(landed=False).convert_message(_result("error_max_budget_usd"))
    )
    assert pause["reason"] == "daily"
    assert pause["checkpoint_saved"] is False


def test_turn_backstop_is_a_step_checkin():
    pause = _paused(_adapter(landed=False).convert_message(_result("error_max_turns")))
    assert pause["reason"] == "steps"
    assert pause["checkpoint_saved"] is False


def test_landed_turn_with_no_reply_pauses_rather_than_reporting_empty():
    adapter = _adapter(landed=True)
    adapter.convert_message(
        AssistantMessage(
            content=[
                ToolUseBlock(id="t-1", name=f"{MCP_TOOL_PREFIX}run_block", input={})
            ],
            model="test",
        )
    )
    adapter.convert_message(AssistantMessage(content=[], model="test"))
    pause = _paused(adapter.convert_message(_result("success", output_tokens=0)))
    assert pause["checkpoint_saved"] is False


def test_an_error_while_landing_is_still_reported_as_an_error():
    adapter = _adapter(landed=True)
    events = adapter.convert_message(_result("error"))
    assert any(isinstance(e, StreamError) for e in events)
    assert not any(isinstance(e, StreamTurnPaused) for e in events)


def test_a_turn_that_did_not_land_finishes_normally():
    adapter = _adapter(landed=False)
    adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="All done.")], model="test")
    )
    events = adapter.convert_message(_result("success"))
    assert isinstance(events[-1], StreamFinish)
    assert not any(isinstance(e, StreamTurnPaused) for e in events)


def test_a_closing_line_filled_in_for_a_thinking_only_reply_is_not_a_checkpoint():
    # The model answered the refused tools with thinking only, twice; the
    # adapter shows a closing line (its thinking, or a placeholder) so the
    # turn visibly ends. The model wrote no checkpoint, so the pause says so.
    adapter = _adapter(landed=True)
    adapter._last_thinking_content = "Maybe I should stop here."
    adapter.thinking_only_reprompted = True
    adapter._any_tool_results_seen = True
    adapter._text_since_last_tool_result = False

    events = adapter.convert_message(_result("success"))

    assert any(isinstance(e, StreamTextDelta) for e in events)
    assert _paused(events)["checkpoint_saved"] is False
