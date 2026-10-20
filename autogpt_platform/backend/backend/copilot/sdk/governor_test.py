"""Landing decisions: when a turn's tools are refused, and how it pauses."""

from datetime import UTC, datetime

import pytest
from claude_agent_sdk import AssistantMessage, StreamEvent, TextBlock

from backend.copilot.turn_budget import TurnBudget
from backend.copilot.turn_pause import TurnPause

from .governor import STEP_BACKSTOP_MARGIN, TurnGovernor
from .security_hooks import create_security_hooks
from .transient_retry_test import _SVC, _reply, _Turn
from .usage_meter import UsageMeter, cli_price_ratio

RESETS = datetime(2026, 10, 15, tzinfo=UTC)
KIMI = "moonshotai/kimi-k2.6"
ALLOWED = {"mcp__copilot__run_block", "Agent"}


class _Meter(UsageMeter):
    """A meter whose readings are set directly, so thresholds are exact."""

    def __init__(self, spent: float = 0.0, calls: int = 0, last: float = 0.0):
        super().__init__()
        self.spent = spent
        self.calls = calls
        self.last = last

    def observe(self, message: AssistantMessage) -> None:
        self.calls += 1

    @property
    def spent_usd(self) -> float:
        return self.spent

    @property
    def main_calls(self) -> int:
        return self.calls

    @property
    def last_main_call_usd(self) -> float:
        return self.last


def _governor(
    budget: TurnBudget | None = TurnBudget(usd=1.0, limiting="daily", resets_at=RESETS),
    *,
    checkin: int = 450,
    meter: _Meter | None = None,
) -> TurnGovernor:
    governor = TurnGovernor(budget=budget, step_checkin=checkin)
    governor.meter = meter or _Meter()
    return governor


def test_tools_run_while_the_budget_has_room():
    governor = _governor(meter=_Meter(spent=0.5, last=0.05))
    assert governor.refusal() is None
    assert governor.landing is None


def test_lands_with_room_left_for_the_checkpoint():
    # Reserve is two calls at the latest call's price (2 x 0.1), so a $1.00
    # budget lands at $0.80 rather than letting the checkpoint overspend.
    meter = _Meter(spent=0.79, last=0.1)
    governor = _governor(meter=meter)
    assert governor.refusal() is None
    meter.spent = 0.8
    refusal = governor.refusal()
    assert refusal is not None
    assert "Where I got to:" in refusal
    assert governor.landing == "daily"


def test_reserve_never_takes_more_than_a_quarter_of_the_budget():
    # One expensive call can't make a small budget land before it starts.
    # The reserve caps at 25% of $0.40, so this lands near $0.30, not at $0.
    meter = _Meter(spent=0.29, last=0.5)
    governor = _governor(TurnBudget(usd=0.4, limiting="weekly"), meter=meter)
    assert governor.refusal() is None
    meter.spent = 0.31
    assert governor.refusal() is not None
    assert governor.landing == "weekly"


def test_landing_is_sticky():
    meter = _Meter(spent=0.99, last=0.01)
    governor = _governor(meter=meter)
    assert governor.refusal() is not None
    meter.spent = 0.0
    assert governor.refusal() is not None


def test_step_checkin_lands_with_a_progress_request():
    governor = _governor(checkin=3, meter=_Meter(calls=2))
    assert governor.refusal() is None
    governor.observe(AssistantMessage(content=[TextBlock(text="x")], model="m"))
    refusal = governor.refusal()
    assert governor.landing == "steps"
    assert refusal is not None
    assert "3 steps" in refusal
    assert "Progress:" in refusal


def test_chatgpt_turns_get_the_step_checkin_only():
    governor = _governor(budget=None, meter=_Meter(spent=999.0, calls=10))
    assert governor.refusal() is None
    assert governor.cli_budget_usd(KIMI) is None


def test_a_turn_that_never_landed_does_not_pause():
    governor = _governor()
    assert governor.pause("success", wrote_checkpoint=True) is None
    assert governor.paused is False


def test_a_landed_turn_pauses_with_its_window_and_reset():
    governor = _governor(meter=_Meter(spent=1.0))
    governor.refusal()
    pause = governor.pause("success", wrote_checkpoint=True)
    assert pause == TurnPause(reason="daily", resets_at=RESETS)
    assert governor.paused is True


@pytest.mark.parametrize("subtype", ["error", "error_during_execution"])
def test_a_genuine_error_while_landing_stays_an_error(subtype):
    governor = _governor(meter=_Meter(spent=1.0))
    governor.refusal()
    assert governor.pause(subtype, wrote_checkpoint=True) is None
    assert governor.paused is False


def test_a_landed_turn_without_a_reply_says_so():
    governor = _governor(meter=_Meter(spent=1.0))
    governor.refusal()
    pause = governor.pause("success", wrote_checkpoint=False)
    assert pause is not None
    assert pause.checkpoint_saved is False


def test_cli_turn_backstop_is_a_step_pause_without_a_checkpoint():
    governor = _governor(checkin=450, meter=_Meter(calls=500))
    pause = governor.pause("error_max_turns", wrote_checkpoint=True)
    assert pause == TurnPause(reason="steps", steps=500, checkpoint_saved=False)


@pytest.mark.parametrize(
    "budget,reason",
    [
        (TurnBudget(usd=2.0, limiting="weekly", resets_at=RESETS), "weekly"),
        (TurnBudget(usd=10.0, limiting="task"), "task"),
        (TurnBudget(usd=0.5, limiting="trial", hard_cap=True), "trial"),
        (None, "task"),
    ],
)
def test_cli_budget_backstop_names_the_limiting_window(budget, reason):
    governor = _governor(budget)
    pause = governor.pause("error_max_budget_usd", wrote_checkpoint=True)
    assert pause is not None
    assert pause.reason == reason
    assert pause.checkpoint_saved is False
    assert pause.resets_at == (RESETS if reason == "weekly" else None)


def test_a_cli_hangup_only_pauses_a_turn_that_was_landing():
    running = _governor(meter=_Meter(spent=0.1, last=0.01))
    assert running.hangup_pause() is None
    assert running.landed is False

    landing = _governor(meter=_Meter(spent=1.0))
    landing.refusal()
    assert landing.landed is True
    pause = landing.hangup_pause()
    assert pause == TurnPause(reason="daily", resets_at=RESETS, checkpoint_saved=False)
    assert landing.paused is True


def test_cli_max_turns_sits_past_the_checkin():
    assert _governor(checkin=450).cli_max_turns == 450 + STEP_BACKSTOP_MARGIN


def test_paid_backstop_adds_headroom_above_the_budget():
    governor = _governor(TurnBudget(usd=2.0, limiting="daily"))
    # max($0.25, 15% of $2.00) of headroom on Anthropic, whose CLI units are ours.
    assert governor.cli_budget_usd("anthropic/claude-sonnet-5") == pytest.approx(2.3)


def test_paid_backstop_never_drops_below_what_the_cli_accepts():
    governor = _governor(TurnBudget(usd=0.05, limiting="daily"))
    assert governor.cli_budget_usd("anthropic/claude-sonnet-5") == 0.5


def test_trial_backstop_sits_exactly_on_the_allowance():
    governor = _governor(TurnBudget(usd=0.05, limiting="trial", hard_cap=True))
    assert governor.cli_budget_usd("anthropic/claude-sonnet-5") == pytest.approx(0.05)


def test_backstop_is_scaled_into_cli_units_for_non_anthropic_models():
    # The CLI counts Kimi at Sonnet prices; an unscaled backstop would stop
    # the turn at a fraction of its real budget.
    governor = _governor(TurnBudget(usd=2.0, limiting="daily"))
    assert cli_price_ratio(KIMI) > 4
    assert governor.cli_budget_usd(KIMI) == pytest.approx(2.3 * cli_price_ratio(KIMI))


def _deny_reason(result: dict) -> str | None:
    output = result.get("hookSpecificOutput", {})
    if output.get("permissionDecision") != "deny":
        return None
    return output.get("permissionDecisionReason")


@pytest.mark.asyncio
async def test_pre_tool_hook_refuses_every_tool_once_landing():
    governor = _governor(meter=_Meter(spent=1.0))
    hooks = create_security_hooks(
        user_id="u1", sdk_cwd="/tmp/w", governor=governor, allowed_tools=ALLOWED
    )
    pre = hooks["PreToolUse"][0].hooks[0]
    for tool in ("mcp__copilot__run_block", "Agent", "WebSearch"):
        result = await pre(
            {"tool_name": tool, "tool_input": {"prompt": "x"}},
            tool_use_id=f"tu-{tool}",
            context={},
        )
        reason = _deny_reason(dict(result))
        assert reason is not None and "Where I got to:" in reason


@pytest.mark.asyncio
async def test_pre_tool_hook_lets_tools_run_below_the_limits():
    governor = _governor(meter=_Meter(spent=0.1, last=0.01))
    hooks = create_security_hooks(
        user_id="u1", sdk_cwd="/tmp/w", governor=governor, allowed_tools=ALLOWED
    )
    pre = hooks["PreToolUse"][0].hooks[0]
    result = await pre(
        {"tool_name": "mcp__copilot__run_block", "tool_input": {}},
        tool_use_id="tu-1",
        context={},
    )
    assert _deny_reason(dict(result)) is None
    unlisted = await pre(
        {"tool_name": "TaskCreate", "tool_input": {}}, tool_use_id="tu-2", context={}
    )
    assert "not available in this session" in (_deny_reason(dict(unlisted)) or "")


@pytest.mark.asyncio
async def test_the_turns_hook_carries_its_governor_and_allowed_tools(
    tmp_path, monkeypatch
):
    # Every other turn-level test mocks create_security_hooks, so a call site
    # that drops either argument stays green there.
    turn = _Turn(tmp_path, monkeypatch)
    await turn.run([_reply("gen-ok")])

    kwargs = turn._mocks[f"{_SVC}.create_security_hooks"].call_args.kwargs
    assert isinstance(kwargs["governor"], TurnGovernor)
    assert kwargs["allowed_tools"] == turn._mocks[f"{_SVC}.get_copilot_tool_names"]()
    assert (
        kwargs["builtin_tool_spans"]
        == turn._mocks[f"{_SVC}.config"].posthog_builtin_tool_spans
    )


def test_a_trial_limited_by_the_task_cap_gets_no_headroom():
    # The task cap binds, but the money is the trial's: the backstop sits
    # exactly on the budget, with no headroom and no CLI floor.
    governor = _governor(TurnBudget(usd=10.0, limiting="task", hard_cap=True))
    assert governor.cli_budget_usd("anthropic/claude-sonnet-5") == pytest.approx(10.0)
    small = _governor(TurnBudget(usd=0.1, limiting="daily", hard_cap=True))
    assert small.cli_budget_usd("anthropic/claude-sonnet-5") == pytest.approx(0.1)


def test_a_threshold_crossed_by_the_final_answer_finishes_the_task():
    # The answer that crosses the budget calls no tool after it: nothing was
    # refused, so the task finished rather than paused.
    governor = _governor(meter=_Meter(spent=1.0))
    governor.observe(AssistantMessage(content=[TextBlock(text="All done")], model="m"))
    assert governor.landing == "daily"
    assert governor.pause("success", wrote_checkpoint=True) is None
    assert governor.hangup_pause() is None
    assert governor.landed is False


def test_a_refused_tool_makes_the_turn_a_landing_one():
    governor = _governor(meter=_Meter(spent=1.0))
    assert governor.refusal() is not None
    assert governor.landed is True


def _stream(event: dict, parent: str | None = None) -> StreamEvent:
    return StreamEvent(uuid="u", session_id="s", event=event, parent_tool_use_id=parent)


def test_the_real_output_count_from_the_stream_lands_the_turn():
    # The assistant message carries only the placeholder output count; the
    # response's message_delta carries the real one, and that is what lands.
    governor = TurnGovernor(
        budget=TurnBudget(usd=0.1, limiting="daily", resets_at=RESETS), step_checkin=450
    )
    usage = {"input_tokens": 100, "output_tokens": 1}
    governor.observe(
        _stream(
            {
                "type": "message_start",
                "message": {"id": "m1", "model": "claude-sonnet-4-5", "usage": usage},
            }
        )
    )
    governor.observe(
        AssistantMessage(
            content=[TextBlock(text="x")],
            model="claude-sonnet-4-5",
            usage=usage,
            message_id="m1",
        )
    )
    assert governor.refusal() is None
    governor.observe(
        _stream({"type": "message_delta", "usage": {"output_tokens": 10_000}})
    )
    assert governor.refusal() is not None
    assert governor.landing == "daily"
