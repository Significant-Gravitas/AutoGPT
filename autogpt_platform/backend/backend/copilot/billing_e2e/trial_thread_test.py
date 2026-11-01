"""A trial user's chat, turn by turn, from the executor through the real engine.

Turn 1 answers once and turn 2 runs a tool round, both on an active trial
that already spent most of its allowance on earlier days. The trial then
converts to a paid plan, and turn 3 runs on that.

Each trial turn must be charged to the trial, as much as the provider billed
for it, and the next turn's budget must come from what the trial has left
after it. Once the trial has converted, nothing more may be charged to it.
A tool the model calls must see the trial as it is when the tool runs, not as
it was when the turn started, on either engine's own tool path. A turn the
CLI's budget backstop cuts off is still charged what its call cost.
"""

from pathlib import Path
from typing import Any

import pytest

from backend.copilot import rate_limit
from backend.copilot.baseline import service as baseline_service
from backend.copilot.gate import policy as gate_policy
from backend.copilot.gate.policy import Effect
from backend.copilot.model import ChatMessage, ChatSession, Usage
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.cli_openrouter_compat_test import _resolve_cli_path
from backend.copilot.sdk.governor import TurnGovernor
from backend.copilot.sdk.tool_adapter import MCP_TOOL_PREFIX
from backend.copilot.sdk.trial_budget import resolve_trial_sdk_budget
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import NoResultsResponse, ToolResponseBase
from backend.copilot.turn_budget import resolve_turn_budget
from backend.data.platform_cost import usd_to_microdollars
from backend.data.subscription_trial import TrialState

from .audit import audit_thread
from .executor_turns import BaselineTurnsViaExecutor, Executor, SdkTurnsViaExecutor
from .ledger import PlatformLedger
from .provider import FakeProvider
from .scenarios import SCENARIOS, Scenario, expected_charge_usd
from .thread_billing_test import PROMPTS, _config, _session
from .trial_account import TOTAL_LIMIT, TRIAL_ID, TrialAccount
from .wire import TOOL_NAME

TRIAL_SCENARIOS = [
    s for s in SCENARIOS if s.id in ("sdk-openrouter", "baseline-openrouter")
]
CONVERTED_TURN = 2
# Spent on earlier days, so the trial's total is what limits the turns.
SPENT_BEFORE = 700_000
# The OpenRouter routes with and without the reconcile; production charges
# without it.
OVERSHOOT_SCENARIOS = [
    s for s in SCENARIOS if s.id in ("sdk-openrouter", "sdk-openrouter-no-reconcile")
]
# Less than the turn's first call costs (about $0.02).
LEFT_BEFORE_OVERSHOOT = 5_000


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("scenario", TRIAL_SCENARIOS, ids=lambda scenario: scenario.id)
async def test_a_trial_users_turns_are_spent_from_the_trial(
    scenario: Scenario,
    provider: FakeProvider,
    ledger: PlatformLedger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if scenario.needs_cli and _resolve_cli_path() is None:
        pytest.skip("No Claude Code CLI binary available")
    session = _session(scenario)
    account = TrialAccount(session.user_id, spent_before=SPENT_BEFORE)
    account.install(monkeypatch, ledger)
    run_turn, executor = _turn_runner(
        scenario, provider, session, tmp_path, monkeypatch
    )
    remaining_at_start: list[float] = []
    paid_budget_at_start: dict[int, float | None] = {}
    usage_by_turn: list[list[Usage]] = []

    for turn, prompt in enumerate(PROMPTS):
        provider.turn = ledger.turn = account.turn = turn
        if turn == CONVERTED_TURN:
            account.convert_to_paid()
        remaining_at_start.append(account.remaining_usd())
        if turn >= CONVERTED_TURN:
            # What the paid plan allows one query as this turn starts, in the
            # CLI's units: the turn's budget plus the backstop's headroom.
            paid_budget_at_start[turn] = TurnGovernor(
                budget=await resolve_turn_budget(session.user_id), step_checkin=1
            ).cli_budget_usd(scenario.model)
        session.messages.append(ChatMessage(role="user", content=prompt))
        usage_before = len(session.usage)
        await run_turn(session, prompt)
        await ledger.settle()
        usage_by_turn.append(session.usage[usage_before:])

    assert executor.errors == [None] * len(PROMPTS)
    cli_resumed = (
        [(turn, launch == session.session_id) for turn, launch in run_turn.resumed]
        if isinstance(run_turn, SdkTurnsViaExecutor)
        else None
    )
    metered = run_turn.metered if isinstance(run_turn, SdkTurnsViaExecutor) else None
    findings = audit_thread(
        scenario, provider, ledger, session, usage_by_turn, cli_resumed, metered
    )
    assert not findings, "\n".join(f.message for f in findings)

    for turn in range(CONVERTED_TURN):
        spent = sum(charge.microdollars for charge in ledger.spend_in(turn))
        charged = account.charged_in(turn)
        assert spent > 0
        assert [c.trial_id for c in charged] == [TRIAL_ID] * len(charged)
        assert sum(c.microdollars for c in charged) == spent
        assert [
            row.metadata["subscription_trial_id"] for row in ledger.rows_in(turn)
        ] == [TRIAL_ID]
    assert account.charged_in(CONVERTED_TURN) == []
    assert [
        row.metadata.get("subscription_trial_id")
        for row in ledger.rows_in(CONVERTED_TURN)
    ] == [None]
    trial_spend = sum(
        c.microdollars for t in range(CONVERTED_TURN) for c in ledger.spend_in(t)
    )
    assert account.trial.cost_microdollars == SPENT_BEFORE + trial_spend

    # Every spend counter the limiter wrote expires, so a window resets.
    assert account.counters.ttls.keys() == account.counters.values.keys()
    assert all(ttl > 0 for ttl in account.counters.ttls.values())

    if isinstance(run_turn, SdkTurnsViaExecutor):
        static_cap = sdk_service.config.claude_agent_max_budget_usd
        assert {turn for turn, _ in run_turn.budgets} == set(range(len(PROMPTS)))
        for turn, budget in run_turn.budgets:
            if turn < CONVERTED_TURN:
                assert budget == pytest.approx(
                    resolve_trial_sdk_budget(static_cap, remaining_at_start[turn])
                )
            else:
                # Sized by the plan the user now pays for, not the trial.
                assert budget == pytest.approx(paid_budget_at_start[turn])
                assert budget > remaining_at_start[0]
    else:
        # The baseline tells the model its budget once, on the chat's first
        # turn (``should_inject_user_context``).
        [first_turn] = run_turn.user_context.await_args_list
        assert first_turn.kwargs["budget_ctx"].startswith(
            f"Approximate remaining USD budget for this user: ${remaining_at_start[0]:.2f}."
        )


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("scenario", TRIAL_SCENARIOS, ids=lambda scenario: scenario.id)
async def test_a_trial_turn_reads_the_trial_once_before_the_model_answers(
    scenario: Scenario,
    provider: FakeProvider,
    ledger: PlatformLedger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allowance gate, the turn's cost attribution and the engine's budget
    all look the trial up before the model's first request, and they share one
    read. A trial this far spent leaves less than the task's cap, so the SDK
    budget also works out which window limits the turn, from the usage status."""
    if scenario.needs_cli and _resolve_cli_path() is None:
        pytest.skip("No Claude Code CLI binary available")
    session = _session(scenario)
    account = TrialAccount(session.user_id, spent_before=SPENT_BEFORE)
    account.install(monkeypatch, ledger)
    store = rate_limit.credit_db()
    read_trial = store.get_subscription_trial.side_effect
    # Each read's turn, and how many model requests that turn had made by then.
    reads: list[tuple[int, int]] = []

    async def read_noting_when(user_id: str) -> TrialState | None:
        reads.append((account.turn, len(provider.calls_in(account.turn))))
        return await read_trial(user_id)

    store.get_subscription_trial.side_effect = read_noting_when
    run_turn, executor = _turn_runner(
        scenario, provider, session, tmp_path, monkeypatch
    )

    for turn, prompt in enumerate(PROMPTS[:CONVERTED_TURN]):
        provider.turn = ledger.turn = account.turn = turn
        session.messages.append(ChatMessage(role="user", content=prompt))
        await run_turn(session, prompt)
        await ledger.settle()

    assert executor.errors == [None] * CONVERTED_TURN
    assert [turn for turn, requests in reads if requests == 0] == list(
        range(CONVERTED_TURN)
    )


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("scenario", TRIAL_SCENARIOS, ids=lambda scenario: scenario.id)
async def test_a_tool_sees_the_trial_as_it_is_when_the_tool_runs(
    scenario: Scenario,
    provider: FakeProvider,
    ledger: PlatformLedger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if scenario.needs_cli and _resolve_cli_path() is None:
        pytest.skip("No Claude Code CLI binary available")
    session = _session(scenario)
    account = TrialAccount(session.user_id, spent_before=SPENT_BEFORE)
    account.install(monkeypatch, ledger)
    tool = _PaywalledLookup(account)
    run_turn, executor = _turn_runner(
        scenario, provider, session, tmp_path, monkeypatch, tool=tool
    )
    prompt = PROMPTS[1]
    session.messages.append(ChatMessage(role="user", content=prompt))

    await run_turn(session, prompt)
    await ledger.settle()

    assert executor.errors == [None]
    assert tool.paywalled == [True]
    # The turn started on the trial, so it is still spent from it.
    spent = sum(charge.microdollars for charge in ledger.spend_in(0))
    assert spent > 0
    assert sum(c.microdollars for c in account.charged_in(0)) == spent


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize(
    "scenario", OVERSHOOT_SCENARIOS, ids=lambda scenario: scenario.id
)
async def test_a_turn_the_budget_backstop_cuts_off_is_still_charged(
    scenario: Scenario,
    provider: FakeProvider,
    ledger: PlatformLedger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trial with less left than one call costs: the CLI stops the turn after
    that call, the call is charged in full, and the next turn is refused before
    it reaches the model rather than run for nothing."""
    if _resolve_cli_path() is None:
        pytest.skip("No Claude Code CLI binary available")
    session = _session(scenario)
    account = TrialAccount(
        session.user_id, spent_before=TOTAL_LIMIT - LEFT_BEFORE_OVERSHOOT
    )
    account.install(monkeypatch, ledger)
    run_turn, executor = _turn_runner(
        scenario, provider, session, tmp_path, monkeypatch
    )
    # The tool round: the CLI would make a second call after the first.
    prompt = PROMPTS[1]

    for turn in range(2):
        provider.turn = ledger.turn = account.turn = turn
        session.messages.append(ChatMessage(role="user", content=prompt))
        await run_turn(session, prompt)
        await ledger.settle()

    [call] = provider.calls_in(0)
    full_cost = usd_to_microdollars(expected_charge_usd(scenario.charge, call))
    assert full_cost is not None and full_cost > LEFT_BEFORE_OVERSHOOT
    assert [c.microdollars for c in ledger.spend_in(0)] == [full_cost]
    assert [row.cost_microdollars for row in ledger.rows_in(0)] == [full_cost]
    assert account.trial.cost_microdollars == (
        TOTAL_LIMIT - LEFT_BEFORE_OVERSHOOT + full_cost
    )
    assert account.remaining_usd() == 0
    assert provider.calls_in(1) == []
    assert ledger.spend_in(1) == []
    assert executor.errors == [None, None]


class _PaywalledLookup(BaseTool):
    """A tool that checks the paywall, after the trial ended mid-turn."""

    def __init__(self, account: TrialAccount) -> None:
        self._account = account
        self.paywalled: list[bool] = []

    @property
    def name(self) -> str:
        return TOOL_NAME

    @property
    def description(self) -> str:
        return "Look a fact up."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        # The trial ends while the model is choosing this tool.
        self._account.end_trial()
        assert user_id is not None
        self.paywalled.append(await rate_limit.is_user_paywalled(user_id))
        return NoResultsResponse(
            message="The fact is 42.", session_id=session.session_id
        )


def _turn_runner(
    scenario: Scenario,
    provider: FakeProvider,
    session: ChatSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tool: BaseTool | None = None,
) -> tuple[SdkTurnsViaExecutor | BaselineTurnsViaExecutor, Executor]:
    """The scenario's engine behind the executor; with ``tool``, the model's
    tool calls run it through that engine's own tool path."""
    if tool is not None:
        # The approval gate classifies tools by name, and an unknown name is a
        # platform edit it may hold or judge. This tool only reads, like the
        # copilot's own lookups, so the gate runs it in every mode.
        monkeypatch.setitem(gate_policy._EFFECTS, TOOL_NAME, Effect.READ)
    if scenario.engine == "baseline":
        config = _config(baseline_service.config, scenario, provider)
        executor = Executor(monkeypatch, config, session, use_sdk=False)
        runner = BaselineTurnsViaExecutor(
            monkeypatch,
            config,
            executor,
            base_url=f"{provider.url}{scenario.client_path}",
            model=scenario.model,
        )
        if tool is not None:
            monkeypatch.setitem(TOOL_REGISTRY, TOOL_NAME, tool)
            runner.run_registry_tools(monkeypatch)
        return runner, executor
    config = _config(sdk_service.config, scenario, provider)
    executor = Executor(monkeypatch, config, session, use_sdk=True)
    served_tools = None
    if tool is not None:
        provider.tool_name = f"{MCP_TOOL_PREFIX}{TOOL_NAME}"
        served_tools = {TOOL_NAME: tool}
    runner = SdkTurnsViaExecutor(
        tmp_path, monkeypatch, provider, config, executor, served_tools=served_tools
    )
    return runner, executor
