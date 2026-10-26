"""A delegated thread's turn stops at its hand-off cap, on both engines.

The thread Otto opened has a cap smaller than one model call, and its task
takes three tool rounds before the answer. The turn runs from the executor
through the real engine against the fake provider, which bills each call.

The engine must end the turn at the cap instead of running on: the baseline
engine finishes the round under way and starts no other, and the SDK
engine's governor refuses every tool call once it has seen the cap spent
(it meters a call after the call, so the first tool can slip through).
The turn ends as a pause at the cap, recorded in the thread, and the
executor holds the thread there for the user, the raise-or-stop question
parked, before it marks the turn done. Everything that reads the thread
afterwards, the waiting parent included, reads that hold rather than a
finished task.
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from backend.copilot import delegation_cap, stream_registry, turn_budget
from backend.copilot.baseline import service as baseline_service
from backend.copilot.delegation_settings import DelegationSettings
from backend.copilot.gate import policy as gate_policy
from backend.copilot.gate.policy import Effect
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.cli_openrouter_compat_test import _resolve_cli_path
from backend.copilot.sdk.tool_adapter import MCP_TOOL_PREFIX
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import NoResultsResponse, ToolResponseBase
from backend.copilot.turn_pause import PAUSE_KEY

from .baseline_turns import BaselineTurns
from .executor_turns import Executor, SdkTurnsViaExecutor
from .ledger import PlatformLedger
from .provider import FakeProvider
from .replies import AFTER_TOOL, TOOL_MARKER
from .scenarios import SCENARIOS, Scenario
from .thread_billing_test import _config, _session
from .trial_account import TrialAccount
from .trial_thread_test import _turn_runner
from .wire import TOOL_NAME

CAP_SCENARIOS = [
    s for s in SCENARIOS if s.id in ("sdk-openrouter", "baseline-openrouter")
]
# Less than any one call the fake provider bills.
CAP_USD = 0.000001
PROMPT = f"Pull the Q3 numbers. {TOOL_MARKER}"
TOOL_ROUNDS = 3


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("scenario", CAP_SCENARIOS, ids=lambda scenario: scenario.id)
async def test_a_delegated_turn_stops_at_its_cap_and_waits_on_the_user(
    scenario: Scenario,
    provider: FakeProvider,
    ledger: PlatformLedger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if scenario.needs_cli and _resolve_cli_path() is None:
        pytest.skip("No Claude Code CLI binary available")
    session = _session(scenario)
    session.metadata.delegated_by_session_id = "otto-chat"
    session.metadata.delegation_cap_usd = CAP_USD
    account = TrialAccount(session.user_id, spent_before=0)
    account.install(monkeypatch, ledger)
    account.convert_to_paid()
    lookup = _Lookup()
    if scenario.engine == "baseline":
        run_turn, executor = _turn_runner(
            scenario, provider, session, tmp_path, monkeypatch, tool=lookup
        )
    else:
        run_turn, executor = _governed_sdk(
            scenario, provider, session, tmp_path, monkeypatch, lookup
        )

    # The thread's earlier turns: none, so the whole cap is left.
    costs = MagicMock(get_session_costs=AsyncMock(return_value={}))
    monkeypatch.setattr(turn_budget, "delegation_db", lambda: costs)
    # What the executor writes to the thread's row when it holds it.
    order: list[str] = []
    row = MagicMock(
        set_session_pending_question=AsyncMock(
            side_effect=lambda *_a, **_k: order.append("parked")
        )
    )
    settings = MagicMock(
        get_delegation_settings=AsyncMock(return_value=DelegationSettings()),
        stop_delegation_at_cap=AsyncMock(),
    )
    monkeypatch.setattr(delegation_cap, "chat_db", lambda: row)
    monkeypatch.setattr(delegation_cap, "delegation_db", lambda: settings)
    dropped = AsyncMock(side_effect=lambda *_a: order.append("cache dropped"))
    monkeypatch.setattr(delegation_cap, "invalidate_session_cache", dropped)
    completed = stream_registry.mark_session_completed
    assert isinstance(completed, AsyncMock)
    marked_done = completed.side_effect

    async def done(*args, **kwargs):
        order.append("done")
        return await marked_done(*args, **kwargs)

    completed.side_effect = done

    provider.tool_rounds = TOOL_ROUNDS
    provider.turn = ledger.turn = account.turn = 0
    session.messages.append(ChatMessage(role="user", content=PROMPT))
    await run_turn(session, PROMPT)
    await ledger.settle()

    assert executor.errors == [None]
    if scenario.engine == "baseline":
        # The round under way finishes; none starts after the cap, so the
        # answer the second call would have written never came.
        assert lookup.runs == 1
        assert len(provider.calls_in(0)) == 1
        assert not any(
            m.role == "assistant" and AFTER_TOOL in (m.content or "")
            for m in session.messages
        )
    else:
        # Every tool call after the governor saw the cap spent was refused.
        assert lookup.runs <= 1
    # The thread ends on the pause, and the pause names the cap.
    last = session.messages[-1]
    assert last.metadata and last.metadata[PAUSE_KEY]["cap_usd"] == CAP_USD
    # Held for the user before the turn was marked done, which is what the
    # waiting parent wakes on.
    assert order == ["parked", "cache dropped", "done"]
    session_id, user_id, question = row.set_session_pending_question.await_args.args[:3]
    assert (session_id, user_id) == (session.session_id, session.user_id)
    assert question == delegation_cap.cap_question(CAP_USD)
    settings.stop_delegation_at_cap.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="function")
async def test_a_cap_that_cannot_be_read_never_leaves_a_counted_turn(
    provider: FakeProvider, ledger: PlatformLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn on the baseline engine is counted on its box, and only the
    turn's try/finally releases that count. Reading the hand-off's cap can
    fail (the cost store is down); it must fail before the turn is counted,
    or the expert's box stays unpaused at every later turn end."""
    scenario = next(s for s in SCENARIOS if s.id == "baseline-openrouter")
    config = _config(baseline_service.config, scenario, provider).model_copy(
        update={"use_e2b_sandbox": True, "e2b_api_key": "e2b-test"}
    )
    run_turn = BaselineTurns(
        monkeypatch,
        config,
        base_url=f"{provider.url}{scenario.client_path}",
        model=scenario.model,
    )
    counted = AsyncMock()
    released = MagicMock()
    for name, value in {
        "get_or_create_sandbox": AsyncMock(return_value=MagicMock()),
        "count_expert_turn": counted,
        "_pause_counted_box": released,
        "_pause_uncounted_box": MagicMock(),
        "cap_guard_for": AsyncMock(side_effect=RuntimeError("cost store down")),
    }.items():
        monkeypatch.setattr(baseline_service, name, value)
    # Not an expert's session: the harness keeps the expert's DB lookups out,
    # and the box is counted for any session that has one.
    session = _session(scenario)
    session.metadata.delegated_by_session_id = "otto-chat"
    session.metadata.delegation_cap_usd = CAP_USD
    session.messages.append(ChatMessage(role="user", content=PROMPT))

    with pytest.raises(RuntimeError, match="cost store down"):
        await run_turn(session, PROMPT, new_pod=False)

    assert counted.await_count == released.call_count


class _Lookup(BaseTool):
    """The task's one tool, counting the times it ran."""

    def __init__(self) -> None:
        self.runs = 0

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
        self.runs += 1
        return NoResultsResponse(
            message="The fact is 42.", session_id=session.session_id
        )


class _GovernedSdkTurns(SdkTurnsViaExecutor):
    """SDK turns with the copilot's own tool hooks left in, since the turn
    governor refuses a tool from its pre-tool hook. The rest of the billing
    harness leaves them out because its scenarios are not about tools."""

    def _stubs(self, session: ChatSession) -> dict[str, Any]:
        stubs = super()._stubs(session)
        del stubs["create_security_hooks"]
        return stubs

    def _client(
        self, *args: Any, options: ClaudeAgentOptions, **kwargs: Any
    ) -> ClaudeSDKClient:
        self.budgets.append((self._turn, options.max_budget_usd))
        self.resumed.append((self._turn, options.resume))
        options.allowed_tools = [f"{MCP_TOOL_PREFIX}{name}" for name in self._tools]
        options.disallowed_tools = []
        return ClaudeSDKClient(*args, options=options, **kwargs)


def _governed_sdk(
    scenario: Scenario,
    provider: FakeProvider,
    session: ChatSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: BaseTool,
) -> tuple[_GovernedSdkTurns, Executor]:
    # The tool only reads, so the approval gate runs it in every mode.
    monkeypatch.setitem(gate_policy._EFFECTS, TOOL_NAME, Effect.READ)
    config = _config(sdk_service.config, scenario, provider)
    executor = Executor(monkeypatch, config, session, use_sdk=True)
    provider.tool_name = f"{MCP_TOOL_PREFIX}{TOOL_NAME}"
    runner = _GovernedSdkTurns(
        tmp_path,
        monkeypatch,
        provider,
        config,
        executor,
        served_tools={TOOL_NAME: tool},
    )
    return runner, executor
