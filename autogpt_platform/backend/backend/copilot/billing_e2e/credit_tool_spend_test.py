"""A credit-billed turn whose tools pay is stopped at what it holds.

The turn holds $1 of a $2 balance on a ledger with nothing more free, and its
model calls a tool on every reply that records 3 cents, as the platform's paid
tools do. The governor counts what the tools recorded against what the model
may still spend, so the stream loop interrupts the CLI once the two together
reach the hold.

Runs the bundled Claude Code CLI against the fake provider, so it skips where
there is none.
"""

from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server, tool

from backend.copilot import credit_billing as billing
from backend.copilot.model import ChatMessage
from backend.copilot.sdk.cli_openrouter_compat_test import _resolve_cli_path

from . import provider as provider_module
from .ledger import CreditBilling, PlatformLedger
from .provider import FakeProvider
from .replies import TOOL_MARKER
from .scenarios import SCENARIOS
from .sdk_turns import SdkTurns
from .thread_billing_test import _session, _turn_runner
from .wire import Reply

_TOOL_COST_MICRODOLLARS = 30_000
_HELD_MICRODOLLARS = 1_000_000
# The reading the stop lands on can pass the hold by a model call and the tool
# that call ran.
_LAST_READING_MICRODOLLARS = 100_000


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize(
    "scenario_id", ["sdk-anthropic-credits", "sdk-openrouter-credits"]
)
async def test_a_turn_whose_tools_pay_is_stopped_at_what_it_holds(
    scenario_id: str,
    provider: FakeProvider,
    ledger: PlatformLedger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _resolve_cli_path() is None:
        pytest.skip("No Claude Code CLI binary available")
    scenario = next(s for s in SCENARIOS if s.id == scenario_id)
    session = _session(scenario)
    recorded: list[int] = []

    @tool("lookup_fact", "Look a fact up.", {"query": str})
    async def lookup_fact(args: dict[str, Any]) -> dict[str, Any]:
        if refusal := await billing.tool_spend_refusal(session.user_id):
            return {"content": [{"type": "text", "text": refusal}]}
        await billing.record_credit_billed_cost(
            session.user_id, _TOOL_COST_MICRODOLLARS
        )
        recorded.append(_TOOL_COST_MICRODOLLARS)
        return {"content": [{"type": "text", "text": "a fact"}]}

    _call_the_tool_every_reply(
        monkeypatch, provider, create_sdk_mcp_server("paid", tools=[lookup_fact])
    )
    monkeypatch.setattr(billing, "LedgerHolds", _OneDollarFree)
    ledger.fund(CreditBilling(balance=200))
    run_turn = _turn_runner(scenario, provider, tmp_path, monkeypatch)
    prompt = f"Go. {TOOL_MARKER}"
    session.messages.append(ChatMessage(role="user", content=prompt))
    provider.turn = ledger.turn = 0

    await run_turn(session, prompt, new_pod=False)
    await ledger.settle()

    model = sum(row.cost_microdollars or 0 for row in ledger.rows_in(0))
    assert recorded, "the model never ran the paid tool"
    assert model + sum(recorded) <= _HELD_MICRODOLLARS + _LAST_READING_MICRODOLLARS


def _call_the_tool_every_reply(
    monkeypatch: pytest.MonkeyPatch,
    provider: FakeProvider,
    server: McpSdkServerConfig,
) -> None:
    monkeypatch.setattr(
        provider_module, "anthropic_reply", lambda _: Reply(kind="tool")
    )
    monkeypatch.setattr(provider_module, "chat_reply", lambda _: Reply(kind="tool"))
    provider.tool_name = "mcp__paid__lookup_fact"
    real_client = SdkTurns._client

    def client(self: SdkTurns, *args: Any, options: Any, **kwargs: Any) -> Any:
        made = real_client(self, *args, options=options, **kwargs)
        made.options.mcp_servers = {"paid": server}
        made.options.allowed_tools = ["mcp__paid__lookup_fact"]
        return made

    monkeypatch.setattr(SdkTurns, "_client", client)


class _OneDollarFree(billing.LedgerHolds):
    """A ledger off Redis with $1 free for the turn's first hold and nothing
    after it, so the turn can't grow."""

    def __init__(self, ledger_id: str, lifetime_seconds: int) -> None:
        super().__init__(ledger_id, lifetime_seconds)
        self.free = 100

    async def hold(
        self, turn_id: str, *, read_balance: Any, credits: int, at_least: int = 1
    ) -> int:
        await read_balance()
        added = min(self.free, credits)
        if added < at_least:
            return 0
        self.free -= added
        return added

    async def lower(self, turn_id: str, credits: int) -> None:
        pass

    async def release(self, turn_id: str) -> None:
        pass
