"""A three-turn chat on every route, each turn charged what its provider billed.

Turn 1 answers once. Turn 2 asks for a tool, so the model is called twice.
Turn 3 runs on a pod that kept nothing on disk: the SDK engine restores its
CLI session from storage and resumes it, the path on which a resumed turn was
charged the whole chat's spend so far until #15342.

The provider keeps its own invoice; the ledger records what the platform
charged. Each turn must charge the user's spend counters and the turn tree
once, for the turn's own calls, and write one cost-log row whose tokens are
the provider's.

The SDK scenarios run the bundled Claude Code CLI, so they skip where there is
none (the Windows wheel ships without one); CI runs them on Linux.
"""

from pathlib import Path

import pytest

from backend.copilot.baseline import service as baseline_service
from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatMessage, ChatSession, Usage
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.cli_openrouter_compat_test import _resolve_cli_path

from .audit import audit_thread
from .baseline_turns import BaselineTurns
from .ledger import PlatformLedger
from .m365_turns import MICROSOFT_365_CREDENTIAL_ID, Microsoft365Turns
from .provider import FakeProvider
from .replies import TOOL_MARKER
from .scenarios import SCENARIOS, Scenario
from .sdk_turns import CODEX_CREDENTIAL_ID, SdkTurns

PROMPTS = ("Say hello.", f"Look something up for me. {TOOL_MARKER}", "Thanks!")
NEW_POD_TURN = 2


# Its own event loop: a background task another test left on the shared
# session loop must not write into this thread's ledger mid-run.
@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.id)
async def test_each_turn_is_charged_what_its_provider_billed(
    scenario: Scenario,
    provider: FakeProvider,
    ledger: PlatformLedger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if scenario.needs_cli and _resolve_cli_path() is None:
        pytest.skip("No Claude Code CLI binary available")
    provider.generation_lookups_fail = scenario.generation_lookups_fail
    provider.faults = [scenario.fault] if scenario.fault else []
    run_turn = _turn_runner(scenario, provider, tmp_path, monkeypatch)
    session = _session(scenario)
    usage_by_turn: list[list[Usage]] = []

    for turn, prompt in enumerate(PROMPTS):
        provider.turn = ledger.turn = turn
        session.messages.append(ChatMessage(role="user", content=prompt))
        usage_before = len(session.usage)
        await run_turn(session, prompt, new_pod=turn == NEW_POD_TURN)
        await ledger.settle()
        usage_by_turn.append(session.usage[usage_before:])

    cli_resumed = (
        [(turn, launch == session.session_id) for turn, launch in run_turn.resumed]
        if isinstance(run_turn, SdkTurns)
        else None
    )
    findings = audit_thread(
        scenario, provider, ledger, session, usage_by_turn, cli_resumed
    )
    gap = scenario.known_gap
    unexplained = [f.message for f in findings if gap is None or f.check != gap.check]
    assert not unexplained, "\n".join(unexplained)
    if gap is not None:
        assert any(
            f.check == gap.check for f in findings
        ), f"no longer reproduces, so drop this known gap: {gap.reason}"
        pytest.xfail(gap.reason)


def _turn_runner(
    scenario: Scenario,
    provider: FakeProvider,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SdkTurns | BaselineTurns | Microsoft365Turns:
    if scenario.engine == "microsoft_365":
        return Microsoft365Turns(monkeypatch, provider)
    if scenario.engine == "baseline":
        return BaselineTurns(
            monkeypatch,
            _config(baseline_service.config, scenario, provider),
            base_url=f"{provider.url}{scenario.client_path}",
            model=scenario.model,
        )
    return SdkTurns(
        tmp_path,
        monkeypatch,
        provider,
        _config(sdk_service.config, scenario, provider),
        codex=scenario.codex,
        cli_max_retries=scenario.cli_max_retries,
    )


def _config(base: ChatConfig, scenario: Scenario, provider: FakeProvider) -> ChatConfig:
    overrides = {
        name: value.format(provider=provider.url) if isinstance(value, str) else value
        for name, value in scenario.config.items()
    }
    # Pinned so the deployment's own settings cannot change the path a
    # scenario takes; each scenario overrides what it is about.
    return base.model_copy(
        update={
            "use_local": False,
            "use_claude_code_subscription": False,
            "use_e2b_sandbox": False,
            "claude_agent_fallback_model": None,
            "claude_agent_cli_path": None,
            "claude_agent_use_resume": True,
            "claude_agent_max_transient_retries": 3,
            "sdk_reconcile_openrouter_cost": True,
            **overrides,
        }
    )


def _session(scenario: Scenario) -> ChatSession:
    if scenario.codex:
        session = ChatSession.new(
            "billing-e2e-user",
            dry_run=False,
            llm_auth_provider="codex",
            llm_credential_id=CODEX_CREDENTIAL_ID,
        )
    elif scenario.engine == "microsoft_365":
        session = ChatSession.new(
            "billing-e2e-user",
            dry_run=False,
            llm_auth_provider="microsoft_365_copilot",
            llm_credential_id=MICROSOFT_365_CREDENTIAL_ID,
        )
    else:
        session = ChatSession.new("billing-e2e-user", dry_run=False)
    # A titled chat skips title generation, a separate call this suite does
    # not account for.
    session.title = "Billing e2e"
    return session
