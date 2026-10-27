"""A turn on a pod whose CLI warm-up is still under way.

The executor warms the CLI binary in a background thread when it starts
(``cli_warmup``). A turn must never wait for it: it launches the CLI and is
charged what the provider billed whether the warm-up is running alongside it
or stuck reading the binary. Nor may a worker thread or a turn warm the CLI
themselves, as each thread's first turn used to.

They run the bundled Claude Code CLI, so they skip where there is none (the
Windows wheel ships without one); CI runs them on Linux.
"""

import logging
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import rate_limit, usage_activation
from backend.copilot.executor import cli_warmup, processor
from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.model import ChatMessage
from backend.copilot.rate_limit import SubscriptionTier
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.cli_openrouter_compat_test import _resolve_cli_path
from backend.copilot.turn_budget import TurnBudget
from backend.data import db_accessors

from .audit import audit_thread
from .executor_turns import Executor, SdkTurnsViaExecutor
from .ledger import PlatformLedger
from .provider import FakeProvider
from .scenarios import SCENARIOS
from .thread_billing_test import PROMPTS, _config, _session
from .trial_account import usage_state_of

[SCENARIO] = [s for s in SCENARIOS if s.id == "sdk-openrouter"]
# How long the stuck warm-up holds; a turn that waited for it would end after.
STUCK_SECONDS = 30
# Starting the warm-up, or a worker thread, returns well within this.
PROMPT_SECONDS = 5


@pytest.fixture(autouse=True)
def _paying_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user on a paid plan: no trial, and the budget is not under test."""
    if _resolve_cli_path() is None:
        pytest.skip("No Claude Code CLI binary available")
    store = MagicMock(get_subscription_trial=AsyncMock(return_value=None))
    monkeypatch.setattr(db_accessors, "credit_db", lambda: store)
    usage = MagicMock(
        get_usage_activation_state=AsyncMock(
            side_effect=lambda user_id: usage_state_of(
                user_id, SubscriptionTier.PRO, None
            )
        )
    )
    monkeypatch.setattr(usage_activation, "pro_activation_db", lambda: usage)
    monkeypatch.setattr(
        rate_limit,
        "get_redis_async",
        AsyncMock(return_value=MagicMock(get=AsyncMock(return_value=None))),
    )
    monkeypatch.setattr(sdk_service, "get_user_tier", AsyncMock(return_value=None))
    monkeypatch.setattr(
        sdk_service,
        "_fetch_graphiti_context",
        AsyncMock(return_value=(False, "")),
    )
    monkeypatch.setattr(sdk_service, "_enqueue_graphiti_turn", AsyncMock())
    monkeypatch.setattr(sdk_service, "capture_chat_turn", AsyncMock())
    monkeypatch.setattr(
        sdk_service,
        "resolve_turn_budget",
        AsyncMock(return_value=TurnBudget(usd=100.0, limiting="task")),
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_a_turn_runs_alongside_the_cli_warm_up(
    provider: FakeProvider,
    ledger: PlatformLedger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=cli_warmup.__name__)
    warm_up = cli_warmup.start_cli_warmup()

    await _one_turn(provider, ledger, tmp_path, monkeypatch)

    warm_up.join(60)
    assert not warm_up.is_alive()
    logged = [r.getMessage() for r in caplog.records if r.name == cli_warmup.__name__]
    assert logged[0] == "CLI warm-up started"
    assert "ran -v (rc=0)" in logged[-1]


@pytest.mark.asyncio(loop_scope="function")
async def test_a_turn_does_not_wait_for_a_stuck_cli_warm_up(
    provider: FakeProvider,
    ledger: PlatformLedger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reading = threading.Event()
    release = threading.Event()
    # Every thread that warmed the CLI: read the binary through, or ran -v.
    warmed_on: list[str] = []

    def stuck_read(_path: str) -> int:
        warmed_on.append(threading.current_thread().name)
        reading.set()
        release.wait(STUCK_SECONDS)
        return 0

    monkeypatch.setattr(cli_warmup, "_read_through", stuck_read)
    _record_version_runs(monkeypatch, warmed_on)
    started = time.monotonic()
    warm_up = cli_warmup.start_cli_warmup()
    try:
        assert time.monotonic() - started < PROMPT_SECONDS
        assert reading.wait(10)
        _start_a_worker_thread(monkeypatch)
        await _one_turn(provider, ledger, tmp_path, monkeypatch)
        assert warm_up.is_alive()
        # However fast a warm binary runs, the worker and the turn left the
        # warming to the warm-up's own thread.
        assert warmed_on == [warm_up.name]
    finally:
        release.set()
        warm_up.join(60)


def _record_version_runs(monkeypatch: pytest.MonkeyPatch, warmed_on: list[str]) -> None:
    """Note the thread of every ``<cli> -v`` run, the CLI's warm-up call."""
    run = subprocess.run

    def recording_run(args, *rest, **kwargs):
        if list(args)[1:] == ["-v"]:
            warmed_on.append(threading.current_thread().name)
        return run(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)


def _start_a_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a worker thread runs before its first turn; it must not wait either."""
    # Process-wide logging and service-name setup, not what is under test.
    monkeypatch.setattr(processor, "configure_logging", lambda: None)
    monkeypatch.setattr(processor, "set_service_name", lambda _name: None)
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    worker = CoPilotProcessor()
    started = time.monotonic()
    worker.on_executor_start()
    try:
        assert time.monotonic() - started < PROMPT_SECONDS
    finally:
        worker.execution_loop.call_soon_threadsafe(worker.execution_loop.stop)


async def _one_turn(
    provider: FakeProvider,
    ledger: PlatformLedger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One chat turn from the executor, checked against the provider's invoice."""
    session = _session(SCENARIO)
    config = _config(sdk_service.config, SCENARIO, provider)
    executor = Executor(monkeypatch, config, session, use_sdk=True)
    run_turn = SdkTurnsViaExecutor(tmp_path, monkeypatch, provider, config, executor)
    session.messages.append(ChatMessage(role="user", content=PROMPTS[0]))

    await run_turn(session, PROMPTS[0])
    await ledger.settle()

    assert executor.errors == [None]
    resumed = [
        (turn, launch == session.session_id) for turn, launch in run_turn.resumed
    ]
    findings = audit_thread(
        SCENARIO, provider, ledger, session, [session.usage], resumed, run_turn.metered
    )
    assert not findings, "\n".join(f.message for f in findings)
