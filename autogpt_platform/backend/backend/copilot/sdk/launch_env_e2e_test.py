"""Every CLI process a copilot turn starts has the backend's own settings
blanked, however the turn comes to start it.

``stream_chat_completion_sdk`` drives the bundled Claude Code CLI against the
billing suite's fake provider (see ``billing_e2e/sdk_turns.py`` for what is
stubbed), and every process the SDK spawns is recorded as it starts. The
turns reaching the provider show the CLI kept what it needs.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import anyio
import pytest

from backend.copilot.billing_e2e.ledger import PlatformLedger
from backend.copilot.billing_e2e.sdk_turns import SdkTurns

from . import service as sdk_service
from .chat_text_e2e_test import (
    _config,
    _RecordingProvider,
    _require_cli,
    _run_turn,
    _session,
)

# A setting of the backend's own, which no CLI process may keep.
_SETTING = ("BACKEND_ONLY_SETTING", "FAKE-backend-setting")


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch) -> PlatformLedger:
    """Takes the spend, cost-log and credit writes each turn makes."""
    ledger = PlatformLedger()
    ledger.install(monkeypatch)
    return ledger


@pytest.fixture(autouse=True)
def _memory_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory is not under test here, and its ingest needs a database."""
    monkeypatch.setattr(
        sdk_service, "_fetch_graphiti_context", AsyncMock(return_value=(False, ""))
    )


# Its own event loop, like the billing suite.
@pytest.mark.asyncio(loop_scope="function")
async def test_every_cli_a_turn_starts_has_the_backends_settings_blanked(
    ledger: PlatformLedger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A fresh turn, a turn resumed on the same pod and one restored on a new
    pod."""
    _require_cli()
    monkeypatch.setenv(*_SETTING)
    # As the copilot executor does: the SDK's own version check runs on the
    # parent's environment, and only reports a version.
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    spawned: list[dict[str, str] | None] = []
    open_process = anyio.open_process

    async def recording_open_process(*args, **kwargs):
        spawned.append(kwargs.get("env"))
        return await open_process(*args, **kwargs)

    # The SDK's transport spawns through the anyio module.
    monkeypatch.setattr(anyio, "open_process", recording_open_process)
    async with _RecordingProvider() as provider:
        turns = SdkTurns(tmp_path, monkeypatch, provider, _config())
        session = _session()
        for turn, (prompt, new_pod) in enumerate(
            [("hello", False), ("and now?", False), ("still there?", True)]
        ):
            await _run_turn(
                turns, provider, ledger, session, turn, prompt, new_pod=new_pod
            )

    assert len(spawned) >= 3, "a turn started no CLI process"
    name, _value = _SETTING
    assert all(
        env is not None and env.get(name) == "" for env in spawned
    ), "a CLI process kept the backend's own setting"
    for turn in range(3):
        assert provider.bodies_in(turn), f"turn {turn} made no model request"
