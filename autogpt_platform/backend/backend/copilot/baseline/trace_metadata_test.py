import hashlib
from unittest.mock import MagicMock

import pytest
from langfuse import propagate_attributes

from backend.copilot import trace_metadata
from backend.copilot.baseline import service
from backend.copilot.billing_e2e.baseline_turns import BaselineTurns
from backend.copilot.billing_e2e.ledger import PlatformLedger
from backend.copilot.billing_e2e.provider import FakeProvider
from backend.copilot.billing_e2e.replies import GREETING
from backend.copilot.model import ChatMessage, ChatSession


@pytest.mark.asyncio
async def test_baseline_turn_propagates_release_and_prompt_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trace_metadata.settings.config, "sentry_release", " build-b\n")
    attributes = MagicMock(wraps=propagate_attributes)
    monkeypatch.setattr(service, "propagate_attributes", attributes)
    ledger = PlatformLedger()
    ledger.install(monkeypatch)
    session = ChatSession.new("trace-test-user", dry_run=False)
    session.title = "Trace test"
    session.messages.append(ChatMessage(role="user", content="Hello"))
    config = service.config.model_copy(
        update={"use_e2b_sandbox": False, "use_local": True}
    )

    async with FakeProvider() as provider:
        run_turn = BaselineTurns(
            monkeypatch,
            config,
            base_url=f"{provider.url}/ollama/v1",
            model="test-model",
        )
        await run_turn(session, "Hello", new_pod=False)
        await ledger.settle()
        assert provider.calls
        assert not provider.unexpected

    assert any(
        m.role == "assistant" and m.content == GREETING for m in session.messages
    )

    assert attributes.call_args.kwargs["metadata"] == {
        "release": "build-b",
        "base_prompt_sha256": hashlib.sha256(
            b"You are a test assistant. Reply briefly."
        ).hexdigest(),
    }
    assert attributes.call_args.kwargs["session_id"] == session.session_id
