"""An empty platform OpenRouter account must not reach the chat as raw text.

OpenRouter answers with a 402 that tells the reader to buy credits on
openrouter.ai. On the platform route that bill is ours, so the user sees one
plain sentence, the turn is retryable, and nothing offers to "switch
connection" as if it were their own limit.
"""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from backend.copilot.baseline import service
from backend.copilot.context import set_execution_context
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.model_router import ResolvedModel
from backend.copilot.response_model import StreamError, StreamProviderFailure

_OPENROUTER_402_BODY = {
    "message": "Insufficient credits. This account never purchased credits. "
    "Make sure your key is on the correct account or org, and if so, purchase "
    "more at https://openrouter.ai/settings/credits",
    "code": 402,
}


def _openrouter_402() -> openai.APIStatusError:
    response = httpx.Response(
        402, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat")
    )
    return openai.APIStatusError(
        f"Error code: 402 - {{'error': {_OPENROUTER_402_BODY}}}",
        response=response,
        body=_OPENROUTER_402_BODY,
    )


@pytest.fixture
def baseline_io(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[ChatSession]]:
    persisted: list[ChatSession] = []

    async def save(session: ChatSession) -> ChatSession:
        persisted.append(session.model_copy(deep=True))
        return session

    monkeypatch.setattr(
        service,
        "config",
        service.config.model_copy(
            update={"use_e2b_sandbox": False, "use_local": False}
        ),
    )
    for name, value in {
        "drain_pending_safe": [],
        "resolve_model_route": ResolvedModel(
            model="anthropic/claude-sonnet-4-6", source="env"
        ),
        "_build_system_prompt": ("System prompt", None),
        "is_enabled_for_user": False,
        "is_feature_enabled": False,
        "persist_and_record_usage": None,
    }.items():
        monkeypatch.setattr(service, name, AsyncMock(return_value=value))
    monkeypatch.setattr(service, "_get_main_client", MagicMock())
    monkeypatch.setattr(service, "upsert_chat_session", AsyncMock(side_effect=save))
    monkeypatch.setattr(
        service, "call_provider_stream", AsyncMock(side_effect=_openrouter_402())
    )
    try:
        yield persisted
    finally:
        set_execution_context(None, None)


async def _run_turn() -> tuple[list[object], ChatSession]:
    session = ChatSession.new("user-1", dry_run=False)
    session.messages.append(ChatMessage(role="user", content="hello"))
    events = [
        event
        async for event in service.stream_chat_completion_baseline(
            session.session_id,
            user_id="user-1",
            session=session,
            is_user_message=False,
        )
    ]
    return events, session


@pytest.mark.asyncio
async def test_platform_402_shows_the_platform_message(
    baseline_io: list[ChatSession],
) -> None:
    events, session = await _run_turn()

    [error] = [e for e in events if isinstance(e, StreamError)]
    assert error.code == "provider_unavailable"
    assert "temporarily unavailable" in error.errorText
    assert "openrouter" not in error.errorText.lower()
    # Not the user's limit, so no "switch connection" envelope.
    assert not [e for e in events if isinstance(e, StreamProviderFailure)]

    # The persisted marker is what a reload renders.
    marker = session.messages[-1].content or ""
    assert "temporarily unavailable" in marker
    assert "openrouter" not in marker.lower()
