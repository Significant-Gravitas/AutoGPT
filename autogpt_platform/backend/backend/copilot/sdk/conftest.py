"""Shared test fixtures for copilot SDK tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio

from backend.util import json

# ---------------------------------------------------------------------------
# Env vars that ``ChatConfig`` validators read — must be cleared so explicit
# constructor values are used.  Centralised here so adding a new env-backed
# field only needs one update across the SDK test suite.
# ---------------------------------------------------------------------------
_CONFIG_ENV_VARS = (
    "CHAT_USE_OPENROUTER",
    "CHAT_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "OPENAI_API_KEY",
    "CHAT_BASE_URL",
    "OPENROUTER_BASE_URL",
    "OPENAI_BASE_URL",
    "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION",
    "CHAT_USE_CLAUDE_AGENT_SDK",
    "CHAT_CLAUDE_AGENT_CROSS_USER_PROMPT_CACHE",
    "CHAT_CLAUDE_AGENT_CLI_PATH",
    "CLAUDE_AGENT_CLI_PATH",
    # Aux-client + title-model + direct-key vars: read by
    # ``_validate_aux_client_for_direct_main`` to decide whether
    # subscription / direct-Anthropic configs are safe.  Local ``.env``
    # files often set these; clearing them ensures the test's explicit
    # constructor kwargs (or lack thereof) drive the validator.
    "CHAT_AUX_API_KEY",
    "CHAT_AUX_BASE_URL",
    "CHAT_TITLE_MODEL",
    "CHAT_DIRECT_ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture()
def _clean_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear env-backed CHAT_* settings so ChatConfig uses constructor values."""
    for var in _CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest_asyncio.fixture(scope="session", loop_scope="session", name="server")
async def _server_noop() -> None:
    """No-op server stub — SDK tests don't need the full backend."""
    return None


@pytest_asyncio.fixture(
    scope="session", loop_scope="session", autouse=True, name="graph_cleanup"
)
async def _graph_cleanup_noop() -> AsyncIterator[None]:
    """No-op graph cleanup stub."""
    yield


@pytest.fixture()
def mock_chat_config():
    """Mock ChatConfig so compact_transcript tests skip real config lookup."""
    with patch(
        "backend.copilot.config.ChatConfig",
        return_value=type("Cfg", (), {"model": "m", "api_key": "k", "base_url": "u"})(),
    ):
        yield


def build_test_transcript(pairs: list[tuple[str, str]]) -> str:
    """Build a minimal valid JSONL transcript from (role, content) pairs.

    Use this helper in any copilot SDK test that needs a well-formed
    transcript without hitting the real storage layer.

    Delegates to ``build_structured_transcript`` — plain content strings
    are automatically wrapped in ``[{"type": "text", "text": ...}]`` for
    assistant messages.
    """
    # Cast widening: tuple[str, str] is structurally compatible with
    # tuple[str, str | list[dict]] but list invariance requires explicit
    # annotation.
    widened: list[tuple[str, str | list[dict]]] = list(pairs)
    return build_structured_transcript(widened)


def build_structured_transcript(
    entries: list[tuple[str, str | list[dict]]],
) -> str:
    """Build a JSONL transcript with structured content blocks.

    Each entry is (role, content) where content is either a plain string
    (for user messages) or a list of content block dicts (for assistant
    messages with thinking/tool_use/text blocks).

    Example::

        build_structured_transcript([
            ("user", "Hello"),
            ("assistant", [
                {"type": "thinking", "thinking": "...", "signature": "sig1"},
                {"type": "text", "text": "Hi there"},
            ]),
        ])
    """
    lines: list[str] = []
    last_uuid: str | None = None
    for role, content in entries:
        uid = str(uuid4())
        entry_type = "assistant" if role == "assistant" else "user"
        if role == "assistant" and isinstance(content, list):
            msg: dict = {
                "role": "assistant",
                "model": "claude-test",
                "id": f"msg_{uid[:8]}",
                "type": "message",
                "content": content,
                "stop_reason": "end_turn",
                "stop_sequence": None,
            }
        elif role == "assistant":
            msg = {
                "role": "assistant",
                "model": "claude-test",
                "id": f"msg_{uid[:8]}",
                "type": "message",
                "content": [{"type": "text", "text": content}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
            }
        else:
            msg = {"role": role, "content": content}
        entry = {
            "type": entry_type,
            "uuid": uid,
            "parentUuid": last_uuid,
            "message": msg,
        }
        lines.append(json.dumps(entry, separators=(",", ":")))
        last_uuid = uid
    return "\n".join(lines) + "\n"
