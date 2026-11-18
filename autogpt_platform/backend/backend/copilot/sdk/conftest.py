"""Shared test fixtures for copilot SDK tests."""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Iterator
from typing import Any, NoReturn
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio

from backend.copilot.sdk.cli_version import _VERSIONS, _cli_the_sdk_finds
from backend.data.db_manager import DatabaseManagerAsyncClient, DatabaseManagerClient
from backend.util import clients, json, service
from backend.util.cache import clear_thread_cache


def build_cli_cost_row(session_id: str, total: float) -> str:
    """Build a native CLI cost snapshot shared by retry and restart tests."""
    return (
        json.dumps(
            {"type": "cost-state", "sessionId": session_id, "totalCostUSD": total}
        )
        + "\n"
    )


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


@pytest.fixture(autouse=True)
def _fresh_cli_version_checks():
    """No test sees a CLI version or path another test read."""
    _VERSIONS.clear()
    _cli_the_sdk_finds.cache_clear()
    yield
    _VERSIONS.clear()
    _cli_the_sdk_finds.cache_clear()


@pytest_asyncio.fixture(scope="session", loop_scope="session", name="server")
async def _server_noop() -> None:
    """No-op server stub — SDK tests don't need the full backend."""
    return None


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


class DatabaseManagerFake:
    """The database manager for tests that start none.

    Every call fails at once with the error an unreachable database manager
    ends in, instead of after the real client's retries (seconds per call at
    best, close to an hour at the default retry count). What the code does
    with the failure is unchanged. ``calls`` lists the methods called.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def client(self, client_type: type) -> _FakeDatabaseManagerClient:
        return _FakeDatabaseManagerClient(self, client_type)


class _FakeDatabaseManagerClient:
    def __init__(self, fake: DatabaseManagerFake, client_type: type) -> None:
        self._fake = fake
        self._client_type = client_type

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._client_type, name, None)
        if method is None:
            raise AttributeError(f"Method {name} not found in {self._client_type}")

        def fail() -> NoReturn:
            self._fake.calls.append(name)
            _CALLS_THIS_SESSION[name] += 1
            raise httpx.ConnectError(
                f"No database manager runs in the copilot SDK tests ({name})"
            )

        if inspect.iscoroutinefunction(method):

            async def async_method(*args: Any, **kwargs: Any) -> Any:
                fail()

            return async_method

        def sync_method(*args: Any, **kwargs: Any) -> Any:
            fail()

        return sync_method


_CALLS_THIS_SESSION: Counter[str] = Counter()


@pytest.fixture(autouse=True)
def database_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[DatabaseManagerFake]:
    """Every test here gets the database manager fake.

    It stands in where the clients are built, so it also reaches modules that
    imported a client getter by name. A test that patches a getter or a
    ``db_accessors`` function itself still wins.
    """
    fake = DatabaseManagerFake()
    real_get_service_client = service.get_service_client

    def get_service_client(client_type: type, *args: Any, **kwargs: Any) -> Any:
        if client_type in (DatabaseManagerClient, DatabaseManagerAsyncClient):
            return fake.client(client_type)
        return real_get_service_client(client_type, *args, **kwargs)

    getters = (
        clients.get_database_manager_client,
        clients.get_database_manager_async_client,
    )
    for getter in getters:
        clear_thread_cache(getter)
    monkeypatch.setattr(service, "get_service_client", get_service_client)
    yield fake
    for getter in getters:
        clear_thread_cache(getter)


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """List the calls the fake answered, so a CI log shows which code paths
    still reach for a database manager instead of a stub of their own."""
    if not _CALLS_THIS_SESSION:
        return
    terminalreporter.write_sep("-", "database manager calls answered by the fake")
    for name, count in _CALLS_THIS_SESSION.most_common():
        terminalreporter.write_line(f"{count:6d}  {name}")


@pytest.fixture(autouse=True)
def login_chain_unchanged(request):
    """Sandbox doubles hold no login files; tests of that check opt out."""
    if request.node.get_closest_marker("real_login_chain"):
        yield
        return
    unchanged = AsyncMock(return_value={})
    with (
        patch("backend.util.sandbox_login.changed_login_files", unchanged),
    ):
        yield
