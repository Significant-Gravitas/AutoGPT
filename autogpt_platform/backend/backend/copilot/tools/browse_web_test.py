"""Unit tests for BrowseWebTool.

All tests run without a running server / database.  External dependencies
(Stagehand, Browserbase) are mocked via sys.modules injection so the suite
stays fast and deterministic.
"""

import sys
import threading
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.copilot.tools.browse_web as _browse_web_mod
from backend.copilot.model import ChatSession
from backend.copilot.tools.models import BrowseWebResponse, ErrorResponse, ResponseType

# Convenience aliases — all resolved from the same module object so that tests
# can reset module-level globals (e.g. _browse_web_mod._stagehand_patched = False)
# without going through a separate from-import reference copy.
BrowseWebTool = _browse_web_mod.BrowseWebTool
_MAX_CONTENT_CHARS = _browse_web_mod._MAX_CONTENT_CHARS
_patch_stagehand_once = _browse_web_mod._patch_stagehand_once

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_session(user_id: str = "test-user") -> ChatSession:
    return ChatSession(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        messages=[],
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        successful_agent_runs={},
        successful_agent_schedules={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_stagehand_patch():
    """Reset the process-level _stagehand_patched flag before every test."""
    _browse_web_mod._stagehand_patched = False
    yield
    _browse_web_mod._stagehand_patched = False


@pytest.fixture()
def env_vars(monkeypatch):
    """Inject the three env vars required by BrowseWebTool."""
    monkeypatch.setenv("STAGEHAND_API_KEY", "test-api-key")
    monkeypatch.setenv("STAGEHAND_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")


@pytest.fixture()
def mock_validate_url(monkeypatch):
    """Prevent real DNS lookups from validate_url in unit tests.

    Patched at the module level where it is imported so that all code paths
    in browse_web.py that call validate_url() use the no-op stub.
    """

    async def _noop(url: str, **kwargs: object) -> None:
        pass  # Accept all URLs (scheme is validated before this is called)

    monkeypatch.setattr(_browse_web_mod, "validate_url", _noop)


@pytest.fixture()
def stagehand_mocks(mock_validate_url, monkeypatch):
    """Inject mock stagehand + stagehand.main into sys.modules.

    Returns a dict with the mock objects so individual tests can
    assert on calls or inject side-effects.
    """
    # --- mock page ---
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {"extraction": "Page content here"}

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=None)
    mock_page.extract = AsyncMock(return_value=mock_result)

    # --- mock client ---
    mock_client = AsyncMock()
    mock_client.page = mock_page
    mock_client.init = AsyncMock(return_value=None)
    mock_client.close = AsyncMock(return_value=None)

    MockStagehand = MagicMock(return_value=mock_client)

    # --- stagehand top-level module ---
    mock_stagehand = MagicMock()
    mock_stagehand.Stagehand = MockStagehand

    # --- stagehand.main (needed by _patch_stagehand_once) ---
    mock_main = MagicMock()
    mock_main.Stagehand = MagicMock()
    mock_main.Stagehand._register_signal_handlers = MagicMock()

    monkeypatch.setitem(sys.modules, "stagehand", mock_stagehand)
    monkeypatch.setitem(sys.modules, "stagehand.main", mock_main)

    return {
        "client": mock_client,
        "page": mock_page,
        "result": mock_result,
        "MockStagehand": MockStagehand,
        "mock_main": mock_main,
    }


# ---------------------------------------------------------------------------
# 1. Tool metadata
# ---------------------------------------------------------------------------


class TestBrowseWebToolMetadata:
    def test_name(self):
        assert BrowseWebTool().name == "browse_web"

    def test_requires_auth(self):
        assert BrowseWebTool().requires_auth is True

    def test_url_is_required_parameter(self):
        params = BrowseWebTool().parameters
        assert "url" in params["properties"]
        assert "url" in params["required"]

    def test_instruction_is_optional(self):
        params = BrowseWebTool().parameters
        assert "instruction" in params["properties"]
        assert "instruction" not in params.get("required", [])

    def test_registered_in_tool_registry(self):
        from backend.copilot.tools import TOOL_REGISTRY

        assert "browse_web" in TOOL_REGISTRY
        assert isinstance(TOOL_REGISTRY["browse_web"], BrowseWebTool)

    def test_response_type_enum_value(self):
        assert ResponseType.BROWSE_WEB == "browse_web"


# ---------------------------------------------------------------------------
# 2. Input validation (no external deps)
# ---------------------------------------------------------------------------


class TestInputValidation:
    async def test_missing_url_returns_error(self):
        result = await BrowseWebTool()._execute(user_id="u1", session=make_session())
        assert isinstance(result, ErrorResponse)
        assert "url" in result.message.lower()

    async def test_empty_url_returns_error(self):
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url=""
        )
        assert isinstance(result, ErrorResponse)

    async def test_ftp_url_rejected(self):
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="ftp://example.com/file"
        )
        assert isinstance(result, ErrorResponse)
        assert "http" in result.message.lower()

    async def test_file_url_rejected(self):
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="file:///etc/passwd"
        )
        assert isinstance(result, ErrorResponse)

    async def test_javascript_url_rejected(self):
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="javascript:alert(1)"
        )
        assert isinstance(result, ErrorResponse)


# ---------------------------------------------------------------------------
# 3. Environment variable checks
# ---------------------------------------------------------------------------


class TestEnvVarChecks:
    async def test_missing_api_key(self, mock_validate_url, monkeypatch):
        monkeypatch.delenv("STAGEHAND_API_KEY", raising=False)
        monkeypatch.setenv("STAGEHAND_PROJECT_ID", "proj")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "not_configured"

    async def test_missing_project_id(self, mock_validate_url, monkeypatch):
        monkeypatch.setenv("STAGEHAND_API_KEY", "key")
        monkeypatch.delenv("STAGEHAND_PROJECT_ID", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "not_configured"

    async def test_missing_anthropic_key(self, mock_validate_url, monkeypatch):
        monkeypatch.setenv("STAGEHAND_API_KEY", "key")
        monkeypatch.setenv("STAGEHAND_PROJECT_ID", "proj")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "not_configured"


# ---------------------------------------------------------------------------
# 4. Stagehand absent (ImportError path)
# ---------------------------------------------------------------------------


class TestStagehandAbsent:
    async def test_returns_not_configured_error(
        self, env_vars, mock_validate_url, monkeypatch
    ):
        """Blocking the stagehand import must return a graceful ErrorResponse."""
        # sys.modules entry set to None → Python raises ImportError on import
        monkeypatch.setitem(sys.modules, "stagehand", None)
        monkeypatch.setitem(sys.modules, "stagehand.main", None)

        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "not_configured"
        assert "not available" in result.message or "not installed" in result.message

    async def test_other_tools_unaffected_when_stagehand_absent(
        self, env_vars, monkeypatch
    ):
        """Registry import must not raise even when stagehand is blocked."""
        monkeypatch.setitem(sys.modules, "stagehand", None)
        # This import already happened at module load; just verify the registry exists
        from backend.copilot.tools import TOOL_REGISTRY

        assert "browse_web" in TOOL_REGISTRY
        assert "web_fetch" in TOOL_REGISTRY  # unrelated tool still present


# ---------------------------------------------------------------------------
# 5. Successful browse
# ---------------------------------------------------------------------------


class TestSuccessfulBrowse:
    async def test_returns_browse_web_response(self, env_vars, stagehand_mocks):
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, BrowseWebResponse)
        assert result.url == "https://example.com"
        assert result.content == "Page content here"
        assert result.truncated is False

    async def test_http_url_accepted(self, env_vars, stagehand_mocks):
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="http://example.com"
        )
        assert isinstance(result, BrowseWebResponse)

    async def test_session_id_propagated(self, env_vars, stagehand_mocks):
        session = make_session()
        result = await BrowseWebTool()._execute(
            user_id="u1", session=session, url="https://example.com"
        )
        assert isinstance(result, BrowseWebResponse)
        assert result.session_id == session.session_id

    async def test_custom_instruction_forwarded_to_extract(
        self, env_vars, stagehand_mocks
    ):
        await BrowseWebTool()._execute(
            user_id="u1",
            session=make_session(),
            url="https://example.com",
            instruction="Extract all pricing plans",
        )
        stagehand_mocks["page"].extract.assert_awaited_once()
        first_arg = stagehand_mocks["page"].extract.call_args[0][0]
        assert first_arg == "Extract all pricing plans"

    async def test_default_instruction_used_when_omitted(
        self, env_vars, stagehand_mocks
    ):
        await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        first_arg = stagehand_mocks["page"].extract.call_args[0][0]
        assert "main content" in first_arg.lower()

    async def test_explicit_timeouts_passed_to_stagehand(
        self, env_vars, stagehand_mocks
    ):
        from backend.copilot.tools.browse_web import (
            _EXTRACT_TIMEOUT_MS,
            _GOTO_TIMEOUT_MS,
        )

        await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        goto_kwargs = stagehand_mocks["page"].goto.call_args[1]
        extract_kwargs = stagehand_mocks["page"].extract.call_args[1]
        assert goto_kwargs.get("timeoutMs") == _GOTO_TIMEOUT_MS
        assert extract_kwargs.get("timeoutMs") == _EXTRACT_TIMEOUT_MS

    async def test_client_closed_after_success(self, env_vars, stagehand_mocks):
        await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        stagehand_mocks["client"].close.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    async def test_short_content_not_truncated(self, env_vars, stagehand_mocks):
        stagehand_mocks["result"].model_dump.return_value = {"extraction": "short"}
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, BrowseWebResponse)
        assert result.truncated is False
        assert result.content == "short"

    async def test_oversized_content_is_truncated(self, env_vars, stagehand_mocks):
        big = "a" * (_MAX_CONTENT_CHARS + 1000)
        stagehand_mocks["result"].model_dump.return_value = {"extraction": big}
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, BrowseWebResponse)
        assert result.truncated is True
        assert result.content.endswith("[Content truncated]")

    async def test_truncated_content_never_exceeds_cap(self, env_vars, stagehand_mocks):
        """The final string must be ≤ _MAX_CONTENT_CHARS regardless of input size."""
        big = "b" * (_MAX_CONTENT_CHARS * 3)
        stagehand_mocks["result"].model_dump.return_value = {"extraction": big}
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, BrowseWebResponse)
        assert len(result.content) == _MAX_CONTENT_CHARS

    async def test_content_exactly_at_limit_not_truncated(
        self, env_vars, stagehand_mocks
    ):
        exact = "c" * _MAX_CONTENT_CHARS
        stagehand_mocks["result"].model_dump.return_value = {"extraction": exact}
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, BrowseWebResponse)
        assert result.truncated is False
        assert len(result.content) == _MAX_CONTENT_CHARS

    async def test_empty_extraction_returns_empty_content(
        self, env_vars, stagehand_mocks
    ):
        stagehand_mocks["result"].model_dump.return_value = {"extraction": ""}
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, BrowseWebResponse)
        assert result.content == ""
        assert result.truncated is False

    async def test_none_extraction_returns_empty_content(
        self, env_vars, stagehand_mocks
    ):
        stagehand_mocks["result"].model_dump.return_value = {"extraction": None}
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, BrowseWebResponse)
        assert result.content == ""


# ---------------------------------------------------------------------------
# 7. Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_stagehand_init_exception_returns_generic_error(
        self, env_vars, stagehand_mocks
    ):
        stagehand_mocks["client"].init.side_effect = RuntimeError("Connection refused")
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "browse_failed"

    async def test_raw_exception_text_not_leaked_to_user(
        self, env_vars, stagehand_mocks
    ):
        """Internal error details must not appear in the user-facing message."""
        stagehand_mocks["client"].init.side_effect = RuntimeError("SECRET_TOKEN_abc123")
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, ErrorResponse)
        assert "SECRET_TOKEN_abc123" not in result.message
        assert result.message == "Failed to browse URL."

    async def test_goto_timeout_returns_error(self, env_vars, stagehand_mocks):
        stagehand_mocks["page"].goto.side_effect = TimeoutError("Navigation timed out")
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "browse_failed"

    async def test_client_closed_after_exception(self, env_vars, stagehand_mocks):
        stagehand_mocks["page"].goto.side_effect = RuntimeError("boom")
        await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        stagehand_mocks["client"].close.assert_awaited_once()

    async def test_close_failure_does_not_propagate(self, env_vars, stagehand_mocks):
        """If close() itself raises, the tool must still return ErrorResponse."""
        stagehand_mocks["client"].init.side_effect = RuntimeError("init failed")
        stagehand_mocks["client"].close.side_effect = RuntimeError("close also failed")
        result = await BrowseWebTool()._execute(
            user_id="u1", session=make_session(), url="https://example.com"
        )
        assert isinstance(result, ErrorResponse)


# ---------------------------------------------------------------------------
# 8. Thread-safety of _patch_stagehand_once
# ---------------------------------------------------------------------------


class TestPatchStagehandOnce:
    def test_idempotent_double_call(self, stagehand_mocks):
        """_stagehand_patched transitions False→True exactly once."""
        assert _browse_web_mod._stagehand_patched is False
        _patch_stagehand_once()
        assert _browse_web_mod._stagehand_patched is True
        _patch_stagehand_once()  # second call — still True, not re-patched
        assert _browse_web_mod._stagehand_patched is True

    def test_safe_register_is_noop_in_worker_thread(self, stagehand_mocks):
        """The patched handler must silently do nothing when called from a worker."""
        _patch_stagehand_once()
        mock_main = sys.modules["stagehand.main"]
        safe_register = mock_main.Stagehand._register_signal_handlers

        errors: list[Exception] = []

        def run():
            try:
                safe_register(MagicMock())
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=run)
        t.start()
        t.join()

        assert errors == [], f"Worker thread raised: {errors}"

    def test_patched_flag_set_after_execution(self, env_vars, stagehand_mocks):
        """After a successful browse, _stagehand_patched must be True."""

        async def _run():
            return await BrowseWebTool()._execute(
                user_id="u1", session=make_session(), url="https://example.com"
            )

        import asyncio

        asyncio.get_event_loop().run_until_complete(_run())
        assert _browse_web_mod._stagehand_patched is True
