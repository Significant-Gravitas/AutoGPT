"""Tests for BaseTool execution logic."""

import asyncio
import json

import pytest

from backend.api.features.chat.model import ChatSession
from backend.api.features.chat.tools.base import (
    DEFAULT_LONG_RUNNING_TOOL_TIMEOUT,
    DEFAULT_TOOL_TIMEOUT,
    BaseTool,
)
from backend.api.features.chat.tools.models import (
    ResponseType,
    ToolResponseBase,
)


# ---------------------------------------------------------------------------
# Concrete test subclass
# ---------------------------------------------------------------------------


class _DummyTool(BaseTool):
    """Minimal concrete BaseTool used for testing."""

    def __init__(
        self,
        *,
        requires_auth: bool = False,
        is_long_running: bool = False,
        execute_fn=None,
    ):
        self._requires_auth = requires_auth
        self._is_long_running = is_long_running
        self._execute_fn = execute_fn

    @property
    def name(self) -> str:
        return "dummy_tool"

    @property
    def description(self) -> str:
        return "A dummy tool for testing."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def requires_auth(self) -> bool:
        return self._requires_auth

    @property
    def is_long_running(self) -> bool:
        return self._is_long_running

    async def _execute(self, user_id, session, **kwargs) -> ToolResponseBase:
        if self._execute_fn is not None:
            return await self._execute_fn(user_id, session, **kwargs)
        return ToolResponseBase(
            type=ResponseType.AGENTS_FOUND,
            message="ok",
            session_id=session.session_id,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TOOL_CALL_ID = "call_abc123"


@pytest.fixture()
def fake_session(mocker):
    """Return a lightweight mock of ChatSession."""
    session = mocker.MagicMock(spec=ChatSession)
    session.session_id = "sess_001"
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSuccessfulExecution:
    async def test_returns_proper_output(self, fake_session):
        tool = _DummyTool()
        result = await tool.execute(
            user_id="user_1",
            session=fake_session,
            tool_call_id=_TOOL_CALL_ID,
        )

        assert result.toolCallId == _TOOL_CALL_ID
        assert result.toolName == "dummy_tool"
        assert result.success is True

        payload = json.loads(result.output)
        assert payload["type"] == ResponseType.AGENTS_FOUND.value
        assert payload["message"] == "ok"
        assert payload["session_id"] == "sess_001"


class TestExecutionTimeout:
    async def test_timeout_returns_error_response(self, fake_session):
        async def _slow_execute(user_id, session, **kwargs):
            await asyncio.sleep(9999)

        tool = _DummyTool(execute_fn=_slow_execute)
        # Override timeout to something tiny so the test is fast.
        tool.__class__ = type(
            "_FastTimeoutTool",
            (_DummyTool,),
            {"timeout_seconds": property(lambda self: 0.01)},
        )

        result = await tool.execute(
            user_id="user_1",
            session=fake_session,
            tool_call_id=_TOOL_CALL_ID,
        )

        assert result.success is False
        assert result.toolCallId == _TOOL_CALL_ID
        assert result.toolName == "dummy_tool"

        payload = json.loads(result.output)
        assert payload["type"] == ResponseType.ERROR.value
        assert payload["error"] == "timeout"
        assert "timed out" in payload["message"]


class TestExecutionException:
    async def test_exception_returns_error_response(self, fake_session):
        async def _boom(user_id, session, **kwargs):
            raise RuntimeError("something broke")

        tool = _DummyTool(execute_fn=_boom)

        result = await tool.execute(
            user_id="user_1",
            session=fake_session,
            tool_call_id=_TOOL_CALL_ID,
        )

        assert result.success is False
        assert result.toolCallId == _TOOL_CALL_ID
        assert result.toolName == "dummy_tool"

        payload = json.loads(result.output)
        assert payload["type"] == ResponseType.ERROR.value
        assert payload["error"] == "something broke"
        assert "error occurred" in payload["message"]


class TestRequiresAuth:
    async def test_no_user_id_returns_need_login(self, fake_session):
        tool = _DummyTool(requires_auth=True)

        result = await tool.execute(
            user_id=None,
            session=fake_session,
            tool_call_id=_TOOL_CALL_ID,
        )

        assert result.success is False
        assert result.toolCallId == _TOOL_CALL_ID
        assert result.toolName == "dummy_tool"

        payload = json.loads(result.output)
        assert payload["type"] == ResponseType.NEED_LOGIN.value
        assert "sign in" in payload["message"].lower()

    async def test_with_user_id_proceeds_normally(self, fake_session):
        tool = _DummyTool(requires_auth=True)

        result = await tool.execute(
            user_id="user_1",
            session=fake_session,
            tool_call_id=_TOOL_CALL_ID,
        )

        assert result.success is True


class TestTimeoutSecondsDefaults:
    def test_default_timeout(self):
        tool = _DummyTool(is_long_running=False)
        assert tool.timeout_seconds == DEFAULT_TOOL_TIMEOUT

    def test_long_running_timeout(self):
        tool = _DummyTool(is_long_running=True)
        assert tool.timeout_seconds == DEFAULT_LONG_RUNNING_TOOL_TIMEOUT
