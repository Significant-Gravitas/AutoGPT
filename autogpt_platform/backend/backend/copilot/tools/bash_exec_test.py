"""Tests for BashExecTool — E2B path with token injection."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._test_data import make_session
from .bash_exec import BashExecTool
from .models import BashExecResponse

_USER = "user-bash-exec-test"


def _make_tool() -> BashExecTool:
    return BashExecTool()


def _make_sandbox(exit_code: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.exit_code = exit_code
    result.stdout = stdout
    result.stderr = stderr

    sandbox = MagicMock()
    sandbox.commands.run = AsyncMock(return_value=result)
    return sandbox


class TestBashExecE2BTokenInjection:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_token_injected_when_user_id_set(self):
        """When user_id is provided, integration env vars are merged into sandbox envs."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")
        env_vars = {"GH_TOKEN": "gh-secret", "GITHUB_TOKEN": "gh-secret"}

        with patch(
            "backend.copilot.tools.bash_exec.get_integration_env_vars",
            new=AsyncMock(return_value=env_vars),
        ) as mock_get_env:
            result = await tool._execute_on_e2b(
                sandbox=sandbox,
                command="echo hi",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )

        mock_get_env.assert_awaited_once_with(_USER)
        call_kwargs = sandbox.commands.run.call_args[1]
        assert call_kwargs["envs"]["GH_TOKEN"] == "gh-secret"
        assert call_kwargs["envs"]["GITHUB_TOKEN"] == "gh-secret"
        assert isinstance(result, BashExecResponse)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_token_injection_when_user_id_is_none(self):
        """When user_id is None, get_integration_env_vars must NOT be called."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")

        with patch(
            "backend.copilot.tools.bash_exec.get_integration_env_vars",
            new=AsyncMock(return_value={"GH_TOKEN": "should-not-appear"}),
        ) as mock_get_env:
            result = await tool._execute_on_e2b(
                sandbox=sandbox,
                command="echo hi",
                timeout=10,
                session_id=session.session_id,
                user_id=None,
            )

        mock_get_env.assert_not_called()
        call_kwargs = sandbox.commands.run.call_args[1]
        assert "GH_TOKEN" not in call_kwargs["envs"]
        assert isinstance(result, BashExecResponse)
