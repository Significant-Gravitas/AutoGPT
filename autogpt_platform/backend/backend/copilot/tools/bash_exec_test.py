"""Tests for BashExecTool — E2B path with token injection."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b import CommandExitException
from pydantic import SecretStr

from backend.copilot import integration_creds
from backend.data.model import APIKeyCredentials

from ._test_data import make_session
from .bash_exec import BashExecTool
from .models import BashExecResponse, ErrorResponse

_USER = "user-bash-exec-test"
_PICKED = {"github": "cred-picked"}


@pytest.fixture(autouse=True)
def picked_credentials():
    """The chat's credential picks, which live in Redis outside these tests."""
    with patch(
        "backend.copilot.tools.bash_exec.selected_credentials",
        new=AsyncMock(return_value=_PICKED),
    ):
        yield


@pytest.fixture(autouse=True)
def proxy_off():
    """No swap proxy unless a test says so: the behaviour before it existed."""
    with patch("backend.copilot.tools.bash_exec.proxy_address", return_value=None):
        yield


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

        with (
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value=env_vars),
            ) as mock_get_env,
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ) as mock_identity,
        ):
            result = await tool._execute_on_e2b(
                sandbox=sandbox,
                command="echo hi",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )

        # The session's picks reach the token lookup, not just its scopes.
        mock_get_env.assert_awaited_once_with(_USER, None, _PICKED, None)
        # And the commit identity comes from that same GitHub account, so a
        # commit made with one account's token is not signed as another's.
        mock_identity.assert_awaited_once_with(_USER, "cred-picked")
        call_kwargs = sandbox.commands.run.call_args[1]
        assert call_kwargs["envs"]["GH_TOKEN"] == "gh-secret"
        assert call_kwargs["envs"]["GITHUB_TOKEN"] == "gh-secret"
        assert isinstance(result, BashExecResponse)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_git_identity_set_from_github_profile(self):
        """When user has a connected GitHub account, git env vars are set from their profile."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")
        identity = {
            "GIT_AUTHOR_NAME": "Test User",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test User",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }

        with (
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=identity),
            ),
        ):
            await tool._execute_on_e2b(
                sandbox=sandbox,
                command="git commit -m test",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )

        call_kwargs = sandbox.commands.run.call_args[1]
        assert call_kwargs["envs"]["GIT_AUTHOR_NAME"] == "Test User"
        assert call_kwargs["envs"]["GIT_AUTHOR_EMAIL"] == "test@example.com"
        assert call_kwargs["envs"]["GIT_COMMITTER_NAME"] == "Test User"
        assert call_kwargs["envs"]["GIT_COMMITTER_EMAIL"] == "test@example.com"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_git_identity_when_github_not_connected(self):
        """When user has no GitHub account, git identity env vars are absent."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")

        with (
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            await tool._execute_on_e2b(
                sandbox=sandbox,
                command="echo hi",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )

        call_kwargs = sandbox.commands.run.call_args[1]
        assert "GIT_AUTHOR_NAME" not in call_kwargs["envs"]
        assert "GIT_COMMITTER_EMAIL" not in call_kwargs["envs"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_nonzero_exit_returned_as_bash_exec_response(self):
        """CommandExitException (non-zero exit) must become a BashExecResponse with scrubbed output."""
        tool = _make_tool()
        session = make_session(user_id=_USER)

        sandbox = MagicMock()
        sandbox.commands.run = AsyncMock(
            side_effect=CommandExitException(
                stdout="not logged in gh-secret",
                stderr="oops gh-secret",
                exit_code=1,
                error=None,
            )
        )

        with (
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={"GH_TOKEN": "gh-secret"}),
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await tool._execute_on_e2b(
                sandbox=sandbox,
                command="gh auth status 2>&1",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )

        assert isinstance(result, BashExecResponse)
        assert result.exit_code == 1
        assert result.timed_out is False
        assert result.stdout == "not logged in [REDACTED]"
        assert result.stderr == "oops [REDACTED]"
        assert result.message == "Command executed with status code 1"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_token_injection_when_user_id_is_none(self):
        """When user_id is None, get_integration_env_vars must NOT be called."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")

        with (
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={"GH_TOKEN": "should-not-appear"}),
            ) as mock_get_env,
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ) as mock_get_identity,
        ):
            result = await tool._execute_on_e2b(
                sandbox=sandbox,
                command="echo hi",
                timeout=10,
                session_id=session.session_id,
                user_id=None,
            )

        mock_get_env.assert_not_called()
        mock_get_identity.assert_not_called()
        call_kwargs = sandbox.commands.run.call_args[1]
        assert "GH_TOKEN" not in call_kwargs["envs"]
        assert "GIT_AUTHOR_NAME" not in call_kwargs["envs"]
        assert isinstance(result, BashExecResponse)


class TestBashExecSdkToolResultRedirect:
    """A command that references an SDK tool-result path (e.g. the model
    tries to ``cat /root/.claude/projects/.../tool-results/foo.json``)
    must be short-circuited with a redirect to ``read_tool_result`` /
    ``@@agptfile`` before the sandbox returns the generic
    ``Permission denied`` that the model can't act on."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_redirect_on_absolute_sdk_path(self):
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox()
        cmd = (
            "cat /root/.claude/projects/-tmp-copilot-abc/"
            "abc/tool-results/toolu_x.json | jq ."
        )
        with patch(
            "backend.copilot.tools.bash_exec.get_current_sandbox",
            return_value=sandbox,
        ):
            result = await tool._execute(
                user_id=_USER,
                session=session,
                command=cmd,
                timeout=10,
            )
        assert isinstance(result, ErrorResponse)
        assert "read_tool_result" in result.message
        assert "@@agptfile" in result.message
        # Offending fragment must be the SDK path, not the executable name
        # — the model needs to know which fragment tripped the redirect.
        assert "tool-results/toolu_x.json" in result.message
        assert "Offending fragment: 'cat'" not in result.message
        # Sandbox must not have been invoked at all.
        sandbox.commands.run.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_redirect_on_user_path_containing_tool_outputs(self):
        """Regression: a user repo path that happens to contain a
        ``tool-outputs`` directory must NOT trigger the redirect, since
        the user's data isn't an SDK tool-result file."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")
        with (
            patch(
                "backend.copilot.tools.bash_exec.get_current_sandbox",
                return_value=sandbox,
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await tool._execute(
                user_id=_USER,
                session=session,
                command="ls my-pipeline/tool-outputs/data.json",
                timeout=10,
            )
        assert isinstance(result, BashExecResponse)
        sandbox.commands.run.assert_called_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_redirect_on_relative_tool_outputs_path(self):
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox()
        with patch(
            "backend.copilot.tools.bash_exec.get_current_sandbox",
            return_value=sandbox,
        ):
            result = await tool._execute(
                user_id=_USER,
                session=session,
                command="cat tool-outputs/toolu_x.json | head -50",
                timeout=10,
            )
        assert isinstance(result, ErrorResponse)
        assert "read_tool_result" in result.message
        sandbox.commands.run.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_normal_command_still_runs(self):
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="hello")
        with (
            patch(
                "backend.copilot.tools.bash_exec.get_current_sandbox",
                return_value=sandbox,
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await tool._execute(
                user_id=_USER,
                session=session,
                command="echo hello",
                timeout=10,
            )
        assert isinstance(result, BashExecResponse)
        sandbox.commands.run.assert_called_once()


class TestBashExecBehindTheSwapProxy:
    """With the box's egress through the swap proxy, the command gets
    placeholders: the stored value never reaches the box."""

    _REAL = "ghp_the_real_stored_value"

    @pytest.fixture(autouse=True)
    def proxy_on(self):
        with patch(
            "backend.copilot.tools.bash_exec.proxy_address", return_value="proxy:1080"
        ):
            yield

    @pytest.fixture(autouse=True)
    def stored_github_credential(self):
        manager = MagicMock()
        manager.store.get_creds_by_provider = AsyncMock(
            return_value=[
                APIKeyCredentials(
                    id="cred-picked",
                    provider="github",
                    api_key=SecretStr(self._REAL),
                    title="GitHub",
                )
            ]
        )
        with (
            patch.object(integration_creds, "_manager", manager),
            patch.object(integration_creds, "_ensure_cache_invalidation_listener"),
        ):
            integration_creds._token_cache.clear()
            integration_creds._null_cache.clear()
            integration_creds._credential_id_cache.clear()
            yield
            integration_creds._token_cache.clear()
            integration_creds._null_cache.clear()
            integration_creds._credential_id_cache.clear()

    @pytest.fixture(autouse=True)
    def grants(self):
        with patch(
            "backend.copilot.tools.bash_exec.grant_to_box", new=AsyncMock()
        ) as grant:
            self.grant = grant
            yield grant

    async def _run(self, sandbox: MagicMock) -> BashExecResponse:
        session = make_session(user_id=_USER)
        sandbox.sandbox_id = "sb-1"
        with patch(
            "backend.copilot.tools.bash_exec.get_github_user_git_identity",
            new=AsyncMock(return_value={"GIT_AUTHOR_NAME": "Ada"}),
        ) as identity:
            result = await _make_tool()._execute_on_e2b(
                sandbox=sandbox,
                command="gh repo list",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )
        # Commits stay attributed to the picked account.
        identity.assert_awaited_once_with(_USER, "cred-picked")
        assert isinstance(result, BashExecResponse)
        return result

    @pytest.mark.asyncio(loop_scope="session")
    async def test_the_command_gets_placeholders_and_no_token(self):
        sandbox = _make_sandbox(stdout="ok")
        with patch(
            "backend.copilot.tools.bash_exec.get_integration_env_vars"
        ) as real_tokens:
            await self._run(sandbox)
        real_tokens.assert_not_called()
        envs = sandbox.commands.run.call_args[1]["envs"]
        assert envs["GH_TOKEN"] == envs["GITHUB_TOKEN"] == "hsurr:github:cred-picked"
        assert envs["GIT_CONFIG_KEY_0"] == "credential.https://github.com.helper"
        assert envs["GIT_AUTHOR_NAME"] == "Ada"
        assert all(self._REAL not in value for value in envs.values())
        # Nothing in the box may drive E2B itself (read its rules, reconnect it).
        assert not any("E2B" in name for name in envs)
        # Granted to this box before the command ran, or it would not resolve.
        self.grant.assert_awaited_once_with("sb-1", {"github": "cred-picked"})

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_deleted_pick_blanks_the_variables_instead_of_falling_back(
        self,
    ):
        """The box's own environment may hold the default account's
        placeholder; left unset, the command would quietly act as it."""
        sandbox = _make_sandbox(stdout="ok")
        with patch(
            "backend.copilot.tools.bash_exec.selected_credentials",
            new=AsyncMock(return_value={"github": "cred-deleted"}),
        ):
            session = make_session(user_id=_USER)
            sandbox.sandbox_id = "sb-1"
            with patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ):
                await _make_tool()._execute_on_e2b(
                    sandbox=sandbox,
                    command="git push",
                    timeout=10,
                    session_id=session.session_id,
                    user_id=_USER,
                )
        envs = sandbox.commands.run.call_args[1]["envs"]
        assert envs["GH_TOKEN"] == envs["GITHUB_TOKEN"] == ""
        assert envs["GIT_CONFIG_COUNT"] == "0"
        self.grant.assert_awaited_once_with("sb-1", {})

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_grant_that_fails_does_not_stop_the_command(self):
        sandbox = _make_sandbox(stdout="ok")
        self.grant.side_effect = ConnectionError("redis down")
        result = await self._run(sandbox)
        assert result.exit_code == 0
        sandbox.commands.run.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_placeholder_in_the_output_is_not_redacted(self):
        sandbox = _make_sandbox(stdout="GH_TOKEN=hsurr:github:cred-picked")
        result = await self._run(sandbox)
        assert result.stdout == "GH_TOKEN=hsurr:github:cred-picked"


class TestBashExecWithoutTheSwapProxy:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_the_real_token_is_injected_as_before(self):
        sandbox = _make_sandbox(stdout="ok")
        session = make_session(user_id=_USER)
        with (
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={"GH_TOKEN": "gh-secret"}),
            ),
            patch("backend.copilot.tools.bash_exec.placeholder_grants") as placeholders,
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _make_tool()._execute_on_e2b(
                sandbox=sandbox,
                command="gh repo list",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )
        placeholders.assert_not_called()
        envs = sandbox.commands.run.call_args[1]["envs"]
        assert envs["GH_TOKEN"] == "gh-secret"
        assert "GIT_CONFIG_COUNT" not in envs


class TestBashExecUnderAProviderCeiling:
    """A run whose permissions leave GitHub out gets no GitHub variable,
    with or without the proxy."""

    @pytest.fixture(autouse=True)
    def no_github(self):
        from backend.copilot.permissions import CopilotPermissions

        with patch(
            "backend.copilot.tools.bash_exec.get_current_permissions",
            return_value=CopilotPermissions(
                providers=["github"], providers_exclude=True
            ),
        ):
            yield

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize("proxy", [None, "proxy:1080"])
    async def test_github_is_left_out(self, proxy):
        sandbox = _make_sandbox(stdout="ok")
        session = make_session(user_id=_USER)
        with (
            patch("backend.copilot.tools.bash_exec.proxy_address", return_value=proxy),
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={}),
            ) as real,
            patch(
                "backend.copilot.tools.bash_exec.placeholder_grants",
                new=AsyncMock(return_value={}),
            ) as placeholders,
            patch("backend.copilot.tools.bash_exec.grant_to_box", new=AsyncMock()),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _make_tool()._execute_on_e2b(
                sandbox=sandbox,
                command="gh repo list",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )
        called = placeholders if proxy else real
        assert called.await_args.args[3] == ()
