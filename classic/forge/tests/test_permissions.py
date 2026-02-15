"""Tests for the permission management system."""

from pathlib import Path

import pytest

from forge.config.workspace_settings import (
    AgentPermissions,
    PermissionsConfig,
    WorkspaceSettings,
)
from forge.permissions import ApprovalScope, CommandPermissionManager


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    return tmp_path / "workspace"


@pytest.fixture
def agent_dir(tmp_path: Path) -> Path:
    """Create a temporary agent directory."""
    agent_dir = tmp_path / "agents" / "test-agent"
    agent_dir.mkdir(parents=True)
    return agent_dir


@pytest.fixture
def workspace_settings() -> WorkspaceSettings:
    """Create default workspace settings."""
    return WorkspaceSettings()


@pytest.fixture
def agent_permissions() -> AgentPermissions:
    """Create empty agent permissions."""
    return AgentPermissions()


@pytest.fixture
def permission_manager(
    workspace: Path,
    agent_dir: Path,
    workspace_settings: WorkspaceSettings,
    agent_permissions: AgentPermissions,
) -> CommandPermissionManager:
    """Create a permission manager for testing."""
    workspace.mkdir(parents=True, exist_ok=True)
    return CommandPermissionManager(
        workspace=workspace,
        agent_dir=agent_dir,
        workspace_settings=workspace_settings,
        agent_permissions=agent_permissions,
        prompt_fn=None,  # No prompting in tests
    )


class TestFormatArgs:
    """Tests for _format_args() method."""

    def test_format_args_read_file(self, permission_manager: CommandPermissionManager):
        """File operations should return resolved absolute path."""
        args = {"filename": "/tmp/test.txt"}
        result = permission_manager._format_args("read_file", args)
        assert result == str(Path("/tmp/test.txt").resolve())

    def test_format_args_read_file_with_path_key(
        self, permission_manager: CommandPermissionManager
    ):
        """File operations should also check 'path' key."""
        args = {"path": "/tmp/test.txt"}
        result = permission_manager._format_args("read_file", args)
        assert result == str(Path("/tmp/test.txt").resolve())

    def test_format_args_write_file(self, permission_manager: CommandPermissionManager):
        """write_to_file should format like read_file."""
        args = {"filename": "/tmp/output.txt"}
        result = permission_manager._format_args("write_to_file", args)
        assert result == str(Path("/tmp/output.txt").resolve())

    def test_format_args_list_folder(
        self, permission_manager: CommandPermissionManager
    ):
        """list_folder should format like read_file."""
        args = {"path": "/tmp"}
        result = permission_manager._format_args("list_folder", args)
        assert result == str(Path("/tmp").resolve())

    def test_format_args_shell_command_with_args(
        self, permission_manager: CommandPermissionManager
    ):
        """Shell commands should use executable:args format."""
        args = {"command_line": "rm -rf /tmp/foo"}
        result = permission_manager._format_args("execute_shell", args)
        assert result == "rm:-rf /tmp/foo"

    def test_format_args_shell_command_no_args(
        self, permission_manager: CommandPermissionManager
    ):
        """Shell commands without args should end with colon."""
        args = {"command_line": "ls"}
        result = permission_manager._format_args("execute_shell", args)
        assert result == "ls:"

    def test_format_args_shell_command_single_arg(
        self, permission_manager: CommandPermissionManager
    ):
        """Shell commands with single arg should format correctly."""
        args = {"command_line": "cat file.txt"}
        result = permission_manager._format_args("execute_shell", args)
        assert result == "cat:file.txt"

    def test_format_args_execute_python(
        self, permission_manager: CommandPermissionManager
    ):
        """execute_python should use same format as execute_shell."""
        args = {"code": "python script.py"}
        result = permission_manager._format_args("execute_python", args)
        assert result == "python:script.py"

    def test_format_args_shell_empty(
        self, permission_manager: CommandPermissionManager
    ):
        """Empty shell command should return empty string."""
        args = {"command_line": ""}
        result = permission_manager._format_args("execute_shell", args)
        assert result == ""

    def test_format_args_web_search(self, permission_manager: CommandPermissionManager):
        """Web search should return the query."""
        args = {"query": "python tutorial"}
        result = permission_manager._format_args("web_search", args)
        assert result == "python tutorial"

    def test_format_args_read_webpage(
        self, permission_manager: CommandPermissionManager
    ):
        """Read webpage should return the URL."""
        args = {"url": "https://example.com"}
        result = permission_manager._format_args("read_webpage", args)
        assert result == "https://example.com"

    def test_format_args_generic_command(
        self, permission_manager: CommandPermissionManager
    ):
        """Unknown commands should join values with colon."""
        args = {"arg1": "value1", "arg2": "value2"}
        result = permission_manager._format_args("unknown_cmd", args)
        assert result == "value1:value2"

    def test_format_args_generic_empty(
        self, permission_manager: CommandPermissionManager
    ):
        """Empty args for unknown commands should return wildcard."""
        result = permission_manager._format_args("unknown_cmd", {})
        assert result == "*"


class TestPatternMatches:
    """Tests for _pattern_matches() method."""

    def test_pattern_matches_exact(self, permission_manager: CommandPermissionManager):
        """Exact pattern should match."""
        assert permission_manager._pattern_matches(
            "read_file(/tmp/test.txt)", "read_file", "/tmp/test.txt"
        )

    def test_pattern_matches_single_wildcard(
        self, permission_manager: CommandPermissionManager
    ):
        """Single wildcard should match non-slash characters."""
        assert permission_manager._pattern_matches(
            "read_file(/tmp/*.txt)", "read_file", "/tmp/test.txt"
        )
        assert not permission_manager._pattern_matches(
            "read_file(/tmp/*.txt)", "read_file", "/tmp/subdir/test.txt"
        )

    def test_pattern_matches_double_wildcard(
        self, permission_manager: CommandPermissionManager
    ):
        """Double wildcard should match any path including slashes."""
        assert permission_manager._pattern_matches(
            "read_file(/tmp/**)", "read_file", "/tmp/test.txt"
        )
        assert permission_manager._pattern_matches(
            "read_file(/tmp/**)", "read_file", "/tmp/subdir/test.txt"
        )
        assert permission_manager._pattern_matches(
            "read_file(/tmp/**)", "read_file", "/tmp/a/b/c/test.txt"
        )

    def test_pattern_matches_workspace_placeholder(
        self, permission_manager: CommandPermissionManager
    ):
        """Workspace placeholder should expand to workspace path."""
        workspace_path = str(permission_manager.workspace)
        assert permission_manager._pattern_matches(
            "read_file({workspace}/**)",
            "read_file",
            f"{workspace_path}/test.txt",
        )

    def test_pattern_matches_wrong_command(
        self, permission_manager: CommandPermissionManager
    ):
        """Pattern should not match different command."""
        assert not permission_manager._pattern_matches(
            "read_file(/tmp/test.txt)", "write_to_file", "/tmp/test.txt"
        )

    def test_pattern_matches_shell_command(
        self, permission_manager: CommandPermissionManager
    ):
        """Shell command patterns should match executable:args format."""
        assert permission_manager._pattern_matches(
            "execute_shell(rm:**)", "execute_shell", "rm:-rf /tmp/foo"
        )
        assert permission_manager._pattern_matches(
            "execute_shell(rm:-rf **)", "execute_shell", "rm:-rf /tmp/foo"
        )

    def test_pattern_matches_shell_sudo(
        self, permission_manager: CommandPermissionManager
    ):
        """Sudo pattern should match any sudo command."""
        assert permission_manager._pattern_matches(
            "execute_shell(sudo:**)", "execute_shell", "sudo:rm -rf /"
        )
        assert permission_manager._pattern_matches(
            "execute_shell(sudo:**)", "execute_shell", "sudo:apt install foo"
        )

    def test_pattern_matches_env_file(
        self, permission_manager: CommandPermissionManager
    ):
        """Pattern should match .env files."""
        assert permission_manager._pattern_matches(
            "read_file(**.env)", "read_file", "/path/to/.env"
        )
        assert permission_manager._pattern_matches(
            "read_file(**.env)", "read_file", "/project/config/.env"
        )

    def test_pattern_matches_invalid_pattern(
        self, permission_manager: CommandPermissionManager
    ):
        """Invalid pattern format should not match."""
        assert not permission_manager._pattern_matches(
            "invalid_pattern", "read_file", "/tmp/test.txt"
        )
        assert not permission_manager._pattern_matches(
            "read_file", "read_file", "/tmp/test.txt"
        )

    def test_pattern_matches_wildcard_only(
        self, permission_manager: CommandPermissionManager
    ):
        """Wildcard-only pattern should match anything."""
        assert permission_manager._pattern_matches("finish(*)", "finish", "any_value")


class TestGeneralizePattern:
    """Tests for _generalize_pattern() method."""

    def test_generalize_file_in_workspace(
        self, permission_manager: CommandPermissionManager
    ):
        """File in workspace should use {workspace} placeholder."""
        workspace_path = permission_manager.workspace
        file_path = str(workspace_path / "subdir" / "test.txt")
        result = permission_manager._generalize_pattern("read_file", file_path)
        assert result == "read_file({workspace}/subdir/*)"

    def test_generalize_file_outside_workspace(
        self, permission_manager: CommandPermissionManager
    ):
        """File outside workspace should use exact path."""
        result = permission_manager._generalize_pattern(
            "read_file", "/outside/path/test.txt"
        )
        assert result == "read_file(/outside/path/test.txt)"

    def test_generalize_shell_command(
        self, permission_manager: CommandPermissionManager
    ):
        """Shell command should extract executable."""
        result = permission_manager._generalize_pattern(
            "execute_shell", "rm:-rf /tmp/foo"
        )
        assert result == "execute_shell(rm:**)"

    def test_generalize_shell_no_colon(
        self, permission_manager: CommandPermissionManager
    ):
        """Shell command without colon should return wildcard."""
        result = permission_manager._generalize_pattern("execute_shell", "invalid")
        assert result == "execute_shell(*)"

    def test_generalize_web_search(self, permission_manager: CommandPermissionManager):
        """Web search should generalize to double wildcard (matches anything incl /)."""
        result = permission_manager._generalize_pattern("web_search", "python tutorial")
        assert result == "web_search(**)"

    def test_generalize_read_webpage(
        self, permission_manager: CommandPermissionManager
    ):
        """Read webpage should extract domain."""
        result = permission_manager._generalize_pattern(
            "read_webpage", "https://example.com/page"
        )
        assert result == "read_webpage(*example.com*)"

    def test_generalize_unknown_command(
        self, permission_manager: CommandPermissionManager
    ):
        """Unknown command should use double wildcard (matches anything incl /)."""
        result = permission_manager._generalize_pattern("unknown_cmd", "some:args")
        assert result == "unknown_cmd(**)"


class TestCheckCommand:
    """Tests for check_command() method."""

    def test_check_command_allowed_by_workspace(
        self, permission_manager: CommandPermissionManager
    ):
        """Commands matching workspace allow list should be allowed."""
        workspace_path = str(permission_manager.workspace)
        # Create the file path that would be resolved
        file_path = f"{workspace_path}/test.txt"
        result = permission_manager.check_command("read_file", {"filename": file_path})
        assert result.allowed

    def test_check_command_denied_by_workspace(
        self, permission_manager: CommandPermissionManager
    ):
        """Commands matching workspace deny list should be denied."""
        # .env files are denied by default
        result = permission_manager.check_command(
            "read_file", {"filename": "/project/.env"}
        )
        assert not result.allowed

    def test_check_command_denied_shell_rm_rf(
        self, permission_manager: CommandPermissionManager
    ):
        """rm -rf should be denied by default."""
        result = permission_manager.check_command(
            "execute_shell", {"command_line": "rm -rf /tmp/foo"}
        )
        assert not result.allowed

    def test_check_command_denied_shell_rm_r(
        self, permission_manager: CommandPermissionManager
    ):
        """rm -r should be denied by default."""
        result = permission_manager.check_command(
            "execute_shell", {"command_line": "rm -r /tmp/foo"}
        )
        assert not result.allowed

    def test_check_command_denied_sudo(
        self, permission_manager: CommandPermissionManager
    ):
        """sudo commands should be denied by default."""
        result = permission_manager.check_command(
            "execute_shell", {"command_line": "sudo apt install foo"}
        )
        assert not result.allowed

    def test_check_command_allowed_safe_shell(self, workspace: Path, agent_dir: Path):
        """Safe shell commands should not match deny patterns."""
        workspace.mkdir(parents=True, exist_ok=True)
        # Create manager with custom settings that allow shell commands
        settings = WorkspaceSettings(
            permissions=PermissionsConfig(
                allow=["execute_shell(ls:**)", "execute_shell(cat:**)"],
                deny=[
                    "execute_shell(rm:-rf **)",
                    "execute_shell(sudo:**)",
                ],
            )
        )
        manager = CommandPermissionManager(
            workspace=workspace,
            agent_dir=agent_dir,
            workspace_settings=settings,
            agent_permissions=AgentPermissions(),
            prompt_fn=None,
        )
        assert manager.check_command(
            "execute_shell", {"command_line": "ls -la"}
        ).allowed
        assert manager.check_command(
            "execute_shell", {"command_line": "cat /tmp/file.txt"}
        ).allowed

    def test_check_command_agent_deny_overrides_workspace_allow(
        self, workspace: Path, agent_dir: Path
    ):
        """Agent deny list should override workspace allow list."""
        workspace.mkdir(parents=True, exist_ok=True)
        workspace_settings = WorkspaceSettings(
            permissions=PermissionsConfig(
                allow=["execute_shell(ls:**)"],
                deny=[],
            )
        )
        agent_permissions = AgentPermissions(
            permissions=PermissionsConfig(
                allow=[],
                deny=["execute_shell(ls:**)"],  # Agent denies ls
            )
        )
        manager = CommandPermissionManager(
            workspace=workspace,
            agent_dir=agent_dir,
            workspace_settings=workspace_settings,
            agent_permissions=agent_permissions,
            prompt_fn=None,
        )
        # Agent deny should block even though workspace allows
        result = manager.check_command("execute_shell", {"command_line": "ls -la"})
        assert not result.allowed

    def test_check_command_agent_allow_overrides_no_workspace(
        self, workspace: Path, agent_dir: Path
    ):
        """Agent allow list should work when workspace has no match."""
        workspace.mkdir(parents=True, exist_ok=True)
        workspace_settings = WorkspaceSettings(
            permissions=PermissionsConfig(allow=[], deny=[])
        )
        agent_permissions = AgentPermissions(
            permissions=PermissionsConfig(
                allow=["execute_shell(echo:**)"],
                deny=[],
            )
        )
        manager = CommandPermissionManager(
            workspace=workspace,
            agent_dir=agent_dir,
            workspace_settings=workspace_settings,
            agent_permissions=agent_permissions,
            prompt_fn=None,
        )
        result = manager.check_command("execute_shell", {"command_line": "echo hello"})
        assert result.allowed

    def test_check_command_no_prompt_fn_denies(self, workspace: Path, agent_dir: Path):
        """Without prompt_fn, unmatched commands should be denied."""
        workspace.mkdir(parents=True, exist_ok=True)
        settings = WorkspaceSettings(permissions=PermissionsConfig(allow=[], deny=[]))
        manager = CommandPermissionManager(
            workspace=workspace,
            agent_dir=agent_dir,
            workspace_settings=settings,
            agent_permissions=AgentPermissions(),
            prompt_fn=None,
        )
        # No allow patterns, no prompt, should deny
        result = manager.check_command(
            "execute_shell", {"command_line": "some_command"}
        )
        assert not result.allowed

    def test_check_command_session_denial(self, workspace: Path, agent_dir: Path):
        """Session denials should persist for the session."""
        workspace.mkdir(parents=True, exist_ok=True)
        denied_commands = []

        def mock_prompt(
            cmd: str, args_str: str, _args: dict
        ) -> tuple[ApprovalScope, str | None]:
            denied_commands.append((cmd, args_str))
            return (ApprovalScope.DENY, None)

        settings = WorkspaceSettings(permissions=PermissionsConfig(allow=[], deny=[]))
        manager = CommandPermissionManager(
            workspace=workspace,
            agent_dir=agent_dir,
            workspace_settings=settings,
            agent_permissions=AgentPermissions(),
            prompt_fn=mock_prompt,
        )

        # First call should prompt and deny
        result = manager.check_command("execute_shell", {"command_line": "bad_cmd"})
        assert not result.allowed
        assert len(denied_commands) == 1

        # Second call with same command should not prompt (session denial)
        result = manager.check_command("execute_shell", {"command_line": "bad_cmd"})
        assert not result.allowed
        assert len(denied_commands) == 1  # Still 1, no new prompt


class TestApprovalScopes:
    """Tests for different approval scopes."""

    def test_approval_once(self, workspace: Path, agent_dir: Path):
        """ONCE approval should allow but not persist."""
        workspace.mkdir(parents=True, exist_ok=True)
        prompt_count = [0]

        def mock_prompt(
            _cmd: str, _args_str: str, _args: dict
        ) -> tuple[ApprovalScope, str | None]:
            prompt_count[0] += 1
            return (ApprovalScope.ONCE, None)

        settings = WorkspaceSettings(permissions=PermissionsConfig(allow=[], deny=[]))
        manager = CommandPermissionManager(
            workspace=workspace,
            agent_dir=agent_dir,
            workspace_settings=settings,
            agent_permissions=AgentPermissions(),
            prompt_fn=mock_prompt,
        )

        # First call should prompt and allow
        result = manager.check_command("execute_shell", {"command_line": "cmd1"})
        assert result.allowed
        assert prompt_count[0] == 1

        # Second call should prompt again (ONCE doesn't persist)
        result = manager.check_command("execute_shell", {"command_line": "cmd1"})
        assert result.allowed
        assert prompt_count[0] == 2

    def test_approval_agent_persists_to_file(self, workspace: Path, agent_dir: Path):
        """AGENT approval should save to agent permissions file."""
        workspace.mkdir(parents=True, exist_ok=True)

        def mock_prompt(
            _cmd: str, _args_str: str, _args: dict
        ) -> tuple[ApprovalScope, str | None]:
            return (ApprovalScope.AGENT, None)

        agent_permissions = AgentPermissions()
        settings = WorkspaceSettings(permissions=PermissionsConfig(allow=[], deny=[]))
        manager = CommandPermissionManager(
            workspace=workspace,
            agent_dir=agent_dir,
            workspace_settings=settings,
            agent_permissions=agent_permissions,
            prompt_fn=mock_prompt,
        )

        result = manager.check_command("execute_shell", {"command_line": "mycmd args"})
        assert result.allowed

        # Check that permission was added to agent permissions
        assert "execute_shell(mycmd:**)" in agent_permissions.permissions.allow

        # Check that file was created
        perm_file = agent_dir / "permissions.yaml"
        assert perm_file.exists()

    def test_approval_workspace_persists_to_file(
        self, workspace: Path, agent_dir: Path
    ):
        """WORKSPACE approval should save to workspace settings file."""
        workspace.mkdir(parents=True, exist_ok=True)

        def mock_prompt(
            _cmd: str, _args_str: str, _args: dict
        ) -> tuple[ApprovalScope, str | None]:
            return (ApprovalScope.WORKSPACE, None)

        workspace_settings = WorkspaceSettings(
            permissions=PermissionsConfig(allow=[], deny=[])
        )
        manager = CommandPermissionManager(
            workspace=workspace,
            agent_dir=agent_dir,
            workspace_settings=workspace_settings,
            agent_permissions=AgentPermissions(),
            prompt_fn=mock_prompt,
        )

        result = manager.check_command("execute_shell", {"command_line": "mycmd args"})
        assert result.allowed

        # Check that permission was added to workspace settings
        assert "execute_shell(mycmd:**)" in workspace_settings.permissions.allow

        # Check that file was created
        settings_file = workspace / ".autogpt" / "autogpt.yaml"
        assert settings_file.exists()

    def test_approval_with_feedback(self, workspace: Path, agent_dir: Path):
        """Approval with feedback should return the feedback."""
        workspace.mkdir(parents=True, exist_ok=True)

        def mock_prompt(
            _cmd: str, _args_str: str, _args: dict
        ) -> tuple[ApprovalScope, str | None]:
            return (ApprovalScope.ONCE, "Be careful with this command")

        settings = WorkspaceSettings(permissions=PermissionsConfig(allow=[], deny=[]))
        manager = CommandPermissionManager(
            workspace=workspace,
            agent_dir=agent_dir,
            workspace_settings=settings,
            agent_permissions=AgentPermissions(),
            prompt_fn=mock_prompt,
        )

        result = manager.check_command("execute_shell", {"command_line": "cmd1"})
        assert result.allowed
        assert result.scope == ApprovalScope.ONCE
        assert result.feedback == "Be careful with this command"

    def test_denial_with_feedback(self, workspace: Path, agent_dir: Path):
        """Denial with feedback should return the feedback."""
        workspace.mkdir(parents=True, exist_ok=True)

        def mock_prompt(
            _cmd: str, _args_str: str, _args: dict
        ) -> tuple[ApprovalScope, str | None]:
            return (ApprovalScope.DENY, "Don't run this, try X instead")

        settings = WorkspaceSettings(permissions=PermissionsConfig(allow=[], deny=[]))
        manager = CommandPermissionManager(
            workspace=workspace,
            agent_dir=agent_dir,
            workspace_settings=settings,
            agent_permissions=AgentPermissions(),
            prompt_fn=mock_prompt,
        )

        result = manager.check_command("execute_shell", {"command_line": "bad_cmd"})
        assert not result.allowed
        assert result.scope == ApprovalScope.DENY
        assert result.feedback == "Don't run this, try X instead"

    def test_agent_approval_auto_approves_subsequent_calls(
        self, workspace: Path, agent_dir: Path
    ):
        """After AGENT approval, subsequent calls should auto-approve.

        This tests the scenario where multiple tools are executed in sequence -
        after approving the first one with 'Always (this agent)', subsequent
        calls should be auto-approved without prompting.
        """
        workspace.mkdir(parents=True, exist_ok=True)
        prompt_count = [0]

        def mock_prompt(
            _cmd: str, _args_str: str, _args: dict
        ) -> tuple[ApprovalScope, str | None]:
            prompt_count[0] += 1
            return (ApprovalScope.AGENT, None)

        agent_permissions = AgentPermissions()
        settings = WorkspaceSettings(permissions=PermissionsConfig(allow=[], deny=[]))
        auto_approved = []

        def on_auto_approve(
            cmd: str, args_str: str, args: dict, scope: ApprovalScope
        ) -> None:
            auto_approved.append((cmd, args_str, scope))

        manager = CommandPermissionManager(
            workspace=workspace,
            agent_dir=agent_dir,
            workspace_settings=settings,
            agent_permissions=agent_permissions,
            prompt_fn=mock_prompt,
            on_auto_approve=on_auto_approve,
        )

        # First call - should prompt and approve
        result1 = manager.check_command("ask_user", {"question": "What is your name?"})
        assert result1.allowed
        assert prompt_count[0] == 1
        assert "ask_user(**)" in agent_permissions.permissions.allow

        # Second call with different args - should auto-approve without prompting
        result2 = manager.check_command("ask_user", {"question": "What is your age?"})
        assert result2.allowed
        assert prompt_count[0] == 1  # Still 1 - no new prompt!
        assert len(auto_approved) == 1
        assert auto_approved[0][0] == "ask_user"
        assert auto_approved[0][2] == ApprovalScope.AGENT

        # Third call - also auto-approved
        result3 = manager.check_command(
            "ask_user", {"question": "Do you want /path/to/file?"}
        )
        assert result3.allowed
        assert prompt_count[0] == 1  # Still 1
        assert len(auto_approved) == 2


class TestDefaultDenyPatterns:
    """Tests to verify default deny patterns work correctly."""

    def test_deny_rm_rf_variations(self, permission_manager: CommandPermissionManager):
        """Various rm -rf commands should all be denied."""
        dangerous_commands = [
            "rm -rf /",
            "rm -rf /tmp",
            "rm -rf ~/",
            "rm -rf /home/user",
            "rm -rf .",
            "rm -rf ./*",
        ]
        for cmd in dangerous_commands:
            result = permission_manager.check_command(
                "execute_shell", {"command_line": cmd}
            )
            assert not result.allowed, f"Command '{cmd}' should be denied"

    def test_deny_rm_r_variations(self, permission_manager: CommandPermissionManager):
        """Various rm -r commands should all be denied."""
        dangerous_commands = [
            "rm -r /tmp",
            "rm -r /home/user",
        ]
        for cmd in dangerous_commands:
            result = permission_manager.check_command(
                "execute_shell", {"command_line": cmd}
            )
            assert not result.allowed, f"Command '{cmd}' should be denied"

    def test_deny_sudo_variations(self, permission_manager: CommandPermissionManager):
        """Various sudo commands should all be denied."""
        dangerous_commands = [
            "sudo rm -rf /",
            "sudo apt install something",
            "sudo chmod 777 /",
            "sudo su",
        ]
        for cmd in dangerous_commands:
            result = permission_manager.check_command(
                "execute_shell", {"command_line": cmd}
            )
            assert not result.allowed, f"Command '{cmd}' should be denied"

    def test_deny_env_files(self, permission_manager: CommandPermissionManager):
        """Reading .env files should be denied."""
        env_files = [
            "/project/.env",
            "/home/user/app/.env",
            "/var/www/.env.local",
            "/app/.env.production",
        ]
        for f in env_files:
            result = permission_manager.check_command("read_file", {"filename": f})
            assert not result.allowed, f"Reading '{f}' should be denied"

    def test_deny_key_files(self, permission_manager: CommandPermissionManager):
        """Reading .key files should be denied."""
        result = permission_manager.check_command(
            "read_file", {"filename": "/home/user/.ssh/id_rsa.key"}
        )
        assert not result.allowed

    def test_deny_pem_files(self, permission_manager: CommandPermissionManager):
        """Reading .pem files should be denied."""
        result = permission_manager.check_command(
            "read_file", {"filename": "/certs/server.pem"}
        )
        assert not result.allowed


class TestWorkspaceSettings:
    """Tests for WorkspaceSettings class."""

    def test_load_or_create_creates_default(self, tmp_path: Path):
        """load_or_create should create default settings file."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        settings = WorkspaceSettings.load_or_create(workspace)

        # Check defaults are set
        assert "read_file({workspace}/**)" in settings.permissions.allow
        assert "execute_shell(rm:-rf **)" in settings.permissions.deny

        # Check file was created
        settings_file = workspace / ".autogpt" / "autogpt.yaml"
        assert settings_file.exists()

    def test_load_or_create_loads_existing(self, tmp_path: Path):
        """load_or_create should load existing settings file."""
        workspace = tmp_path / "workspace"
        autogpt_dir = workspace / ".autogpt"
        autogpt_dir.mkdir(parents=True)

        # Create custom settings file
        settings_file = autogpt_dir / "autogpt.yaml"
        settings_file.write_text(
            """
permissions:
  allow:
    - custom_command(*)
  deny: []
"""
        )

        settings = WorkspaceSettings.load_or_create(workspace)

        assert settings.permissions.allow == ["custom_command(*)"]
        assert settings.permissions.deny == []

    def test_add_permission(self, tmp_path: Path):
        """add_permission should add and save permission."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        settings = WorkspaceSettings(permissions=PermissionsConfig(allow=[], deny=[]))
        settings.add_permission("new_pattern(*)", workspace)

        assert "new_pattern(*)" in settings.permissions.allow

        # Reload and verify persisted
        loaded = WorkspaceSettings.load_or_create(workspace)
        assert "new_pattern(*)" in loaded.permissions.allow


class TestAgentPermissions:
    """Tests for AgentPermissions class."""

    def test_load_or_create_returns_empty(self, tmp_path: Path):
        """load_or_create should return empty permissions if no file."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()

        permissions = AgentPermissions.load_or_create(agent_dir)

        assert permissions.permissions.allow == []
        assert permissions.permissions.deny == []
        # Should NOT create file if empty
        assert not (agent_dir / "permissions.yaml").exists()

    def test_load_or_create_loads_existing(self, tmp_path: Path):
        """load_or_create should load existing permissions file."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()

        # Create custom permissions file
        perm_file = agent_dir / "permissions.yaml"
        perm_file.write_text(
            """
permissions:
  allow:
    - agent_specific(*)
  deny:
    - agent_denied(*)
"""
        )

        permissions = AgentPermissions.load_or_create(agent_dir)

        assert permissions.permissions.allow == ["agent_specific(*)"]
        assert permissions.permissions.deny == ["agent_denied(*)"]

    def test_add_permission(self, tmp_path: Path):
        """add_permission should add and save permission."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()

        permissions = AgentPermissions()
        permissions.add_permission("new_agent_pattern(*)", agent_dir)

        assert "new_agent_pattern(*)" in permissions.permissions.allow

        # Verify file was created
        assert (agent_dir / "permissions.yaml").exists()

        # Reload and verify persisted
        loaded = AgentPermissions.load_or_create(agent_dir)
        assert "new_agent_pattern(*)" in loaded.permissions.allow
