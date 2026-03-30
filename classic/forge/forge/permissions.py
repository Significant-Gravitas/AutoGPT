"""Permission management for agent command execution."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from forge.config.workspace_settings import AgentPermissions, WorkspaceSettings


class ApprovalScope(str, Enum):
    """Scope of permission approval."""

    ONCE = "once"  # Allow this one time only (not saved)
    AGENT = "agent"  # Always allow for this agent
    WORKSPACE = "workspace"  # Always allow for all agents
    DENY = "deny"  # Deny this command


class UserFeedbackProvided(Exception):
    """Raised when user provides feedback instead of approving/denying a command.

    This exception should be caught by the main loop to pass feedback to the agent
    via do_not_execute() instead of executing the command.
    """

    def __init__(self, feedback: str):
        self.feedback = feedback
        super().__init__(f"User provided feedback: {feedback}")


class PermissionCheckResult:
    """Result of a permission check.

    Attributes:
        allowed: Whether the command is allowed to execute.
        scope: The scope of the permission decision.
        feedback: Optional user feedback provided along with the decision.
    """

    __slots__ = ("allowed", "scope", "feedback")

    def __init__(
        self,
        allowed: bool,
        scope: ApprovalScope,
        feedback: str | None = None,
    ):
        self.allowed = allowed
        self.scope = scope
        self.feedback = feedback


class CommandPermissionManager:
    """Manages layered permissions for agent command execution.

    Check order (first match wins):
    1. Agent deny list → block
    2. Workspace deny list → block
    3. Agent allow list → allow
    4. Workspace allow list → allow
    5. No match → prompt user
    """

    def __init__(
        self,
        workspace: Path,
        agent_dir: Path,
        workspace_settings: WorkspaceSettings,
        agent_permissions: AgentPermissions,
        prompt_fn: (
            Callable[[str, str, dict], tuple[ApprovalScope, str | None]] | None
        ) = None,
        on_auto_approve: Callable[[str, str, dict, ApprovalScope], None] | None = None,
    ):
        """Initialize the permission manager.

        Args:
            workspace: Path to the workspace directory.
            agent_dir: Path to the agent's data directory.
            workspace_settings: Workspace-level permission settings.
            agent_permissions: Agent-specific permission settings.
            prompt_fn: Callback to prompt user for permission.
                Takes (command_name, args_str, arguments) and returns
                (ApprovalScope, feedback) tuple.
            on_auto_approve: Callback fired when a command is auto-approved
                from the allow lists (not prompted). Takes (command_name,
                args_str, arguments, scope).
        """
        self.workspace = workspace.resolve()
        self.agent_dir = agent_dir
        self.workspace_settings = workspace_settings
        self.agent_permissions = agent_permissions
        self.prompt_fn = prompt_fn
        self.on_auto_approve = on_auto_approve
        self._session_denied: set[str] = set()

    def check_command(
        self, command_name: str, arguments: dict[str, Any]
    ) -> PermissionCheckResult:
        """Check if command execution is allowed. Prompts if needed.

        Args:
            command_name: Name of the command to check.
            arguments: Command arguments.

        Returns:
            PermissionCheckResult with allowed status, scope, and optional feedback.
        """
        args_str = self._format_args(command_name, arguments)
        perm_string = f"{command_name}({args_str})"

        # 1. Check agent deny list
        if self._matches_patterns(
            command_name, args_str, self.agent_permissions.permissions.deny
        ):
            return PermissionCheckResult(False, ApprovalScope.DENY)

        # 2. Check workspace deny list
        if self._matches_patterns(
            command_name, args_str, self.workspace_settings.permissions.deny
        ):
            return PermissionCheckResult(False, ApprovalScope.DENY)

        # 3. Check agent allow list
        if self._matches_patterns(
            command_name, args_str, self.agent_permissions.permissions.allow
        ):
            if self.on_auto_approve:
                self.on_auto_approve(
                    command_name, args_str, arguments, ApprovalScope.AGENT
                )
            return PermissionCheckResult(True, ApprovalScope.AGENT)

        # 4. Check workspace allow list
        if self._matches_patterns(
            command_name, args_str, self.workspace_settings.permissions.allow
        ):
            if self.on_auto_approve:
                self.on_auto_approve(
                    command_name, args_str, arguments, ApprovalScope.WORKSPACE
                )
            return PermissionCheckResult(True, ApprovalScope.WORKSPACE)

        # 5. Check session denials
        if perm_string in self._session_denied:
            return PermissionCheckResult(False, ApprovalScope.DENY)

        # 6. Prompt user
        if self.prompt_fn is None:
            return PermissionCheckResult(False, ApprovalScope.DENY)

        scope, feedback = self.prompt_fn(command_name, args_str, arguments)
        pattern = self._generalize_pattern(command_name, args_str)

        if scope == ApprovalScope.ONCE:
            # Allow this one time only, don't save anywhere
            return PermissionCheckResult(True, ApprovalScope.ONCE, feedback)
        elif scope == ApprovalScope.WORKSPACE:
            self.workspace_settings.add_permission(pattern, self.workspace)
            return PermissionCheckResult(True, ApprovalScope.WORKSPACE, feedback)
        elif scope == ApprovalScope.AGENT:
            self.agent_permissions.add_permission(pattern, self.agent_dir)
            return PermissionCheckResult(True, ApprovalScope.AGENT, feedback)
        else:
            # Denied - feedback goes to agent instead of execution
            self._session_denied.add(perm_string)
            return PermissionCheckResult(False, ApprovalScope.DENY, feedback)

    def _format_args(self, command_name: str, arguments: dict[str, Any]) -> str:
        """Format command arguments for pattern matching.

        Args:
            command_name: Name of the command.
            arguments: Command arguments dict.

        Returns:
            Formatted arguments string.
        """
        # For file operations, use the resolved file path for symlink handling
        if command_name in ("read_file", "write_to_file", "list_folder"):
            path = arguments.get("filename") or arguments.get("path") or ""
            if path:
                return str(Path(path).resolve())
            return ""

        # For shell commands, format as "executable:args" (first word is executable)
        if command_name in ("execute_shell", "execute_python"):
            cmd = arguments.get("command_line") or arguments.get("code") or ""
            if not cmd:
                return ""
            parts = str(cmd).split(maxsplit=1)
            if len(parts) == 2:
                return f"{parts[0]}:{parts[1]}"
            return f"{parts[0]}:"

        # For web operations
        if command_name == "web_search":
            query = arguments.get("query", "")
            return str(query)
        if command_name == "read_webpage":
            url = arguments.get("url", "")
            return str(url)

        # Generic: join all argument values
        if arguments:
            return ":".join(str(v) for v in arguments.values())
        return "*"

    def _matches_patterns(self, cmd: str, args: str, patterns: list[str]) -> bool:
        """Check if command matches any pattern in the list.

        Args:
            cmd: Command name.
            args: Formatted arguments string.
            patterns: List of permission patterns.

        Returns:
            True if any pattern matches.
        """
        for pattern in patterns:
            if self._pattern_matches(pattern, cmd, args):
                return True
        return False

    def _pattern_matches(self, pattern: str, cmd: str, args: str) -> bool:
        """Check if a single pattern matches the command.

        Args:
            pattern: Permission pattern like "command_name(glob_pattern)".
            cmd: Command name.
            args: Formatted arguments string.

        Returns:
            True if pattern matches.
        """
        # Parse pattern: command_name(args_pattern)
        match = re.match(r"^(\w+)\((.+)\)$", pattern)
        if not match:
            return False

        pattern_cmd, args_pattern = match.groups()

        # Command name must match
        if pattern_cmd != cmd:
            return False

        # Expand {workspace} placeholder
        args_pattern = args_pattern.replace("{workspace}", str(self.workspace))

        # Convert glob pattern to regex
        # ** matches any path (including /)
        # * matches any characters except /
        regex_pattern = args_pattern
        regex_pattern = re.escape(regex_pattern)
        # Restore glob patterns
        regex_pattern = regex_pattern.replace(r"\*\*", ".*")
        regex_pattern = regex_pattern.replace(r"\*", "[^/]*")
        regex_pattern = f"^{regex_pattern}$"

        try:
            return bool(re.match(regex_pattern, args))
        except re.error:
            return False

    def _generalize_pattern(self, command_name: str, args_str: str) -> str:
        """Create a generalized pattern from specific command args.

        Args:
            command_name: Name of the command.
            args_str: Formatted arguments string.

        Returns:
            Generalized permission pattern.
        """
        # For file paths, generalize to parent directory
        if command_name in ("read_file", "write_to_file", "list_folder"):
            path = Path(args_str)
            # If within workspace, use {workspace} placeholder
            try:
                rel = path.resolve().relative_to(self.workspace)
                return f"{command_name}({{workspace}}/{rel.parent}/*)"
            except ValueError:
                # Outside workspace, use exact path
                return f"{command_name}({path})"

        # For shell commands, use executable:** pattern
        if command_name in ("execute_shell", "execute_python"):
            # args_str is in format "executable:args", extract executable
            if ":" in args_str:
                executable = args_str.split(":", 1)[0]
                return f"{command_name}({executable}:**)"
            return f"{command_name}(*)"

        # For web operations
        if command_name == "web_search":
            return "web_search(**)"
        if command_name == "read_webpage":
            # Extract domain
            match = re.match(r"https?://([^/]+)", args_str)
            if match:
                domain = match.group(1)
                return f"read_webpage(*{domain}*)"
            return "read_webpage(**)"

        # Generic: use ** wildcard to match any arguments including those with /
        return f"{command_name}(**)"
