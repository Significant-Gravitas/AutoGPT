"""Workspace and agent permission settings for AutoGPT."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PermissionsConfig(BaseModel):
    """Configuration for allow/deny permission patterns."""

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class WorkspaceSettings(BaseModel):
    """Workspace-level permissions that apply to all agents."""

    permissions: PermissionsConfig = Field(
        default_factory=lambda: PermissionsConfig(
            allow=[
                "read_file({workspace}/**)",
                "write_file({workspace}/**)",
                "list_folder({workspace}/**)",
            ],
            deny=[
                "read_file(**.env)",
                "read_file(**.env.*)",
                "read_file(**.key)",
                "read_file(**.pem)",
                # Shell commands use format "executable:args"
                # Use ** to match paths containing /
                "execute_shell(rm:-rf **)",
                "execute_shell(rm:-r **)",
                "execute_shell(sudo:**)",
            ],
        )
    )

    @classmethod
    def load_or_create(cls, workspace: Path) -> "WorkspaceSettings":
        """Load settings from workspace or create default settings file.

        Args:
            workspace: Path to the workspace directory.

        Returns:
            WorkspaceSettings instance.
        """
        autogpt_dir = workspace / ".autogpt"
        settings_path = autogpt_dir / "autogpt.yaml"
        if settings_path.exists():
            with open(settings_path) as f:
                data = yaml.safe_load(f)
                return cls.model_validate(data or {})
        settings = cls()
        settings.save(workspace)
        return settings

    def save(self, workspace: Path) -> None:
        """Save settings to the workspace .autogpt/autogpt.yaml file.

        Args:
            workspace: Path to the workspace directory.
        """
        autogpt_dir = workspace / ".autogpt"
        autogpt_dir.mkdir(parents=True, exist_ok=True)
        settings_path = autogpt_dir / "autogpt.yaml"
        with open(settings_path, "w") as f:
            f.write("# autogpt.yaml - Workspace Permissions (all agents)\n")
            f.write("# Auto-generated and updated as you grant permissions\n\n")
            yaml.safe_dump(
                self.model_dump(), f, default_flow_style=False, sort_keys=False
            )

    def add_permission(self, pattern: str, workspace: Path) -> None:
        """Add a permission pattern to the allow list.

        Args:
            pattern: The permission pattern to add.
            workspace: Path to the workspace directory for saving.
        """
        if pattern not in self.permissions.allow:
            self.permissions.allow.append(pattern)
            self.save(workspace)


class AgentPermissions(BaseModel):
    """Agent-specific permissions that override workspace settings."""

    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)

    @classmethod
    def load_or_create(cls, agent_dir: Path) -> "AgentPermissions":
        """Load agent permissions or create empty permissions.

        Args:
            agent_dir: Path to the agent's data directory.

        Returns:
            AgentPermissions instance.
        """
        settings_path = agent_dir / "permissions.yaml"
        if settings_path.exists():
            with open(settings_path) as f:
                data = yaml.safe_load(f)
                return cls.model_validate(data or {})
        return cls()

    def save(self, agent_dir: Path) -> None:
        """Save agent permissions to permissions.yaml.

        Args:
            agent_dir: Path to the agent's data directory.
        """
        settings_path = agent_dir / "permissions.yaml"
        # Ensure directory exists
        agent_dir.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w") as f:
            f.write("# Agent-specific permissions\n")
            f.write("# These override workspace-level permissions\n\n")
            yaml.safe_dump(
                self.model_dump(), f, default_flow_style=False, sort_keys=False
            )

    def add_permission(self, pattern: str, agent_dir: Path) -> None:
        """Add a permission pattern to the agent's allow list.

        Args:
            pattern: The permission pattern to add.
            agent_dir: Path to the agent's data directory for saving.
        """
        if pattern not in self.permissions.allow:
            self.permissions.allow.append(pattern)
            self.save(agent_dir)
