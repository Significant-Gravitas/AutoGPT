"""
Shim configuration — loaded from ~/.autogpt/shim-config.toml or CLI args.
"""

from __future__ import annotations

import platform
import uuid
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class ShimConfig(BaseSettings):
    """
    All configuration for the local executor shim.

    Loaded from (in order of precedence):
    1. CLI flags
    2. Environment variables (AUTOGPT_SHIM_*)
    3. ~/.autogpt/shim-config.toml
    4. Defaults below
    """

    # --- Platform connection ---
    platform_ws_url: str = Field(
        default="wss://platform.autogpt.net/ws/local-executor",
        description="WebSocket URL of the AutoGPT platform. Override for self-hosted.",
    )
    platform_oauth_url: str = Field(
        default="https://platform.autogpt.net/auth",
        description="Base URL for the AutoGPT OAuth endpoints.",
    )
    oauth_client_id: str = Field(
        default="autogpt-local-executor",
        description="Well-known OAuth client ID for the shim. Do not change.",
    )
    oauth_redirect_port: int = Field(
        default=41899,
        description="Local port for the OAuth callback server during auth flow.",
    )

    # --- Machine identity ---
    machine_id: str = Field(
        default_factory=lambda: f"{platform.node()}-{uuid.uuid4().hex[:8]}",
        description="Stable identifier for this machine. Auto-generated on first run, "
                    "persisted to config file.",
    )

    # --- File access ---
    allowed_root: Path = Field(
        default_factory=lambda: Path.home() / ".autogpt" / "workspace",
        description="All file operations are jailed to this directory. "
                    "NEVER set to / or your home directory.",
    )

    # --- Capabilities to advertise ---
    enable_shell: bool = Field(
        default=True,
        description="Allow the platform to execute shell commands.",
    )
    enable_computer_use: bool = Field(
        default=False,
        description="Allow the platform to take screenshots and inject input. "
                    "REQUIRES explicit opt-in. Off by default.",
    )
    enable_local_llm: bool = Field(
        default=False,
        description="Allow the platform to route LLM inference to a local Ollama instance.",
    )
    enable_hardware: bool = Field(
        default=False,
        description="Allow the platform to access serial ports, USB devices, GPIO.",
    )

    # --- Rate limits ---
    max_commands_per_minute: int = Field(default=60)
    max_concurrent_commands: int = Field(default=4)
    max_file_size_bytes: int = Field(default=100 * 1024 * 1024)  # 100MB
    command_timeout_seconds: int = Field(default=30)

    # --- Audit ---
    audit_log_path: Path = Field(
        default_factory=lambda: Path.home() / ".autogpt" / "shim-audit.log",
    )

    # --- Reconnect ---
    reconnect_base_delay: float = Field(default=1.0)
    reconnect_max_delay: float = Field(default=60.0)

    class Config:
        env_prefix = "AUTOGPT_SHIM_"
