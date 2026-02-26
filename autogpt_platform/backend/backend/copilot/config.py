"""Configuration management for chat system."""

import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class ChatConfig(BaseSettings):
    """Configuration for the chat system."""

    # OpenAI API Configuration
    model: str = Field(
        default="anthropic/claude-opus-4.6", description="Default model to use"
    )
    title_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Model to use for generating session titles (should be fast/cheap)",
    )
    api_key: str | None = Field(default=None, description="OpenAI API key")
    base_url: str | None = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for API (e.g., for OpenRouter)",
    )

    # Session TTL Configuration - 12 hours
    session_ttl: int = Field(default=43200, description="Session TTL in seconds")

    # Streaming Configuration
    max_retries: int = Field(
        default=3,
        description="Max retries for fallback path (SDK handles retries internally)",
    )
    max_agent_runs: int = Field(default=30, description="Maximum number of agent runs")
    max_agent_schedules: int = Field(
        default=30, description="Maximum number of agent schedules"
    )

    # Stream registry configuration for SSE reconnection
    stream_ttl: int = Field(
        default=3600,
        description="TTL in seconds for stream data in Redis (1 hour)",
    )
    stream_lock_ttl: int = Field(
        default=120,
        description="TTL in seconds for stream lock (2 minutes). Short timeout allows "
        "reconnection after refresh/crash without long waits.",
    )
    stream_max_length: int = Field(
        default=10000,
        description="Maximum number of messages to store per stream",
    )

    # Redis key prefixes for stream registry
    session_meta_prefix: str = Field(
        default="chat:task:meta:",
        description="Prefix for session metadata hash keys",
    )
    turn_stream_prefix: str = Field(
        default="chat:stream:",
        description="Prefix for turn message stream keys",
    )

    # Langfuse Prompt Management Configuration
    # Note: Langfuse credentials are in Settings().secrets (settings.py)
    langfuse_prompt_name: str = Field(
        default="CoPilot Prompt",
        description="Name of the prompt in Langfuse to fetch",
    )

    # Claude Agent SDK Configuration
    use_claude_agent_sdk: bool = Field(
        default=True,
        description="Use Claude Agent SDK for chat completions",
    )
    claude_agent_model: str | None = Field(
        default=None,
        description="Model for the Claude Agent SDK path. If None, derives from "
        "the `model` field by stripping the OpenRouter provider prefix.",
    )
    claude_agent_max_buffer_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB (default SDK is 1MB)
        description="Max buffer size in bytes for Claude Agent SDK JSON message parsing. "
        "Increase if tool outputs exceed the limit.",
    )
    claude_agent_max_subtasks: int = Field(
        default=10,
        description="Max number of sub-agent Tasks the SDK can spawn per session.",
    )
    claude_agent_use_resume: bool = Field(
        default=True,
        description="Use --resume for multi-turn conversations instead of "
        "history compression. Falls back to compression when unavailable.",
    )

    # E2B Sandbox Configuration
    use_e2b_sandbox: bool = Field(
        default=False,
        description="Use E2B cloud sandboxes for persistent bash/python execution. "
        "When enabled, bash_exec routes commands to E2B and the session workspace "
        "is mounted via sshfs so SDK file tools share the same filesystem.",
    )
    e2b_api_key: str | None = Field(
        default=None,
        description="E2B API key. Falls back to E2B_API_KEY environment variable.",
    )
    e2b_sandbox_template: str = Field(
        default="base",
        description="E2B sandbox template to use for copilot sessions.",
    )
    e2b_sandbox_timeout: int = Field(
        default=43200,  # 12 hours — same as session_ttl
        description="E2B sandbox keepalive timeout in seconds.",
    )

    # Extended thinking configuration for Claude models
    thinking_enabled: bool = Field(
        default=True,
        description="Enable adaptive thinking for Claude models via OpenRouter",
    )

    @field_validator("use_e2b_sandbox", mode="before")
    @classmethod
    def get_use_e2b_sandbox(cls, v):
        """Get use_e2b_sandbox from environment if not provided."""
        env_val = os.getenv("CHAT_USE_E2B_SANDBOX", "").lower()
        if env_val:
            return env_val in ("true", "1", "yes", "on")
        return False if v is None else v

    @field_validator("e2b_api_key", mode="before")
    @classmethod
    def get_e2b_api_key(cls, v):
        """Get E2B API key from environment if not provided."""
        if v is None:
            v = os.getenv("CHAT_E2B_API_KEY") or os.getenv("E2B_API_KEY")
        return v

    @field_validator("api_key", mode="before")
    @classmethod
    def get_api_key(cls, v):
        """Get API key from environment if not provided."""
        if v is None:
            # Try to get from environment variables
            # First check for CHAT_API_KEY (Pydantic prefix)
            v = os.getenv("CHAT_API_KEY")
            if not v:
                # Fall back to OPEN_ROUTER_API_KEY
                v = os.getenv("OPEN_ROUTER_API_KEY")
            if not v:
                # Fall back to OPENAI_API_KEY
                v = os.getenv("OPENAI_API_KEY")
        return v

    @field_validator("base_url", mode="before")
    @classmethod
    def get_base_url(cls, v):
        """Get base URL from environment if not provided."""
        if v is None:
            # Check for OpenRouter or custom base URL
            v = os.getenv("CHAT_BASE_URL")
            if not v:
                v = os.getenv("OPENROUTER_BASE_URL")
            if not v:
                v = os.getenv("OPENAI_BASE_URL")
            if not v:
                v = "https://openrouter.ai/api/v1"
        return v

    @field_validator("use_claude_agent_sdk", mode="before")
    @classmethod
    def get_use_claude_agent_sdk(cls, v):
        """Get use_claude_agent_sdk from environment if not provided."""
        # Check environment variable - default to True if not set
        env_val = os.getenv("CHAT_USE_CLAUDE_AGENT_SDK", "").lower()
        if env_val:
            return env_val in ("true", "1", "yes", "on")
        # Default to True (SDK enabled by default)
        return True if v is None else v

    # Prompt paths for different contexts
    PROMPT_PATHS: dict[str, str] = {
        "default": "prompts/chat_system.md",
        "onboarding": "prompts/onboarding_system.md",
    }

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables
