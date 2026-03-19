"""Configuration management for chat system."""

import os
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from backend.util.clients import OPENROUTER_BASE_URL


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
        default=OPENROUTER_BASE_URL,
        description="Base URL for API (e.g., for OpenRouter)",
    )

    # Session TTL Configuration - 12 hours
    session_ttl: int = Field(default=43200, description="Session TTL in seconds")

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
    langfuse_prompt_cache_ttl: int = Field(
        default=300,
        description="Cache TTL in seconds for Langfuse prompt (0 to disable caching)",
    )

    # Rate limiting — token-based limits per day and per week.
    # Per-turn token cost varies with context size: ~10-15K for early turns,
    # ~30-50K mid-session, up to ~100K pre-compaction. Average across a
    # session with compaction cycles is ~25-35K tokens/turn, so 2.5M daily
    # allows ~70-100 turns/day.
    # Checked at the HTTP layer (routes.py) before each turn.
    #
    # TODO: These are deploy-time constants applied identically to every user.
    #  If per-user or per-plan limits are needed (e.g., free tier vs paid), these
    #  must move to the database (e.g., a UserPlan table) and get_usage_status /
    #  check_rate_limit would look up each user's specific limits instead of
    #  reading config.daily_token_limit / config.weekly_token_limit.
    daily_token_limit: int = Field(
        default=2_500_000,
        description="Max tokens per day, resets at midnight UTC (0 = unlimited)",
    )
    weekly_token_limit: int = Field(
        default=12_500_000,
        description="Max tokens per week, resets Monday 00:00 UTC (0 = unlimited)",
    )

    # Claude Agent SDK Configuration
    use_claude_agent_sdk: bool = Field(
        default=True,
        description="Use Claude Agent SDK (True) or OpenAI-compatible LLM baseline (False)",
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
        description="Max number of concurrent sub-agent Tasks the SDK can run per session.",
    )
    claude_agent_use_resume: bool = Field(
        default=True,
        description="Use --resume for multi-turn conversations instead of "
        "history compression. Falls back to compression when unavailable.",
    )
    use_openrouter: bool = Field(
        default=True,
        description="Enable routing API calls through the OpenRouter proxy. "
        "The actual decision also requires ``api_key`` and ``base_url`` — "
        "use the ``openrouter_active`` property for the final answer.",
    )
    use_claude_code_subscription: bool = Field(
        default=False,
        description="For personal/dev use: use Claude Code CLI subscription auth instead of API keys. Requires `claude login` on the host. Only works with SDK mode.",
    )
    test_mode: bool = Field(
        default=False,
        description="Use dummy service instead of real LLM calls. "
        "Send __test_transient_error__, __test_fatal_error__, or "
        "__test_slow_response__ to trigger specific scenarios.",
    )

    # E2B Sandbox Configuration
    use_e2b_sandbox: bool = Field(
        default=True,
        description="Use E2B cloud sandboxes for persistent bash/python execution. "
        "When enabled, bash_exec routes commands to E2B and SDK file tools "
        "operate directly on the sandbox via E2B's filesystem API.",
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
        default=420,  # 7 min safety net — allows headroom for compaction retries
        description="E2B sandbox running-time timeout (seconds). "
        "E2B timeout is wall-clock (not idle). Explicit per-turn pause is the primary "
        "mechanism; this is the safety net.",
    )
    e2b_sandbox_on_timeout: Literal["kill", "pause"] = Field(
        default="pause",
        description="E2B lifecycle action on timeout: 'pause' (default, free) or 'kill'.",
    )

    @property
    def openrouter_active(self) -> bool:
        """True when OpenRouter is enabled AND credentials are usable.

        Single source of truth for "will the SDK route through OpenRouter?".
        Checks the flag *and* that ``api_key`` + a valid ``base_url`` are
        present — mirrors the fallback logic in ``_build_sdk_env``.
        """
        if not self.use_openrouter:
            return False
        base = (self.base_url or "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return bool(self.api_key and base and base.startswith("http"))

    @property
    def e2b_active(self) -> bool:
        """True when E2B is enabled and the API key is present.

        Single source of truth for "should we use E2B right now?".
        Prefer this over combining ``use_e2b_sandbox`` and ``e2b_api_key``
        separately at call sites.
        """
        return self.use_e2b_sandbox and bool(self.e2b_api_key)

    @property
    def active_e2b_api_key(self) -> str | None:
        """Return the E2B API key when E2B is enabled and configured, else None.

        Combines the ``use_e2b_sandbox`` flag check and key presence into one.
        Use in callers::

            if api_key := config.active_e2b_api_key:
                # E2B is active; api_key is narrowed to str
        """
        return self.e2b_api_key if self.e2b_active else None

    @field_validator("e2b_api_key", mode="before")
    @classmethod
    def get_e2b_api_key(cls, v):
        """Get E2B API key from environment if not provided."""
        if not v:
            v = os.getenv("CHAT_E2B_API_KEY") or os.getenv("E2B_API_KEY")
        return v

    @field_validator("api_key", mode="before")
    @classmethod
    def get_api_key(cls, v):
        """Get API key from environment if not provided."""
        if not v:
            # Try to get from environment variables
            # First check for CHAT_API_KEY (Pydantic prefix)
            v = os.getenv("CHAT_API_KEY")
            if not v:
                # Fall back to OPEN_ROUTER_API_KEY
                v = os.getenv("OPEN_ROUTER_API_KEY")
            if not v:
                # Fall back to OPENAI_API_KEY
                v = os.getenv("OPENAI_API_KEY")
            # Note: ANTHROPIC_API_KEY is intentionally NOT included here.
            # The SDK CLI picks it up from the env directly. Including it
            # would pair it with the OpenRouter base_url, causing auth failures.
        return v

    @field_validator("base_url", mode="before")
    @classmethod
    def get_base_url(cls, v):
        """Get base URL from environment if not provided."""
        if not v:
            # Check for OpenRouter or custom base URL
            v = os.getenv("CHAT_BASE_URL")
            if not v:
                v = os.getenv("OPENROUTER_BASE_URL")
            if not v:
                v = os.getenv("OPENAI_BASE_URL")
            if not v:
                v = OPENROUTER_BASE_URL
        return v

    # Prompt paths for different contexts
    PROMPT_PATHS: dict[str, str] = {
        "default": "prompts/chat_system.md",
        "onboarding": "prompts/onboarding_system.md",
    }

    class Config:
        """Pydantic config."""

        env_prefix = "CHAT_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables
