"""Configuration management for chat system."""

import os
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from backend.util.clients import OPENROUTER_BASE_URL

# Per-request routing mode for a single chat turn.
# - 'fast': route to the baseline OpenAI-compatible path with the cheaper model.
# - 'extended_thinking': route to the Claude Agent SDK path with the default
#   (opus) model.
# ``None`` means "no override"; the server falls back to the Claude Code
# subscription flag → LaunchDarkly COPILOT_SDK → config.use_claude_agent_sdk.
CopilotMode = Literal["fast", "extended_thinking"]

# Per-request model tier set by the frontend model toggle.
# 'standard' uses the global config default (currently Sonnet).
# 'advanced' forces the highest-capability model (currently Opus).
# None means no preference — falls through to LD per-user targeting, then config.
# Using tier names instead of model names keeps the contract model-agnostic.
CopilotLlmModel = Literal["standard", "advanced"]


class ChatConfig(BaseSettings):
    """Configuration for the chat system."""

    # OpenAI API Configuration
    model: str = Field(
        default="anthropic/claude-sonnet-4",
        description="Default model for extended thinking mode. "
        "Changed from Opus ($15/$75 per M) to Sonnet ($3/$15 per M) — "
        "5x cheaper. Override via CHAT_MODEL env var for Opus.",
    )
    fast_model: str = Field(
        default="anthropic/claude-sonnet-4",
        description="Model for fast mode (baseline path). Should be faster/cheaper than the default model.",
    )
    title_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Model to use for generating session titles (should be fast/cheap)",
    )
    simulation_model: str = Field(
        default="google/gemini-2.5-flash",
        description="Model for dry-run block simulation (should be fast/cheap with good JSON output)",
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
    # These are base limits for the FREE tier. Higher tiers (PRO, BUSINESS,
    # ENTERPRISE) multiply these by their tier multiplier (see
    # rate_limit.TIER_MULTIPLIERS). User tier is stored in the
    # User.subscriptionTier DB column and resolved inside
    # get_global_rate_limits().
    daily_token_limit: int = Field(
        default=2_500_000,
        description="Max tokens per day, resets at midnight UTC (0 = unlimited)",
    )
    weekly_token_limit: int = Field(
        default=12_500_000,
        description="Max tokens per week, resets Monday 00:00 UTC (0 = unlimited)",
    )

    # Cost (in credits / cents) to reset the daily rate limit using credits.
    # When a user hits their daily limit, they can spend this amount to reset
    # the daily counter and keep working.  Set to 0 to disable the feature.
    rate_limit_reset_cost: int = Field(
        default=500,
        ge=0,
        description="Credit cost (in cents) for resetting the daily rate limit. 0 = disabled.",
    )
    max_daily_resets: int = Field(
        default=5,
        ge=0,
        description="Maximum number of credit-based rate limit resets per user per day. 0 = unlimited.",
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
    claude_agent_fallback_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Fallback model when the primary model is unavailable (e.g. 529 "
        "overloaded). The SDK automatically retries with this cheaper model.",
    )
    claude_agent_max_turns: int = Field(
        default=50,
        ge=1,
        le=10000,
        description="Maximum number of agentic turns (tool-use loops) per query. "
        "Prevents runaway tool loops from burning budget. "
        "Changed from 1000 to 50 in SDK 0.1.58 upgrade — override via "
        "CHAT_CLAUDE_AGENT_MAX_TURNS env var if your workflows need more.",
    )
    claude_agent_max_budget_usd: float = Field(
        default=10.0,
        ge=0.01,
        le=1000.0,
        description="Maximum spend in USD per SDK query. The CLI attempts "
        "to wrap up gracefully when this budget is reached. "
        "Set to $10 to allow most tasks to complete (p50=$5.37, p75=$13.07). "
        "Override via CHAT_CLAUDE_AGENT_MAX_BUDGET_USD env var.",
    )
    claude_agent_max_thinking_tokens: int = Field(
        default=8192,
        ge=1024,
        le=128000,
        description="Maximum thinking/reasoning tokens per LLM call. "
        "Extended thinking on Opus can generate 50k+ tokens at $75/M — "
        "capping this is the single biggest cost lever. "
        "8192 is sufficient for most tasks; increase for complex reasoning.",
    )
    claude_agent_thinking_effort: Literal["low", "medium", "high", "max"] | None = (
        Field(
            default=None,
            description="Thinking effort level: 'low', 'medium', 'high', 'max', or None. "
            "Only applies to models with extended thinking (Opus). "
            "Sonnet doesn't have extended thinking — setting effort on Sonnet "
            "can cause <internal_reasoning> tag leaks. "
            "None = let the model decide. Override via CHAT_CLAUDE_AGENT_THINKING_EFFORT.",
        )
    )
    claude_agent_max_transient_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries for transient API errors "
        "(429, 5xx, ECONNRESET) before surfacing the error to the user.",
    )
    claude_agent_cross_user_prompt_cache: bool = Field(
        default=True,
        description="Enable cross-user prompt caching via SystemPromptPreset. "
        "The Claude Code default prompt becomes a cacheable prefix shared "
        "across all users, and our custom prompt is appended after it. "
        "Dynamic sections (working dir, git status, auto-memory) are excluded "
        "from the prefix. Set to False to fall back to passing the system "
        "prompt as a raw string.",
    )
    claude_agent_cli_path: str | None = Field(
        default=None,
        description="Optional explicit path to a Claude Code CLI binary. "
        "When set, the SDK uses this binary instead of the version bundled "
        "with the installed `claude-agent-sdk` package — letting us pin "
        "the Python SDK and the CLI independently. Critical for keeping "
        "OpenRouter compatibility while still picking up newer SDK API "
        "features (the bundled CLI version in 0.1.46+ is broken against "
        "OpenRouter — see PR #12294 and "
        "anthropics/claude-agent-sdk-python#789). Falls back to the "
        "bundled binary when unset. Reads from `CHAT_CLAUDE_AGENT_CLI_PATH` "
        "or the unprefixed `CLAUDE_AGENT_CLI_PATH` environment variable "
        "(same pattern as `api_key` / `base_url`).",
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
        present — mirrors the fallback logic in ``build_sdk_env``.
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

    @field_validator("claude_agent_cli_path", mode="before")
    @classmethod
    def get_claude_agent_cli_path(cls, v):
        """Resolve the Claude Code CLI override path from environment.

        Accepts either the Pydantic-prefixed ``CHAT_CLAUDE_AGENT_CLI_PATH``
        or the unprefixed ``CLAUDE_AGENT_CLI_PATH`` (matching the same
        fallback pattern used by ``api_key`` / ``base_url``). Keeping the
        unprefixed form working is important because the field is
        primarily an operator escape hatch set via container/host env,
        and the unprefixed name is what the PR description, the field
        docstrings, and the reproduction test in
        ``cli_openrouter_compat_test.py`` refer to.
        """
        if not v:
            v = os.getenv("CHAT_CLAUDE_AGENT_CLI_PATH")
            if not v:
                v = os.getenv("CLAUDE_AGENT_CLI_PATH")
        if v:
            if not os.path.exists(v):
                raise ValueError(
                    f"claude_agent_cli_path '{v}' does not exist. "
                    "Check the path or unset CLAUDE_AGENT_CLI_PATH to use "
                    "the bundled CLI."
                )
            if not os.path.isfile(v):
                raise ValueError(f"claude_agent_cli_path '{v}' is not a regular file.")
            if not os.access(v, os.X_OK):
                raise ValueError(
                    f"claude_agent_cli_path '{v}' exists but is not executable. "
                    "Check file permissions."
                )
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
