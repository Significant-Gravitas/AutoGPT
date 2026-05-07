"""Configuration management for chat system."""

import os
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
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
# 'standard' picks the cheaper everyday model for the active path —
#   ``fast_standard_model`` on the baseline path, ``thinking_standard_model``
#   on the SDK path.
# 'advanced' picks the premium model for the active path — ``fast_advanced_model``
#   on the baseline path, ``thinking_advanced_model`` on the SDK path (both
#   default to Opus today).
# None means no preference — falls through to LD per-user targeting, then config.
# Using tier names instead of model names keeps the contract model-agnostic.
CopilotLlmModel = Literal["standard", "advanced"]


class ChatConfig(BaseSettings):
    """Configuration for the chat system."""

    # Chat model tiers — a 2×2 of (path, tier).  ``path`` = ``CopilotMode``
    # (``"fast"`` → baseline OpenAI-compat / any OpenRouter model;
    # ``"extended_thinking"`` → Claude Agent SDK, Anthropic-only CLI).
    # ``tier`` = ``CopilotLlmModel`` (``"standard"`` / ``"advanced"``).
    # Each cell has its own config so the two paths can evolve
    # independently (cheap provider on baseline, Anthropic on SDK) at each
    # tier without conflating one path's needs with the other's constraint.
    #
    # Historical env var names (``CHAT_MODEL`` / ``CHAT_ADVANCED_MODEL`` /
    # ``CHAT_FAST_MODEL``) are preserved via ``validation_alias`` so
    # existing deployments continue to override the same effective cell.
    fast_standard_model: str = Field(
        default="anthropic/claude-sonnet-4-6",
        validation_alias=AliasChoices(
            "CHAT_FAST_STANDARD_MODEL",
            "CHAT_FAST_MODEL",
        ),
        description="Baseline path, 'standard' / ``None`` tier.  Per-user "
        "overrides flow through ``copilot-model-routing[fast][standard]`` "
        "(see ``copilot/model_router.py``); this value is the fallback.",
    )
    fast_advanced_model: str = Field(
        default="anthropic/claude-opus-4.7",
        validation_alias=AliasChoices("CHAT_FAST_ADVANCED_MODEL"),
        description="Baseline path, 'advanced' tier.  LD override: "
        "``copilot-model-routing[fast][advanced]``.",
    )
    thinking_standard_model: str = Field(
        default="anthropic/claude-sonnet-4-6",
        validation_alias=AliasChoices(
            "CHAT_THINKING_STANDARD_MODEL",
            "CHAT_MODEL",
        ),
        description="SDK (extended-thinking) path, 'standard' / ``None`` "
        "tier.  LD override: ``copilot-model-routing[thinking][standard]``.",
    )
    thinking_advanced_model: str = Field(
        default="anthropic/claude-opus-4.7",
        validation_alias=AliasChoices(
            "CHAT_THINKING_ADVANCED_MODEL",
            "CHAT_ADVANCED_MODEL",
        ),
        description="SDK (extended-thinking) path, 'advanced' tier.  LD "
        "override: ``copilot-model-routing[thinking][advanced]``.",
    )
    title_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Model to use for generating session titles (should be fast/cheap)",
    )
    simulation_model: str = Field(
        default="google/gemini-2.5-flash-lite",
        description="Model for dry-run block simulation (should be fast/cheap with good JSON output). "
        "Gemini 2.5 Flash-Lite is ~3x cheaper than Flash ($0.10/$0.40 vs $0.30/$1.20 per MTok) "
        "with JSON-mode reliability adequate for shape-matching block outputs.",
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

    # Rate limiting — cost-based limits per day and per week, stored in
    # microdollars (1 USD = 1_000_000).  The counter tracks the real
    # generation cost reported by the provider (OpenRouter ``usage.cost``
    # or Claude Agent SDK ``total_cost_usd``), so cache discounts and
    # cross-model price differences are already reflected — no token
    # weighting or model multiplier is applied on top.
    # Checked at the HTTP layer (routes.py) before each turn.
    #
    # These are base limits for the FREE tier.  Higher tiers (PRO, BUSINESS,
    # ENTERPRISE) multiply these by their tier multiplier (see
    # rate_limit.TIER_MULTIPLIERS).  User tier is stored in the
    # User.subscriptionTier DB column and resolved inside
    # get_global_rate_limits().
    #
    # These defaults act as the ceiling when LaunchDarkly is unreachable;
    # the live per-tier values come from the COPILOT_*_COST_LIMIT flags.
    daily_cost_limit_microdollars: int = Field(
        default=1_000_000,
        description="Max cost per day in microdollars, resets at midnight UTC "
        "(0 = unlimited).",
    )
    weekly_cost_limit_microdollars: int = Field(
        default=5_000_000,
        description="Max cost per week in microdollars, resets Monday 00:00 UTC "
        "(0 = unlimited).",
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
        "`thinking_standard_model` by stripping the OpenRouter provider prefix.",
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
        default="",
        description="Fallback model when the primary model is unavailable (e.g. 529 "
        "overloaded). The SDK automatically retries with this cheaper model. "
        "Empty string disables the fallback (no --fallback-model flag passed to CLI).",
    )
    agent_max_turns: int = Field(
        default=100,
        ge=1,
        le=10000,
        validation_alias=AliasChoices(
            "CHAT_AGENT_MAX_TURNS",
            "CHAT_CLAUDE_AGENT_MAX_TURNS",
        ),
        description="Maximum number of tool-call rounds per turn — applies to "
        "both the baseline and Claude Agent SDK paths. Prevents runaway tool "
        "loops from burning budget. Override via CHAT_AGENT_MAX_TURNS env var "
        "(legacy CHAT_CLAUDE_AGENT_MAX_TURNS still accepted).",
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
    claude_agent_autocompact_pct_override: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Auto-compaction trigger threshold as a percentage of the "
        "CLI's perceived window (sets ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`` on the "
        "SDK subprocess). The CLI caps at its default (~93% of window); values "
        "above that have no effect. 50 (= 100K of a 200K window) keeps Anthropic "
        "context creation costs down. Set to 0 to omit the env var entirely "
        "and let the CLI use its default ~93% threshold — useful when the "
        "post-compaction floor (system prompt + tool defs ≈ 65-110K) is close "
        "to the trigger and a more aggressive value causes back-to-back "
        "compaction cascades. Skipped unconditionally for Moonshot routes.",
    )
    claude_agent_max_thinking_tokens: int = Field(
        deprecated=(
            "Setting a thinking token budget is not supported in Claude 4.7+. "
            "Use `claude_agent_thinking_effort` instead to steer thinking effort."
        ),
        default=8192,
        ge=0,
        le=128000,
        description="Maximum thinking/reasoning tokens per LLM call. Applies "
        "to both the Claude Agent SDK path (as ``max_thinking_tokens``) and "
        "the baseline OpenRouter path (as ``extra_body.reasoning.max_tokens`` "
        "on Anthropic routes). Extended thinking on Opus can generate 50k+ "
        "tokens at $75/M — capping this is the single biggest cost lever. "
        "8192 is sufficient for most tasks; increase for complex reasoning. "
        "Set to 0 to disable extended thinking on both paths (kill switch): "
        "baseline skips the ``reasoning`` extra_body; SDK omits the "
        "``max_thinking_tokens`` kwarg so the CLI falls back to model default "
        "(which, without the flag, leaves extended thinking off).",
    )
    render_reasoning_in_ui: bool = Field(
        default=True,
        description="Render reasoning as live UI parts "
        "(``StreamReasoning*`` wire events). False suppresses the live "
        "wire events only; ``role='reasoning'`` rows are always persisted "
        "so the reasoning bubble hydrates on reload. Tokens are billed "
        "upstream regardless.",
    )
    stream_replay_count: int = Field(
        default=200,
        ge=1,
        le=10000,
        description="Max Redis stream entries replayed on SSE reconnect.",
    )
    claude_agent_thinking_effort: Literal["low", "medium", "high", "max"] | None = (
        # TODO: add xhigh when SDK support catches up
        Field(
            default=None,
            description="Thinking effort level: 'low', 'medium', 'high', 'max', or None. "
            "Applies to models that emit a reasoning channel — Sonnet, Opus, "
            "and Mythos (adaptive thinking) and Kimi K2.6 "
            "(OpenRouter ``reasoning`` extension lit up by #12871). "
            "Check https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking "  # noqa
            "for model compatibility and guidance. "
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
    baseline_prompt_cache_ttl: str = Field(
        default="1h",
        description="TTL for the ephemeral prompt-cache markers on the baseline "
        "OpenRouter path. Anthropic supports only `5m` (default, 1.25x input "
        "price for the write) or `1h` (2x input price for the write). 1h is "
        "strictly cheaper overall when the static prefix gets >7 reads per "
        "write-window; since the system prompt + tools array is identical "
        "across all users in our workspace, 1h is the default so cross-user "
        "reads amortise the higher write cost. Anthropic has no longer "
        "(24h, permanent) TTL option — see "
        "https://platform.claude.com/docs/en/build-with-claude/prompt-caching.",
    )
    sdk_include_partial_messages: bool = Field(
        default=True,
        description="Stream SDK responses token-by-token instead of in "
        "one lump at the end.  Set to False if the SDK path starts "
        "double-writing text or dropping the tail of long messages.",
    )
    sdk_reconcile_openrouter_cost: bool = Field(
        default=True,
        description="Query OpenRouter's ``/api/v1/generation?id=`` after each "
        "SDK turn and record the authoritative ``total_cost`` instead of the "
        "Claude Agent SDK CLI's estimate.  Covers every OpenRouter-routed "
        "SDK turn regardless of vendor — the CLI's static Anthropic pricing "
        "table is accurate for Anthropic models (Sonnet/Opus via OpenRouter "
        "bill at Anthropic's own rates, penny-for-penny), but the reconcile "
        "catches any future rate change the CLI hasn't picked up and makes "
        "non-Anthropic cost (Kimi et al) correct — real billed amount, "
        "matching the baseline path's ``usage.cost`` read since #12864.  "
        "Kill-switch for emergencies: set ``CHAT_SDK_RECONCILE_OPENROUTER_COST"
        "=false`` to fall back to the CLI's ``total_cost_usd`` reported "
        "synchronously (accurate-for-Anthropic / over-billed-for-Kimi).  "
        "Tradeoff: 0.5-2s window between turn end and cost write; rate-limit "
        "counter briefly unaware, back-to-back turns in that window see "
        "stale state.  The alternative (writing an estimate sync then a "
        "correction delta) would double-count the rate limit.",
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
        """True when OpenRouter config is shape-valid (flag + credentials).

        Indicates whether OpenRouter settings are present and usable —
        ``use_openrouter`` set, plus ``api_key`` + a valid ``base_url``,
        mirroring the fallback logic in ``build_sdk_env``.

        Note: this is a **config-shape check only**.  Runtime SDK routing
        is governed by ``effective_transport`` — subscription mode
        bypasses OpenRouter entirely even when these fields are set, so
        callers asking "will the SDK actually route through OpenRouter
        for this turn?" should use ``effective_transport`` instead.
        """
        if not self.use_openrouter:
            return False
        base = (self.base_url or "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return bool(self.api_key and base and base.startswith("http"))

    @property
    def effective_transport(
        self,
    ) -> Literal["subscription", "openrouter", "direct_anthropic"]:
        """The transport the SDK CLI subprocess actually uses for this turn.

        Detection order:

        1. ``subscription`` — when ``use_claude_code_subscription`` is True
           the CLI uses OAuth from the keychain or
           ``CLAUDE_CODE_OAUTH_TOKEN`` and ignores ``CHAT_BASE_URL`` /
           ``CHAT_API_KEY`` entirely (see ``build_sdk_env`` mode 1).
        2. ``openrouter`` — when ``openrouter_active`` (use_openrouter +
           api_key + a valid base_url).
        3. ``direct_anthropic`` — fallback (CLI talks to api.anthropic.com
           with ``ANTHROPIC_API_KEY`` from parent env).

        Use this when the question is "which model-name format will the
        CLI accept?" — the OpenRouter slug ``anthropic/claude-opus-4.7``
        works through the proxy but is rejected by the subscription /
        direct-Anthropic transports.
        """
        if self.use_claude_code_subscription:
            return "subscription"
        if self.openrouter_active:
            return "openrouter"
        return "direct_anthropic"

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

    @model_validator(mode="after")
    def _validate_sdk_model_vendor_compatibility(self) -> "ChatConfig":
        """Fail at config load when an SDK model slug is incompatible with
        explicit direct-Anthropic mode.

        The SDK path's ``_normalize_model_name`` raises ``ValueError`` when
        a non-Anthropic vendor slug (e.g. ``moonshotai/kimi-k2.6``) is paired
        with direct-Anthropic mode — but that fires inside the request loop,
        so a misconfigured deployment would surface a 500 to every user
        instead of failing visibly at boot.

        Only the **explicit** opt-out (``use_openrouter=False``) is checked
        here, not the credential-missing path.  Build environments and
        OpenAPI-schema export jobs construct ``ChatConfig()`` without any
        OpenRouter credentials in the env — that's not a misconfiguration,
        it's "config loads ok, but no SDK turn will succeed until creds are
        wired".  The runtime guard in ``_normalize_model_name`` still
        catches the credential-missing path on the first SDK turn.

        Covers all three SDK fields that flow through
        ``_normalize_model_name``: primary tier
        (``thinking_standard_model``), advanced tier
        (``thinking_advanced_model``), and fallback model
        (``claude_agent_fallback_model`` via ``_resolve_fallback_model``).

        Skipped when ``use_claude_code_subscription=True`` because the
        subscription path normally resolves the static config to ``None``
        (CLI default). An LD-served override under subscription does
        flow through ``_normalize_model_name``; the runtime guard first
        falls back to the tier default, and only avoids a request error
        when that default is itself valid (otherwise the original LD
        ValueError is re-raised — see ``_resolve_sdk_model_for_request``).
        Empty fallback strings are also skipped (no fallback configured).
        """
        if self.use_claude_code_subscription:
            return self
        if self.use_openrouter:
            return self
        for field_name in (
            "thinking_standard_model",
            "thinking_advanced_model",
            "claude_agent_fallback_model",
        ):
            value: str = getattr(self, field_name)
            if not value or "/" not in value:
                continue
            if value.split("/", 1)[0] != "anthropic":
                raise ValueError(
                    f"Direct-Anthropic mode (use_openrouter=False) "
                    f"requires an Anthropic model for {field_name}, got "
                    f"{value!r}. Set CHAT_THINKING_STANDARD_MODEL / "
                    f"CHAT_THINKING_ADVANCED_MODEL / "
                    f"CHAT_CLAUDE_AGENT_FALLBACK_MODEL to an anthropic/* "
                    f"slug, or set CHAT_USE_OPENROUTER=true."
                )
        return self

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
        # Accept both the Python attribute name and the validation_alias when
        # constructing a ``ChatConfig`` directly (e.g. in tests passing
        # ``thinking_standard_model=...``).  Without this, pydantic only
        # accepts the alias names (``CHAT_THINKING_STANDARD_MODEL`` env) and
        # rejects field-name kwargs — breaking ``ChatConfig(field=...)`` in
        # every test that constructs a config.
        populate_by_name = True
