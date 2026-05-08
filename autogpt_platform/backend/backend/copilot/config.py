"""Configuration management for chat system."""

import os
from typing import Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from backend.util.clients import OPENROUTER_BASE_URL


def _host_matches(base_url: str | None, suffix: str) -> bool:
    """True when ``base_url``'s parsed hostname equals or ends with
    ``.suffix``.

    Substring checks (``"anthropic.com" in base_url``) are flagged by
    CodeQL as incomplete URL sanitization — an attacker-controlled URL
    like ``https://evil.com/anthropic.com/x`` would match.  Parse the
    URL and compare the hostname so only the actual host is considered.
    """
    if not base_url:
        return False
    host = (urlparse(base_url).hostname or "").lower()
    suffix = suffix.lower()
    return host == suffix or host.endswith("." + suffix)


# Anthropic's OpenAI-compatible endpoint. Used by the baseline path when
# ``use_openrouter=False`` so the OpenAI SDK stays in place but talks
# directly to api.anthropic.com instead of going through OpenRouter.
ANTHROPIC_OPENAI_COMPAT_BASE_URL = "https://api.anthropic.com/v1/"

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
        default="anthropic/claude-haiku-4-5",
        description="Model to use for generating session titles (should be "
        "fast/cheap). Default is Anthropic Haiku so direct-Anthropic "
        "deployments (``CHAT_USE_OPENROUTER=false``) can route the title "
        "call through the same client without 404-ing on a non-Anthropic "
        "vendor prefix.  OpenRouter deployments can override to a cheaper "
        "non-Anthropic alternative (e.g. ``CHAT_TITLE_MODEL=openai/gpt-4o-mini``) "
        "via env without code changes.",
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

    # Auxiliary client credentials — used for non-Anthropic models (title
    # generation, simulator, builder helpers).  Kept independent of the
    # main client so flipping ``use_openrouter=False`` (main → direct
    # Anthropic) does not break aux calls that need OpenAI / Google / etc.
    # via OpenRouter.  Default to OpenRouter; fall back to the main
    # ``api_key`` / ``base_url`` when unset (preserves current behaviour
    # for deployments that haven't split the keys yet).
    aux_api_key: str | None = Field(
        default=None,
        description="API key for auxiliary models (title, builder helpers). "
        "Kept separate from ``api_key`` so direct-Anthropic main mode does not "
        "break non-Anthropic aux models. Falls back to OPEN_ROUTER_API_KEY / "
        "``api_key`` when unset.",
    )
    aux_base_url: str | None = Field(
        default=None,
        description="Base URL for auxiliary models. Falls back to ``base_url`` "
        "when unset (i.e. OpenRouter).",
    )

    direct_anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for direct mode (use_openrouter=False). "
        "Used by the baseline OpenAI-compat client when pointed at "
        "api.anthropic.com. Falls back to ANTHROPIC_API_KEY env var.",
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
        description="Max cost per day in microdollars, resets at midnight UTC. "
        "0 means no spend allowed (will block); there is no unlimited tier.",
    )
    weekly_cost_limit_microdollars: int = Field(
        default=5_000_000,
        description="Max cost per week in microdollars, resets Monday 00:00 UTC. "
        "0 means no spend allowed (will block); there is no unlimited tier.",
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
        default=8192,
        ge=0,
        le=128000,
        description="Maximum thinking/reasoning tokens per LLM call. Applies "
        "to both the Claude Agent SDK path (as ``max_thinking_tokens``) and "
        "the baseline path (as ``extra_body.reasoning.max_tokens`` on "
        "OpenRouter Anthropic routes, and as ``extra_body.thinking.budget_tokens`` "
        "on direct-Anthropic OpenAI-compat routes — the OAI-compat schema has "
        "no ``effort`` equivalent so this remains the only knob there). "
        "Extended thinking on Opus can generate 50k+ tokens at $75/M — capping "
        "this is the single biggest cost lever. 8192 is sufficient for most "
        "tasks; increase for complex reasoning. Set to 0 to disable extended "
        "thinking on both paths (kill switch): baseline skips the ``reasoning`` "
        "extra_body; SDK omits the ``max_thinking_tokens`` kwarg so the CLI "
        "falls back to model default (which, without the flag, leaves "
        "extended thinking off). On the SDK path with Claude 4.7+, prefer "
        "``claude_agent_thinking_effort`` for adaptive control — the SDK "
        "ignores ``max_thinking_tokens`` for those models.",
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
        default=False,
        description="Route copilot LLM calls through the OpenRouter proxy. "
        "Default is ``False``: main path goes direct to api.anthropic.com "
        "via the OpenAI-compat endpoint (requires ``CHAT_DIRECT_ANTHROPIC_API_KEY`` "
        "or ``ANTHROPIC_API_KEY``).  Set ``CHAT_USE_OPENROUTER=true`` (with "
        "``CHAT_API_KEY`` + ``CHAT_BASE_URL=https://openrouter.ai/api/v1``) "
        "to keep using OpenRouter.  The actual decision also requires the "
        "credentials to be valid — use the ``openrouter_active`` property "
        "for the final answer.",
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
    def main_client_credentials(self) -> tuple[str | None, str | None]:
        """``(api_key, base_url)`` for the main OpenAI-compatible client.

        Gated on ``openrouter_active`` (use_openrouter + valid creds), not
        the raw flag, so the baseline path matches the SDK's
        ``effective_transport`` behaviour: when ``CHAT_USE_OPENROUTER=true``
        but the OR creds aren't actually present, both paths fall back to
        direct Anthropic instead of attempting OR with no key.

        - **OpenRouter active**: returns ``(api_key, base_url)`` — the
          existing OR creds.
        - **Otherwise** (direct mode or OR misconfigured): returns
          ``(direct_anthropic_api_key, ANTHROPIC_OPENAI_COMPAT_BASE_URL)``
          so the baseline OpenAI-compat client talks straight to
          api.anthropic.com.
        """
        if self.openrouter_active:
            return self.api_key, self.base_url
        return self.direct_anthropic_api_key, ANTHROPIC_OPENAI_COMPAT_BASE_URL

    @property
    def aux_client_credentials(self) -> tuple[str | None, str | None]:
        """``(api_key, base_url)`` for the auxiliary client.

        Auxiliary calls (title generation, builder helpers) use this
        client.  Defaults to OpenRouter; can be split from the main
        client via ``CHAT_AUX_API_KEY`` / ``CHAT_AUX_BASE_URL`` so
        flipping main to direct-Anthropic does not break non-Anthropic
        aux models like ``openai/gpt-4o-mini``.

        Resolution order:

        1. **Both aux env vars unset** — fall back to
           ``main_client_credentials``.  In a single-key direct-Anthropic
           deployment this routes aux through Anthropic (the title model
           must be Anthropic too — boot validator catches mismatches);
           in a single-key OpenRouter deployment it routes through OR
           with the existing OR key.
        2. **At least one aux env var set** — use the resolved aux
           values, falling back to the raw ``api_key`` / ``base_url``
           for whichever wasn't set.  This is the explicit-split flow.

        Step 1 matters for the direct-Anthropic case where the raw
        ``api_key`` is None and ``base_url`` defaults to OpenRouter:
        without this, aux would silently get
        ``(None, https://openrouter.ai/api/v1)`` and 401 every call.
        """
        if self.aux_api_key is None and self.aux_base_url is None:
            return self.main_client_credentials
        api_key = self.aux_api_key or self.api_key
        base_url = self.aux_base_url or self.base_url
        return api_key, base_url

    @property
    def aux_uses_openrouter(self) -> bool:
        """True when the aux client is pointed at OpenRouter.

        Used to gate OR-specific request fields (extra_body keys like
        ``usage.include`` and PostHog tracing) at the call site.
        """
        _, base_url = self.aux_client_credentials
        return _host_matches(base_url, "openrouter.ai")

    @property
    def aux_provider_label(self) -> str:
        """Cost-log ``provider`` label tracking the aux client's actual
        transport.

        Three buckets:

        - ``"open_router"`` — base URL points at OpenRouter.
        - ``"anthropic"`` — base URL points at api.anthropic.com.  A
          single-key direct-Anthropic deployment falls into this case
          when ``aux_*`` is unset, because ``aux_client_credentials``
          inherits from the (Anthropic-pointed) main creds.
        - ``"openai"`` — anything else (custom OAI-compat endpoint,
          plain api.openai.com, ...).
        """
        _, base_url = self.aux_client_credentials
        if not base_url:
            return "openai"
        if _host_matches(base_url, "openrouter.ai"):
            return "open_router"
        if _host_matches(base_url, "anthropic.com"):
            return "anthropic"
        return "openai"

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

    @field_validator("aux_api_key", mode="before")
    @classmethod
    def get_aux_api_key(cls, v):
        """Auxiliary API key — explicit ``CHAT_AUX_API_KEY`` only.

        Deliberately does NOT fall back to ``OPEN_ROUTER_API_KEY`` like
        ``api_key`` does.  An explicit aux key signals "I'm splitting
        the aux client from main"; an env-pulled OR key would silently
        force aux to OR even in direct-Anthropic deployments where a
        leftover ``OPEN_ROUTER_API_KEY`` happens to be in the env —
        producing OR-key-with-Anthropic-URL 401s on every title call.

        When unset, ``aux_client_credentials`` inherits
        ``main_client_credentials`` (which itself reads
        ``OPEN_ROUTER_API_KEY`` for OR mode), so single-key
        deployments keep working unchanged.
        """
        if not v:
            v = os.getenv("CHAT_AUX_API_KEY")
        return v

    @field_validator("aux_base_url", mode="before")
    @classmethod
    def get_aux_base_url(cls, v):
        """Auxiliary base URL — defaults to OpenRouter."""
        if not v:
            v = os.getenv("CHAT_AUX_BASE_URL")
        return v

    @field_validator("direct_anthropic_api_key", mode="before")
    @classmethod
    def get_direct_anthropic_api_key(cls, v):
        """Anthropic API key for direct mode.

        Reads ``CHAT_DIRECT_ANTHROPIC_API_KEY`` first (Pydantic prefix),
        then plain ``ANTHROPIC_API_KEY`` so the same env var the SDK CLI
        already consumes also drives the baseline OpenAI-compat client.
        """
        if not v:
            v = os.getenv("CHAT_DIRECT_ANTHROPIC_API_KEY") or os.getenv(
                "ANTHROPIC_API_KEY"
            )
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

        Covers every model field that flows through
        ``normalize_model_for_transport``: SDK tiers
        (``thinking_standard_model``, ``thinking_advanced_model``),
        baseline tiers (``fast_standard_model``,
        ``fast_advanced_model``), and the SDK fallback
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
            "fast_standard_model",
            "fast_advanced_model",
            "claude_agent_fallback_model",
        ):
            value: str = getattr(self, field_name)
            if not value:
                continue
            if "/" in value:
                if value.split("/", 1)[0] != "anthropic":
                    raise ValueError(
                        f"Direct-Anthropic mode (use_openrouter=False) "
                        f"requires an Anthropic model for {field_name}, got "
                        f"{value!r}. Set CHAT_THINKING_STANDARD_MODEL / "
                        f"CHAT_THINKING_ADVANCED_MODEL / "
                        f"CHAT_FAST_STANDARD_MODEL / "
                        f"CHAT_FAST_ADVANCED_MODEL / "
                        f"CHAT_CLAUDE_AGENT_FALLBACK_MODEL to an "
                        f"``anthropic/*`` or ``claude-*`` slug, or set "
                        f"CHAT_USE_OPENROUTER=true."
                    )
            elif not value.startswith("claude-"):
                # Bare slug must be ``claude-*`` to be valid for direct
                # Anthropic — bare ``gpt-4o-mini`` would otherwise pass
                # the ``"/" not in value`` short-circuit and fail at
                # request time (``normalize_model_for_transport`` raises).
                raise ValueError(
                    f"Direct-Anthropic mode (use_openrouter=False) "
                    f"requires an Anthropic model slug for {field_name}, "
                    f"got {value!r}. Use an ``anthropic/*`` or "
                    f"``claude-*`` slug, or set CHAT_USE_OPENROUTER=true."
                )
        return self

    @model_validator(mode="after")
    def _validate_aux_client_for_direct_main(self) -> "ChatConfig":
        """Fail at boot when the resolved aux client + ``title_model``
        combination would 401 / 404 on every title call.

        The validator is **transport-driven**, not gated on the main
        ``use_openrouter`` flag.  Three observed traps:

        1. ``aux_base_url`` set with no resolvable api key — the aux
           client would route to that URL with no creds and 401 on
           every title call regardless of title model.  Catches the
           operator-typo case (set ``CHAT_AUX_BASE_URL`` but forgot
           ``CHAT_AUX_API_KEY``).
        2. Non-Anthropic ``title_model`` with the aux client landing
           on the Anthropic OpenAI-compat endpoint (``aux_base_url``
           explicitly set to api.anthropic.com, OR aux falls back to
           a direct-Anthropic main).  Anthropic 404s ``gpt-4o-mini``.
        3. Subscription mode (``use_claude_code_subscription=True``) +
           ``use_openrouter=False`` + neither aux creds nor
           ``direct_anthropic_api_key`` — the SDK CLI uses OAuth so
           direct creds are optional for it, but the **aux** client
           still runs the baseline OpenAI-compat path and inherits
           ``(None, api.anthropic.com)`` from main, 401-ing every call.

        Skipped when the resolved aux transport is OpenRouter — OR can
        serve any vendor prefix.  Also skipped on a fully-empty config
        (no main creds + no aux creds + no subscription) so build /
        OpenAPI-schema-export environments that construct
        ``ChatConfig()`` without env vars don't fail.
        """
        # Empty-config escape hatch: build / openapi-export / pytest-
        # collection environments construct ``ChatConfig()`` without any
        # creds.  In that shape ``main_client_credentials`` returns
        # ``(None, api.anthropic.com)`` (because ``openrouter_active``
        # requires a key) and aux inherits the same — so
        # ``aux_uses_openrouter`` is False even though the operator
        # never opted in to direct-Anthropic.  Skip the title-model
        # check when no creds are wired and aux/direct are untouched;
        # real-request credential errors still surface as 401s
        # downstream via the title-cost log path.
        if (
            not self.use_claude_code_subscription
            and not self.api_key
            and not self.aux_api_key
            and not self.direct_anthropic_api_key
            and not self.aux_base_url
        ):
            return self
        # An explicit ``aux_base_url`` without a resolvable key fails
        # fast regardless of which transport that URL points at — the
        # aux client would route to that URL with ``(None, aux_base_url)``
        # and 401 every call.  Catches the operator-typo case where
        # ``CHAT_AUX_BASE_URL`` is set but ``CHAT_AUX_API_KEY`` was
        # forgotten (and ``CHAT_API_KEY`` is also absent).
        if self.aux_base_url and not (self.aux_api_key or self.api_key):
            raise ValueError(
                "CHAT_AUX_BASE_URL is set but no CHAT_AUX_API_KEY (and "
                "no fallback CHAT_API_KEY) — the aux client would 401 "
                "every title call.  Either unset CHAT_AUX_BASE_URL to "
                "inherit the main client, or set CHAT_AUX_API_KEY."
            )
        # Fast-path: aux's resolved transport is OpenRouter — OR serves
        # any vendor prefix so the title-model check doesn't apply.
        if self.aux_uses_openrouter:
            return self
        # Subscription mode trap: SDK uses OAuth so direct creds are
        # optional for it, but the aux client still runs the baseline
        # OpenAI-compat path.  When use_openrouter=False AND no aux
        # creds AND no direct_anthropic_api_key, aux inherits
        # ``(None, api.anthropic.com)`` from main and 401s every title
        # call.  (Skipped when use_openrouter=True because that would
        # have hit the aux_uses_openrouter fast-path above.)
        if (
            self.use_claude_code_subscription
            and not self.use_openrouter
            and self.aux_api_key is None
            and self.aux_base_url is None
            and not self.direct_anthropic_api_key
        ):
            raise ValueError(
                "Subscription mode (CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true) "
                "with CHAT_USE_OPENROUTER=false and no CHAT_AUX_API_KEY: "
                "the aux client (title generation, builder helpers) would "
                "inherit (None, api.anthropic.com) and 401 every call. "
                "Set CHAT_AUX_API_KEY (recommended: route aux through "
                "OpenRouter via CHAT_AUX_API_KEY+CHAT_AUX_BASE_URL), or "
                "set CHAT_DIRECT_ANTHROPIC_API_KEY so aux can reach "
                "Anthropic directly with the title model on Claude."
            )
        # Only ``title_model`` is checked here.  ``simulation_model``
        # uses its own client acquisition path
        # (``backend.util.clients.get_openai_client(prefer_openrouter=True)``)
        # backed by the platform-level OR key — independent of
        # ``ChatConfig`` aux settings — so validating it here would
        # block valid configs that wire the simulator separately.
        title = self.title_model
        if not title:
            return self
        if "/" in title and title.split("/", 1)[0] == "anthropic":
            return self
        # Bare slug must start with ``claude-`` to be valid for direct
        # Anthropic — bare ``gpt-4o-mini`` would otherwise pass the
        # ``"/" not in title`` short-circuit and fail later at request
        # time.  Mirrors the runtime guard in
        # ``normalize_model_for_transport``.
        if "/" not in title and title.startswith("claude-"):
            return self
        raise ValueError(
            f"Aux client resolves to a non-OpenRouter transport but "
            f"title_model={title!r} is non-Anthropic — Anthropic's API "
            f"will 404 the request.  Either set CHAT_AUX_API_KEY + "
            f"CHAT_AUX_BASE_URL=https://openrouter.ai/api/v1 so title "
            f"generation routes through OpenRouter, or override "
            f"CHAT_TITLE_MODEL to an ``anthropic/`` or ``claude-`` slug."
        )

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
