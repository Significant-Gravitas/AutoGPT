"""Configuration management for chat system.

The platform route and its few settings are below; what each route does is
code (``route_table.py``). Every service checks the chat settings before it
starts (``chat_env.check_chat_environment``), so constructing a ``ChatConfig``
never fails on them: a route value that is not a route reads as ``disabled``
here, and the startup check is the one place that reports it.
"""

import logging
import os
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.copilot.chat_env import parse_local_adapter, parse_platform_route
from backend.copilot.route_table import (
    DEFAULT_ADVANCED_MODEL,
    DEFAULT_STANDARD_MODEL,
    OPENROUTER_ROOT_URL,
    ROUTES,
    LocalAdapter,
    PlatformRoute,
    RouteRow,
    is_openrouter_url,
    platform_row,
)
from backend.util.settings import BehaveAs, Settings

logger = logging.getLogger(__name__)
settings = Settings()

ANTHROPIC_API_URL = "https://api.anthropic.com"
# The dry-run simulator's model where the route names none.
DEFAULT_SIMULATION_MODEL = "google/gemini-2.5-flash-lite"
# The key an OpenAI client needs when a local server takes none.
_LOCAL_PLACEHOLDER_KEY = "local"

CopilotLlmAuthProvider = Literal["platform", "codex", "microsoft_365_copilot"]

# Per-request model tier set by the frontend model toggle: the route's
# standard or advanced model (``ChatConfig.tier_model``). None means no
# preference — falls through to per-user flag targeting, then the tier
# default. Tier names keep the contract model-agnostic.
CopilotLLMModel = Literal["standard", "advanced"]


class ChatConfig(BaseSettings):
    """Configuration for the chat system."""

    platform_route: PlatformRoute = Field(
        default="disabled",
        description="The route the platform's own chat model runs on: "
        "openrouter, direct_anthropic, local, subscription (a self-hosted "
        "install's Claude login), or disabled (no platform model; connected "
        "accounts still work). Required at startup.",
    )
    openrouter_api_key: str | None = Field(
        default=None, description="The openrouter route's key."
    )
    openrouter_base_url: str = Field(
        default=OPENROUTER_ROOT_URL,
        description="OpenRouter's Anthropic-compatible root, without /v1. Side "
        "calls (titles, summaries) use its OpenAI-compatible /v1.",
    )
    anthropic_api_key: str | None = Field(
        default=None, description="The direct_anthropic route's key."
    )
    anthropic_base_url: str | None = Field(
        default=None,
        description="Where the direct_anthropic route sends requests; unset is "
        "Anthropic's API.",
    )
    local_base_url: str | None = Field(
        default=None,
        description="The local route's server: its OpenAI-compatible /v1 URL "
        "for the openai_chat adapter, its root for anthropic_passthrough.",
    )
    local_api_key: str | None = Field(
        default=None,
        description="The local server's key; leave unset when it takes none.",
    )
    local_adapter: LocalAdapter | None = Field(
        default=None,
        description="How the local server is spoken to: openai_chat (Chat "
        "Completions) or anthropic_passthrough (Anthropic Messages). Until the "
        "shared gateway carries the local route, the fast engine uses the "
        "server's OpenAI-compatible /v1 API either way.",
    )
    local_context_tokens: int | None = Field(
        default=None,
        description="The local model's real context window, in tokens, as the "
        "server has it loaded.",
    )
    standard_model: str | None = Field(
        default=None,
        description="The standard tier's model; unset takes the route's default "
        "(``route_table.py``). Required on the local route.",
    )
    advanced_model: str | None = Field(
        default=None,
        description="The advanced tier's model; unset takes the route's default. "
        "Required on the local route.",
    )
    learning_reviewer_model: str = Field(
        default="",
        description="Model for the nightly skill-learning review. Empty (the "
        "default) is the route's standard tier, like the platform's other "
        "background jobs; set CHAT_LEARNING_REVIEWER_MODEL to pin a model "
        "chosen through evaluation.",
    )

    gate_timeout_s: float = Field(
        default=6.0,
        description="Hard timeout for one gate classification. Expiry is not "
        "an error path — it resolves to 'ask'.",
    )
    gate_first_stage: Literal["none", "jev"] = Field(
        default="jev",
        description="First stage of the action supervisor: Jev decides every "
        "judged call and the LLM (``gate/classifier.py``) runs only on an ask, to write "
        "the reason. Off without ``TYPESAFE_API_KEY``.",
    )
    gate_jev_model: str = Field(
        default="jev-1.13.0",
        description="Pinned Jev version. Once TypeSafe retires it, calls run on "
        "``jev-latest`` and log an ERROR naming it.",
    )
    gate_jev_ask_threshold: float | None = Field(
        default=None,
        description="Unset: Jev's allow/ask choice decides. Set: the call also "
        "asks when Jev's must-ask probability reaches it (0.4 was measured).",
    )
    gate_jev_timeout_s: float = Field(
        default=2.0,
        description="Timeout for one Jev call; expiry falls through to the LLM.",
    )
    content_judge_timeout_s: float = Field(
        default=15.0,
        description="Hard timeout for one content-judge call on an outside "
        "read. Provisional until the supervisor measurement sets it from the "
        "judge's p95; expiry holds the read.",
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
        ge=0,
        description="How long a process may serve a cached Langfuse prompt before "
        "re-fetching it (0 to disable caching)",
    )

    # Rate limiting — cost-based limits per day and per week, stored in
    # microdollars (1 USD = 1_000_000).  The counter tracks the real
    # generation cost: the provider's own bill where it reports one
    # (OpenRouter), otherwise the platform's rate card for the tokens the
    # provider reported, never the CLI's estimate. Cache discounts and
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
    #
    # A negative value disables that window's cap. Self-hosted distributions
    # (the single-container image and the unraid template) export -1 for both
    # because the operator pays the model provider directly and, without
    # LaunchDarkly, every account resolves to NO_TIER → BASIC multiplier and
    # would otherwise inherit these cloud ceilings. The defaults here stay
    # positive so a LaunchDarkly outage on cloud never removes the cap.
    daily_cost_limit_microdollars: int = Field(
        default=2_500_000,
        description="Max cost per day in microdollars, resets at midnight UTC. "
        "0 means no spend allowed (will block); a negative value disables the "
        "daily cap (the self-hosted default).",
    )
    weekly_cost_limit_microdollars: int = Field(
        default=5_000_000,
        description="Max cost per week in microdollars, resets Monday 00:00 UTC. "
        "0 means no spend allowed (will block); a negative value disables the "
        "weekly cap (the self-hosted default).",
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
    claude_agent_max_buffer_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB (default SDK is 1MB)
        description="Max buffer size in bytes for Claude Agent SDK JSON message parsing. "
        "Increase if tool outputs exceed the limit.",
    )
    claude_agent_max_subtasks: int = Field(
        default=10,
        description="Max number of concurrent sub-agent Tasks the SDK can run per session.",
    )
    claude_agent_extra_builtin_tools: list[str] = Field(
        default_factory=list,
        description="Claude Code built-in tools to allow and declare on the SDK "
        "path on top of the default set, for an operator or an eval arm, e.g. "
        "CHAT_CLAUDE_AGENT_EXTRA_BUILTIN_TOOLS='[\"Agent\"]'. Names on the SDK "
        "disallowed list stay off; a name that is no Claude Code built-in is "
        "logged once and ignored.",
    )
    agent_max_turns: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of tool-call rounds per turn where no "
        "spend governor runs: the baseline engine and the orchestrator block's "
        "CLI launch. AutoPilot's SDK turns, Codex models included, check in at "
        "``sdk_step_checkin`` instead. Override via CHAT_AGENT_MAX_TURNS.",
    )
    sdk_step_checkin: int = Field(
        default=450,
        ge=1,
        le=10000,
        description="Model calls an SDK turn may make before its tools are "
        "refused for a check-in: the agent writes a progress summary and the "
        "user can choose to keep going, which starts a fresh count. The CLI's "
        "own max_turns sits a little above this as a backstop. Spend, not "
        "steps, is what bounds cost: the governor lands a turn at its budget.",
    )
    claude_agent_max_budget_usd: float = Field(
        default=10.0,
        ge=0.01,
        le=1000.0,
        description="Maximum spend in USD per SDK query — a backstop, not a "
        "target: it sits ~10x above the observed per-TURN p90. Measured over "
        "60 days ending 2026-09-02, one PlatformCostLog row = one turn: "
        "sonnet-5 p50 $0.28 / p75 $0.46 / p90 $0.93 (n=707); "
        "opus-4-8 p50 $0.49 / p90 $1.34 (n=18). Note request < turn < session: "
        "over half of all rows are near-zero utility calls (flash-lite, haiku, "
        "sonar), so unfiltered aggregates understate a real turn. "
        "Override via CHAT_CLAUDE_AGENT_MAX_BUDGET_USD env var.",
    )
    tree_ceiling_fraction_of_daily: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Share of the user's tier-scaled DAILY limit that one tree "
        "(a root turn plus everything it spawns) may consume. Scaling off the "
        "tier rather than a flat number keeps the ceiling proportionate: a "
        "NO_TIER user (multiplier 0.0) gets 0, i.e. no spawns at all.",
    )
    tree_ceiling_floor_microdollars: int = Field(
        default=500_000,
        ge=0,
        description="Lower bound on a tree ceiling, in microdollars ($0.50), so "
        "a small daily limit still affords a tree at least one real turn — "
        "per-SESSION spend is p50 $0.49 / p75 $1.14 / p90 $3.25 (n=319, "
        "60 days ending 2026-09-02).",
    )
    tree_ceiling_microdollars: int = Field(
        default=10_000_000,
        ge=0,
        description="Absolute cap on a tree ceiling, in microdollars ($10.00), "
        "applied after the tier-scaled fraction and the floor. The effective "
        "ceiling is min(remaining budget, max(fraction x tier daily, floor), "
        "this cap). Admission reads settled spend, so a tree can overshoot by "
        "up to (max_nodes - 1) concurrently admitted turns; the node cap is "
        "what bounds it.",
    )
    spend_ceiling_reset: Literal["never", "daily"] = Field(
        default="daily",
        description="When a chat's spend ceiling starts over: at each UTC "
        "midnight, like the daily usage limit (the ceiling and any approved "
        "raises reset with the day), or never within the chat.",
    )
    tree_max_nodes: int = Field(
        default=8,
        ge=1,
        description="Max turns (root included) one root turn may spawn, "
        "counted per tree rather than per node so it is enforceable atomically.",
    )
    tree_budget_signal_enabled: bool = Field(
        default=True,
        description="Prepend a one-line <budget_status> block to every turn's "
        "message, and the wrap-up checkpoint once the tree crosses "
        "``tree_wrapup_threshold``. Off, or on a turn with no tree envelope, "
        "the prompt is byte-identical to what it was before.",
    )
    tree_wrapup_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Share of a tree's ceiling at which the wrap-up checkpoint "
        "is injected, once per tree. Below 1.0 on purpose: the instruction is "
        "only useful while there is still budget to act on it.",
    )
    stream_replay_count: int = Field(
        default=200,
        ge=1,
        le=10000,
        description="Redis stream entries read per replay batch on SSE reconnect.",
    )
    claude_agent_max_transient_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries for transient API errors "
        "(429, 5xx, ECONNRESET) before surfacing the error to the user.",
    )
    claude_agent_live_system_prompt: bool = Field(
        default=True,
        description="Send the system prompt each CLI launch was given, also "
        "when it resumes a session. When False, the CLI records the prompt "
        "and tool descriptions on a session's first request and resends that "
        "record on every later request and resume until the session compacts, "
        "so a changed prompt (the building-mode restart's guide) never reaches "
        "the model. A live prompt re-writes a session's prompt cache once "
        "whenever the prompt differs between two turns; set False to roll "
        "back to the recorded prompt. False also keeps recorded prompt "
        "snapshots in stored transcripts, whatever "
        "claude_agent_strip_prompt_snapshots says, so this one switch "
        "restores the recorded-prompt behaviour.",
    )
    claude_agent_strip_prompt_snapshots: bool = Field(
        default=True,
        description="Drop the CLI's recorded prompt snapshots (the system "
        "prompt and every tool description, 134-180KB per recording) from a "
        "session's transcript when it is stored or restored. A live-prompt "
        "launch never reads them. Applies only while "
        "claude_agent_live_system_prompt is on: a launch with the live prompt "
        "off that finds no snapshot sends its own prompt and records it "
        "again, so the snapshots are kept for it.",
    )
    sdk_include_partial_messages: bool = Field(
        default=True,
        description="Stream SDK responses token-by-token instead of in "
        "one lump at the end.  Set to False if the SDK path starts "
        "double-writing text or dropping the tail of long messages.",
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
        "or the unprefixed `CLAUDE_AGENT_CLI_PATH` environment variable.",
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
        default="agpt-desktop-1x2-004d6e73",
        description="E2B sandbox template for copilot sessions. The default is our "
        "own image (E2B's desktop image at 1 vCPU / 2 GiB, ~$0.08/h running, "
        "no display started), built on the team automatically the first time "
        "it is needed; see backend.util.e2b_template. Any other value is used "
        "as-is and must already exist on the team, and it must carry what the "
        "desktop needs (Xvfb, XFCE, x11vnc and noVNC, as E2B's desktop image "
        "does): the screen is turned on inside this same box, so a plain "
        "image such as 'base' makes every start_desktop fail.",
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

    def __init__(self, **values: Any) -> None:
        # Keyword arguments are code, not operator input: a name that is no
        # setting is a mistake (often a retired one), never silently dropped.
        known = set(type(self).model_fields)
        unknown = sorted(k for k in values if k not in known and not k.startswith("_"))
        if unknown:
            raise TypeError(f"ChatConfig has no setting {', '.join(unknown)}")
        super().__init__(**values)

    @field_validator("platform_route", mode="before")
    @classmethod
    def _unknown_route_reads_as_disabled(cls, v: Any) -> Any:
        return parse_platform_route(v) or "disabled"

    @field_validator("local_adapter", mode="before")
    @classmethod
    def _unknown_adapter_reads_as_unset(cls, v: Any) -> Any:
        return parse_local_adapter(v)

    @field_validator("local_context_tokens", mode="before")
    @classmethod
    def _unreadable_window_reads_as_unset(cls, v: Any) -> Any:
        if isinstance(v, int):
            return v
        text = str(v or "").strip()
        return int(text) if text.isdigit() else None

    @property
    def route_row(self) -> RouteRow | None:
        """The platform route's row; None when the route is disabled."""
        return platform_row(self.platform_route)

    def tier_model(self, tier: CopilotLLMModel) -> str | None:
        """The operator's model for *tier*, else the route's default."""
        configured = self.advanced_model if tier == "advanced" else self.standard_model
        if configured and configured.strip():
            return configured.strip()
        row = self.route_row
        if row is None:
            return None
        return row.advanced_model if tier == "advanced" else row.standard_model

    def job_tier_model(self, tier: CopilotLLMModel) -> str:
        """The tier's model for the platform's own background jobs (dream
        passes): the route's, else the hosted default; a job on a route that
        serves neither fails at its own call."""
        default = (
            DEFAULT_ADVANCED_MODEL if tier == "advanced" else DEFAULT_STANDARD_MODEL
        )
        return self.tier_model(tier) or default

    @property
    def has_platform_api(self) -> bool:
        """Whether the platform route has an API for the platform's own
        background calls (dream passes, briefings, consults). A Claude login
        (subscription) has none, and neither has a disabled route: that work
        skips before any provider call."""
        row = self.route_row
        return row is not None and row.dispatch_provider is not None

    @property
    def batch_anthropic_api_key(self) -> str | None:
        """The key for Anthropic's Batch API on the routes whose dream passes
        use it: the direct route's key, else the deployment's Anthropic
        integration key. The Batch API cannot be reached through OpenRouter,
        so it keeps a key of its own. No other route has one: a Claude login
        and a disabled route have no platform API, and a local server's
        passes stay on that server."""
        if self.platform_route not in ("openrouter", "direct_anthropic"):
            return None
        return self.anthropic_api_key or settings.secrets.anthropic_api_key or None

    @property
    def aux_model(self) -> str | None:
        """The side-call model (titles, summaries, briefings); None when the
        platform route is disabled. On local, the standard tier."""
        row = self.route_row
        if row is None:
            return None
        return row.aux_model or self.tier_model("standard")

    @property
    def simulation_model(self) -> str:
        """The dry-run simulator's model: the route's, or the standard tier on
        a route whose server serves only its own models."""
        row = self.route_row
        if row is not None and row.simulation_model is None:
            return self.tier_model("standard") or DEFAULT_SIMULATION_MODEL
        return (row.simulation_model if row else None) or DEFAULT_SIMULATION_MODEL

    @property
    def local_request_timeout_s(self) -> float:
        """How long one request to the local server may take: a CPU-only
        host can need many minutes for one turn of AutoPilot's prompt."""
        return ROUTES["local"].api_timeout_ms / 1000

    @property
    def local_openai_base_url(self) -> str | None:
        """The local server's OpenAI-compatible base, whichever adapter is set."""
        if not self.local_base_url:
            return None
        base = self.local_base_url.rstrip("/")
        if self.local_adapter == "anthropic_passthrough" and not base.endswith("/v1"):
            return base + "/v1"
        return base

    @property
    def main_client_credentials(self) -> tuple[str | None, str | None]:
        """``(api_key, base_url)`` of the platform route's OpenAI-compatible API.

        ``(None, None)`` on a route with no API (subscription, disabled).
        """
        if self.platform_route == "openrouter":
            return self.openrouter_api_key, self.openrouter_base_url.rstrip("/") + "/v1"
        if self.platform_route == "direct_anthropic":
            base = (self.anthropic_base_url or ANTHROPIC_API_URL).rstrip("/")
            return self.anthropic_api_key, base + "/v1/"
        if self.platform_route == "local":
            return (
                self.local_api_key or _LOCAL_PLACEHOLDER_KEY,
                self.local_openai_base_url,
            )
        return None, None

    @property
    def aux_client_credentials(self) -> tuple[str | None, str | None]:
        """``(api_key, base_url)`` for side calls: the platform route's own API,
        never a connected user's account or the host's Claude login."""
        return self.main_client_credentials

    @property
    def aux_uses_openrouter(self) -> bool:
        """Whether side calls go to OpenRouter itself, which takes its extra
        fields; another server on the OpenRouter route gets plain requests."""
        return self.platform_route == "openrouter" and is_openrouter_url(
            self.openrouter_base_url
        )

    @property
    def aux_provider_label(self) -> str:
        """Cost-log ``provider`` of a side call: the platform route's label."""
        row = self.route_row
        return row.cost_log_provider if row is not None else "none"

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
            v = os.getenv("E2B_API_KEY")
        return v

    @field_validator("claude_agent_cli_path", mode="before")
    @classmethod
    def get_claude_agent_cli_path(cls, v):
        """Resolve the Claude Code CLI override path from environment.

        Accepts either the Pydantic-prefixed ``CHAT_CLAUDE_AGENT_CLI_PATH``
        or the unprefixed ``CLAUDE_AGENT_CLI_PATH``. Keeping the
        unprefixed form working is important because the field is
        primarily an operator escape hatch set via container/host env,
        and the unprefixed name is what the PR description, the field
        docstrings, and the reproduction test in
        ``cli_openrouter_compat_test.py`` refer to.
        """
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
    def _refuse_claude_subscription_on_cloud(self) -> "ChatConfig":
        """A Claude subscription stays on a self-hosted deployment.

        The subscription route carries one person's Claude OAuth token for
        every user of the deployment, so the hosted platform must never run
        it.  Failing at load keeps a stray env var from turning it on.
        """
        if self.platform_route != "subscription":
            return self
        if settings.config.behave_as != BehaveAs.CLOUD:
            return self
        raise ValueError(
            "CHAT_PLATFORM_ROUTE=subscription is not allowed when "
            "BEHAVE_AS=cloud: a Claude subscription may only be used on a "
            "self-hosted deployment."
        )

    # --- Voice mode TTS (see ``copilot/speech.py``) ---
    voice_tts_model: str = Field(
        default="gpt-4o-mini-tts",
        description="OpenAI speech model used for voice-mode replies.",
    )
    voice_tts_voice: str = Field(
        default="marin",
        description="Default OpenAI voice. Must be in ``speech.ALLOWED_VOICES``.",
    )
    voice_tts_instructions: str = Field(
        default=(
            "Speak like a colleague at the next desk: natural pace, warm and "
            "engaged, never flat. Move briskly through comma-separated lists — "
            "items get a light separation, not a pause each. No newsreader "
            "delivery and no slowing down for emphasis."
        ),
        description="Delivery instruction for a spoken reply. gpt-4o-mini-tts "
        "takes its pacing from this rather than from ``speed``.",
    )
    voice_tts_speed: float = Field(
        default=1.2,
        ge=0.25,
        le=4.0,
        description="Playback rate for synthesis. The instruction alone barely "
        "moves comma-heavy listings; this does (12.6s to 10.3s at 1.3).",
    )
    voice_tts_usd_per_1k_chars: float = Field(
        default=0.02,
        ge=0,
        # A negative or non-finite rate yields a cost the usage recorder
        # cannot write — and TTS has no token counts to fall back on, so
        # synthesis would keep running unmetered.
        allow_inf_nan=False,
        description="Metering rate for voice-mode TTS. gpt-4o-mini-tts bills "
        "per audio token, which the request does not report; ~$0.015/min of "
        "speech at ~750 chars/min is the estimate behind this default.",
    )

    model_config = SettingsConfigDict(
        env_prefix="CHAT_",
        env_file=".env",
        env_file_encoding="utf-8",
        # Unknown CHAT_* variables are ignored here; the startup check
        # (``chat_env.py``) refuses the retired ones by name.
        extra="ignore",
    )
