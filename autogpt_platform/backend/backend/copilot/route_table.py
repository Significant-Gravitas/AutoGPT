"""What each chat route does, in code: one row per route.

An operator picks the platform route (``CHAT_PLATFORM_ROUTE``) and gives it a
key, a URL and two model tiers; a user may instead connect a ChatGPT account,
which serves their turns over the ``codex`` row. Everything else about a route
is a row here: the context window and compaction trigger, the CLI timeout,
how model names are written, where a turn's cost comes from and who
pays it, whether the CLI's attribution header and Anthropic thinking go out,
and which gateway adapter carries the turn.

Rows are policy only. ``route_profile.py`` turns a row plus a turn's model,
credential and account limits into the profile the turn runs on.
"""

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from backend.util.llm.providers import ProviderLiteral

PlatformRoute = Literal[
    "openrouter", "direct_anthropic", "local", "subscription", "disabled"
]
RouteName = Literal["openrouter", "direct_anthropic", "local", "subscription", "codex"]
LocalAdapter = Literal["openai_chat", "anthropic_passthrough"]
# The routing layer that picked a turn's model, stamped on its messages.
# Defined here, the lowest layer, so the persistence model, the router and
# the route profile share one Literal. ("fallback" is stamped, never
# routed: it marks an answer from another model than the one asked.)
RoutingSource = Literal[
    "ld",
    "catalog",
    "env",
    "fallback",
    "preferred",
    "account_default",
    "account_available",
]
GatewayAdapter = Literal["openai_chat", "anthropic_passthrough", "codex_responses"]
# Where a turn's charge comes from. ``openrouter_reconcile``: OpenRouter's own
# bill for each generation, read back after the turn. ``rate_card``: the
# platform's price list for the tokens the provider reported. ``codex_gateway``:
# the gateway's token counts, charged to nobody. ``not_charged``: tokens are
# recorded and nobody's budget is charged. The CLI's own estimate is never one.
CostSource = Literal[
    "openrouter_reconcile", "rate_card", "codex_gateway", "not_charged"
]
Payer = Literal["platform", "user_subscription", "self_host"]

OPENROUTER_ROOT_URL = "https://openrouter.ai/api"
CODEX_DEFAULT_WINDOW = 272_000
CODEX_DEFAULT_TRIGGER_PCT = 90
# The CLI keeps the main conversation's prompt cache for five minutes unless
# told otherwise, and the platform no longer tells it otherwise.
CLI_CACHE_TTL = "5m"


class RouteRow(BaseModel):
    """One route's policy. Frozen: the table is a constant."""

    model_config = ConfigDict(frozen=True)

    name: RouteName
    # Tokens; None where the turn supplies it (local: the operator's real
    # window; Codex: what the account advertises, else the default above).
    window: int | None
    trigger_pct: int
    # Sonnet 5 counts ~30% more tokens for the same text, so its trigger is
    # scaled up to compact at the same point of the conversation.
    sonnet_5_trigger_scaling: bool
    # Whether model names are sent as configured (``vendor/model`` on
    # OpenRouter, a local server's own names) or written as native Claude ids.
    keeps_model_names: bool
    standard_model: str | None
    advanced_model: str | None
    cost_source: CostSource
    cost_log_provider: str
    payer: Payer
    attribution_header: bool
    anthropic_thinking: bool
    # Whether the CLI gets a USD ceiling for the query.
    usd_budget: bool
    gateway_adapter: GatewayAdapter | None
    api_timeout_ms: int
    # For the platform's own background calls (dream passes, briefings,
    # consults) on this route. None where the platform has no API of its
    # own: a Claude login, or a connected account.
    dispatch_provider: ProviderLiteral | None
    supports_flex_tier: bool
    # The side-call model for titles and summaries; None follows the standard
    # tier (a local server serves only the models it was given).
    aux_model: str | None
    simulation_model: str | None


DEFAULT_STANDARD_MODEL = "anthropic/claude-sonnet-5-5"
DEFAULT_ADVANCED_MODEL = "anthropic/claude-opus-5-5"
_SONNET = DEFAULT_STANDARD_MODEL
_OPUS = DEFAULT_ADVANCED_MODEL
# The side-call model for titles and summaries: OpenRouter's slug, which the
# Anthropic routes send as ``claude-haiku-5-5``.
_HAIKU = "anthropic/claude-haiku-5.5"
# The direct route's dry-run model stays on Haiku 4.5.
_HAIKU_4_5 = "anthropic/claude-haiku-4-5"
_FLASH_LITE = "google/gemini-2.5-flash-lite"

ROUTES: dict[RouteName, RouteRow] = {
    "openrouter": RouteRow(
        name="openrouter",
        window=200_000,
        trigger_pct=50,
        sonnet_5_trigger_scaling=True,
        keeps_model_names=True,
        standard_model=_SONNET,
        advanced_model=_OPUS,
        cost_source="openrouter_reconcile",
        cost_log_provider="open_router",
        payer="platform",
        attribution_header=False,
        anthropic_thinking=True,
        usd_budget=True,
        gateway_adapter=None,
        api_timeout_ms=600_000,
        dispatch_provider="open_router",
        supports_flex_tier=True,
        aux_model=_HAIKU,
        simulation_model=_FLASH_LITE,
    ),
    "direct_anthropic": RouteRow(
        name="direct_anthropic",
        # The CLI clamps a pin above its own model table, so 1M lets the
        # table decide.
        window=1_000_000,
        trigger_pct=50,
        sonnet_5_trigger_scaling=True,
        keeps_model_names=False,
        standard_model=_SONNET,
        advanced_model=_OPUS,
        cost_source="rate_card",
        cost_log_provider="anthropic",
        payer="platform",
        attribution_header=True,
        anthropic_thinking=True,
        usd_budget=True,
        gateway_adapter=None,
        api_timeout_ms=600_000,
        dispatch_provider="anthropic",
        supports_flex_tier=False,
        aux_model=_HAIKU,
        # Sent as ``claude-haiku-4-5``; Anthropic serves no Gemini.
        simulation_model=_HAIKU_4_5,
    ),
    "subscription": RouteRow(
        name="subscription",
        # Held to 200K: the turns draw on the subscriber's own plan, and a
        # chat resending 700K a turn drains a usage window in a few messages.
        # Within it the route fills the window rather than compacting at half.
        window=200_000,
        trigger_pct=90,
        sonnet_5_trigger_scaling=False,
        keeps_model_names=False,
        standard_model=_SONNET,
        advanced_model=_OPUS,
        cost_source="not_charged",
        cost_log_provider="anthropic",
        payer="user_subscription",
        attribution_header=True,
        anthropic_thinking=True,
        usd_budget=True,
        gateway_adapter=None,
        api_timeout_ms=600_000,
        dispatch_provider=None,
        supports_flex_tier=False,
        aux_model=_HAIKU,
        simulation_model=_FLASH_LITE,
    ),
    "local": RouteRow(
        name="local",
        window=None,
        trigger_pct=90,
        sonnet_5_trigger_scaling=False,
        keeps_model_names=True,
        standard_model=None,
        advanced_model=None,
        cost_source="not_charged",
        cost_log_provider="local",
        payer="self_host",
        attribution_header=False,
        anthropic_thinking=False,
        usd_budget=False,
        gateway_adapter=None,
        api_timeout_ms=1_800_000,
        dispatch_provider="ollama",
        supports_flex_tier=False,
        aux_model=None,
        simulation_model=None,
    ),
    "codex": RouteRow(
        name="codex",
        window=None,
        trigger_pct=CODEX_DEFAULT_TRIGGER_PCT,
        sonnet_5_trigger_scaling=False,
        keeps_model_names=True,
        standard_model="gpt-5.6-terra",
        advanced_model="gpt-6-astra",
        cost_source="codex_gateway",
        cost_log_provider="codex",
        payer="user_subscription",
        attribution_header=False,
        anthropic_thinking=False,
        usd_budget=False,
        gateway_adapter="codex_responses",
        api_timeout_ms=600_000,
        dispatch_provider=None,
        supports_flex_tier=False,
        aux_model=None,
        simulation_model=None,
    ),
}


def platform_row(route: PlatformRoute) -> RouteRow | None:
    """The row the operator's platform route runs on; None when disabled."""
    if route == "disabled":
        return None
    return ROUTES[route]


def is_openrouter_url(url: str) -> bool:
    """Whether *url* is OpenRouter's own API. Only OpenRouter takes its extra
    request fields (usage accounting, Broadcast attribution); another server
    on the OpenRouter route, a proxy or a gateway, rejects fields it does not
    know, so it gets plain requests."""
    host = urlparse(url).hostname or ""
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")
