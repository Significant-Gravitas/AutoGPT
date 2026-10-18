"""The route one launch runs on: a row of ``route_table.py`` made concrete.

A turn builds its profile once, after its route, model and credential are
known; retries and restarts reuse it. The profile holds every route-specific
value the launch needs (the model as the route writes it, the window and
trigger, the credential and URL, thinking), and ``sdk/env.py`` turns it into
the CLI's environment. The orchestrator block builds one from a user's own key
the same way (:func:`user_key_profile`).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr

from backend.copilot.config import ChatConfig
from backend.copilot.moonshot import is_moonshot_model, moonshot_context_window
from backend.copilot.route_table import (
    CODEX_DEFAULT_TRIGGER_PCT,
    CODEX_DEFAULT_WINDOW,
    OPENROUTER_ROOT_URL,
    ROUTES,
    GatewayAdapter,
    RouteRow,
    RoutingSource,
)
from backend.copilot.sdk.context_window import (
    CodexEngineWindow,
    cli_autocompact_threshold,
    compaction_target_tokens,
    retry_target_tokens,
    seed_target_tokens,
)
from backend.integrations.codex.models import CodexReasoningEffort

ThinkingEffort = Literal["low", "medium", "high", "max"]
# The Anthropic routes' thinking: adaptive, at high effort, with the legacy
# token budget for models that still read one.
_THINKING_EFFORT: ThinkingEffort = "high"
THINKING_BUDGET_TOKENS = 8_192
# Sonnet 5's tokenizer counts ~30% more tokens than 4.x for the same text;
# scaling its trigger keeps the compaction point at the same amount of text.
_SONNET_5_TOKENIZER_INFLATION = 1.3
# The CLI's own trigger ceiling is ~93% of the window.
_MAX_TRIGGER_PCT = 90
# Where a Moonshot SKU's real window is unknown.
_MOONSHOT_FALLBACK_WINDOW = 200_000


class RouteProfile(BaseModel):
    """Everything route-specific about one launch. Never holds a secret in
    a form that prints."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    row: RouteRow
    model: str
    routing_source: RoutingSource
    window: int
    # None: the CLI keeps its own trigger.
    trigger_pct: int | None
    # Where a Codex window came from: the account's model list or the default.
    window_source: str | None = None
    # The CLI's ANTHROPIC_BASE_URL; empty is Anthropic's own API.
    base_url: str = ""
    api_key: SecretStr | None = None
    local_adapter: GatewayAdapter | None = None
    codex_effort: CodexReasoningEffort | None = None
    codex_engine: CodexEngineWindow | None = None

    @property
    def name(self) -> str:
        return self.row.name

    @property
    def platform_pays(self) -> bool:
        """Whether the platform's own account is billed for this route."""
        return self.row.payer == "platform"

    @property
    def prices_tokens(self) -> bool:
        """Whether a rate card prices this route's tokens: Codex has none, and
        neither does the operator's own model server."""
        return self.row.cost_source != "codex_gateway" and self.row.payer != "self_host"

    @property
    def gateway_adapter(self) -> GatewayAdapter | None:
        """The adapter of the loopback gateway this route launches behind."""
        return self.row.gateway_adapter or self.local_adapter

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        return _THINKING_EFFORT if self.row.anthropic_thinking else None

    @property
    def thinking_budget(self) -> int | None:
        return THINKING_BUDGET_TOKENS if self.row.anthropic_thinking else None

    @property
    def pins_max_context(self) -> bool:
        """Whether the CLI's own model-size assumption is raised to the window:
        on a route whose model names the CLI does not know, its assumption
        would otherwise clamp the window away."""
        return self.gateway_adapter is not None

    def compaction_target_tokens(self) -> int:
        return compaction_target_tokens(self.window, self.trigger_pct)

    def retry_target_tokens(self) -> tuple[int, int]:
        return retry_target_tokens(self.window)

    def seed_target_tokens(self) -> int:
        return seed_target_tokens(self.window)

    def cli_autocompact_threshold(self) -> int:
        return cli_autocompact_threshold(self.window, self.trigger_pct)


def normalize_model(row: RouteRow, raw: str) -> str:
    """*raw* as the route sends it.

    OpenRouter and a local server take names as configured (``vendor/model``,
    a server's own tags). The Anthropic routes take native Claude ids: the
    ``anthropic/`` prefix goes and dotted versions become dashed; any other
    vendor is refused here rather than as an opaque error from the API.
    """
    if row.keeps_model_names:
        return raw
    model = raw
    if "/" in model:
        vendor, model = model.split("/", 1)
        if vendor != "anthropic":
            raise ValueError(
                f"The {row.name} route serves Anthropic models only, got "
                f"{raw!r}. Set CHAT_STANDARD_MODEL and CHAT_ADVANCED_MODEL to "
                "anthropic/* or claude-* models, or use CHAT_PLATFORM_ROUTE="
                "openrouter."
            )
    elif not model.startswith("claude-"):
        raise ValueError(
            f"The {row.name} route serves Anthropic models only, got {raw!r}. "
            "Use an anthropic/* or claude-* model."
        )
    return model.replace(".", "-")


def route_window(
    row: RouteRow,
    model: str | None,
    *,
    local_window: int | None = None,
    codex_engine: CodexEngineWindow | None = None,
) -> int:
    """The window the CLI is held to on this route for *model*.

    A Moonshot SKU on OpenRouter is held to its own window too: a pin past the
    point the provider refuses the turn would put the trigger where it never
    fires. The Anthropic routes take the row's window as given; the CLI clamps
    it to its own model table.
    """
    if row.name == "codex":
        if codex_engine is not None:
            return codex_engine.context_window
        return CODEX_DEFAULT_WINDOW
    if row.window is None:
        if local_window is None:
            raise ValueError(f"The {row.name} route needs its model's window")
        return local_window
    if row.keeps_model_names and is_moonshot_model(model):
        return min(
            row.window, moonshot_context_window(model) or _MOONSHOT_FALLBACK_WINDOW
        )
    return row.window


def route_trigger_pct(
    row: RouteRow,
    model: str | None,
    *,
    codex_engine: CodexEngineWindow | None = None,
) -> int | None:
    """The compaction trigger on this route, as a percentage of its window.

    Codex follows what the account advertises for the model, else codex-rs's
    own 90%. A Moonshot SKU keeps the CLI's trigger: its endpoint writes no
    cache, so there is no cache cost to cap, and an aggressive trigger
    cascades into repeated compactions.
    """
    if row.name == "codex":
        advertised = codex_engine.trigger_pct if codex_engine is not None else None
        return advertised if advertised is not None else CODEX_DEFAULT_TRIGGER_PCT
    if row.keeps_model_names and is_moonshot_model(model):
        return None
    pct = row.trigger_pct
    if row.sonnet_5_trigger_scaling and model and "claude-sonnet-5" in model:
        pct = min(round(pct * _SONNET_5_TOKENIZER_INFLATION), _MAX_TRIGGER_PCT)
    return pct


def platform_profile(
    config: ChatConfig, model: str, routing_source: RoutingSource
) -> RouteProfile:
    """The operator's platform route, for a turn on *model*."""
    row = config.route_row
    if row is None:
        raise RuntimeError(
            "The platform's chat model is turned off (CHAT_PLATFORM_ROUTE="
            "disabled); connect an account to chat."
        )
    base_url, api_key = _route_endpoint(config, row)
    return RouteProfile(
        row=row,
        model=model,
        routing_source=routing_source,
        window=route_window(row, model, local_window=config.local_context_tokens),
        trigger_pct=route_trigger_pct(row, model),
        base_url=base_url,
        api_key=SecretStr(api_key) if api_key else None,
        local_adapter=config.local_adapter if row.name == "local" else None,
    )


def codex_profile(
    model: str,
    *,
    effort: CodexReasoningEffort | None,
    routing_source: RoutingSource,
    codex_engine: CodexEngineWindow | None,
) -> RouteProfile:
    """A connected ChatGPT account's route, for a turn on *model*."""
    row = ROUTES["codex"]
    return RouteProfile(
        row=row,
        model=model,
        routing_source=routing_source,
        window=route_window(row, model, codex_engine=codex_engine),
        trigger_pct=route_trigger_pct(row, model, codex_engine=codex_engine),
        window_source="account" if codex_engine is not None else "engine-default",
        codex_effort=effort,
        codex_engine=codex_engine,
    )


def user_key_profile(provider: str, api_key: str, model: str | None) -> RouteProfile:
    """A launch on a user's own key: OpenRouter's key goes to OpenRouter and
    any other to Anthropic, never to a URL the deployment configured for its
    own chat."""
    if provider == "open_router":
        row, base_url = ROUTES["openrouter"], OPENROUTER_ROOT_URL
    else:
        row, base_url = ROUTES["direct_anthropic"], ""
    return RouteProfile(
        row=row,
        model=model or "",
        routing_source="env",
        window=route_window(row, model),
        trigger_pct=route_trigger_pct(row, model),
        base_url=base_url,
        api_key=SecretStr(api_key) if api_key else None,
    )


def _route_endpoint(config: ChatConfig, row: RouteRow) -> tuple[str, str | None]:
    """``(base_url, api_key)`` the CLI is given on the platform route."""
    if row.name == "openrouter":
        return _without_v1(config.openrouter_base_url), config.openrouter_api_key
    if row.name == "direct_anthropic":
        return _without_v1(config.anthropic_base_url or ""), config.anthropic_api_key
    if row.name == "local":
        return config.local_base_url or "", config.local_api_key
    return "", None


def _without_v1(url: str) -> str:
    base = url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base
