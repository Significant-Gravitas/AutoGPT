"""Map the active ChatConfig transport to ``call_provider`` kwargs.

Single source of truth for any non-chat caller (dream pass, graphiti
memory, future scheduled batch jobs) that needs to dispatch an LLM
call through ``backend.util.llm.providers.call_provider``. Combines
the *static* identity baked into the active ``TransportProfile``
(``dispatch_provider``, ``supports_flex_tier``, ``cost_log_provider``)
with the *runtime* credentials carried by ``ChatConfig`` + ``Settings``
so call sites don't have to dual-read.

Holy-grail design: the per-transport routing table lives on
``TransportProfile`` itself; this module just resolves credentials.
Adding a fifth transport (e.g. ``"vllm"`` or ``"lm_studio"``) means
"add a row to ``_TRANSPORT_PROFILES``" — no edit here required as
long as the resolution falls back through ``CHAT_API_KEY`` / Settings.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

from backend.util.llm.providers import ProviderLiteral

logger = logging.getLogger(__name__)


class ProviderRoutingKwargs(BaseModel):
    """Kwargs to pass through to ``call_provider`` for the active transport.

    ``provider`` / ``api_key`` / ``base_url`` map 1:1 to ``call_provider``
    parameters. ``supports_flex`` and ``cost_log_provider`` are carried
    along as hints for callers that decide *whether* to ask for flex or
    what label to write on a ``PlatformCostLog`` row.

    Returned by ``routing_kwargs_for_chat_transport`` — never construct
    directly; the constructor is plumbing, not an API surface. Frozen
    Pydantic model per the backend architecture rule
    (``util/architecture_test.py::test_backend_uses_pydantic_not_dataclasses``).
    """

    model_config = ConfigDict(frozen=True)

    provider: ProviderLiteral
    api_key: str
    base_url: str | None
    supports_flex: bool
    cost_log_provider: str


def routing_kwargs_for_chat_transport() -> ProviderRoutingKwargs:
    """Resolve the right ``call_provider`` kwargs for the active transport.

    Reads ``ChatConfig.transport`` (the cached singleton in
    ``copilot.sdk.env``) for the static identity, then resolves the
    matching credential field. Lazy imports both ``chat_cfg`` and
    ``Settings`` to avoid the ``copilot.sdk.env`` ↔ ``util.clients``
    import cycle dev's PR #12993 already established.

    Runtime credential resolution:

    - ``local`` — Ollama's bearer token is a placeholder (any non-empty
      string works), so we take it from ``chat_cfg.api_key`` whether
      the operator set it explicitly or not. ``base_url`` is the
      ``CHAT_BASE_URL`` value the operator pointed at the local
      backend (e.g. ``http://localhost:11434/v1``).
    - ``subscription`` / ``direct_anthropic`` — both dispatch through
      the Messages API. Subscription users have an OAuth token that
      can't be used for direct API calls (see
      ``docs/platform/copilot-local-llm.md`` for the rationale + the
      Anthropic Feb-2026 ToS update), so they need
      ``ANTHROPIC_API_KEY`` separately. ``direct_anthropic`` callers
      may have ``chat.direct_anthropic_api_key`` set independently.
    - ``openrouter`` — the canonical cloud path. Pulls from
      ``chat_cfg.api_key`` first (operator may have set it directly),
      then ``settings.secrets.open_router_api_key`` (platform-wide).

    ``api_key`` may be the empty string when no credential is
    configured — callers must surface a friendly error before
    invoking ``call_provider`` (which would otherwise 401 with an
    opaque message). For ``local`` an empty key is fine (Ollama
    doesn't validate it).
    """
    # Lazy import: ``copilot.sdk.env`` instantiates ``ChatConfig`` at
    # import time, which runs the full validator chain; importing it at
    # module scope would force every consumer of this helper into the
    # same boot path even when they only need the dataclass.
    from backend.copilot.sdk.env import config as chat_cfg
    from backend.util.settings import Settings

    settings = Settings()
    transport = chat_cfg.transport

    if transport.name == "local":
        api_key = chat_cfg.api_key or ""
        base_url = chat_cfg.base_url
    elif transport.name in ("subscription", "direct_anthropic"):
        api_key = (
            chat_cfg.direct_anthropic_api_key
            or settings.secrets.anthropic_api_key
            or ""
        )
        base_url = None
    else:  # "openrouter"
        api_key = chat_cfg.api_key or settings.secrets.open_router_api_key or ""
        base_url = None

    return ProviderRoutingKwargs(
        provider=transport.dispatch_provider,
        api_key=api_key,
        base_url=base_url,
        supports_flex=transport.supports_flex_tier,
        cost_log_provider=transport.cost_log_provider,
    )
