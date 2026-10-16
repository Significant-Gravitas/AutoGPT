"""Map the active ChatConfig transport to ``call_provider`` kwargs.

Single source of truth for any non-chat caller (dream pass, graphiti
memory, future scheduled batch jobs) that needs to dispatch an LLM
call through ``backend.util.llm.providers.call_provider``. Combines
the *static* identity on the platform route's row
(``dispatch_provider``, ``supports_flex_tier``, ``cost_log_provider``)
with the *runtime* credentials carried by ``ChatConfig`` + ``Settings``
so call sites don't have to dual-read.

The per-route identity lives on the route's row (``route_table.py``);
this module just resolves credentials.
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

    # None: the platform route has no API for background calls.
    provider: ProviderLiteral | None
    api_key: str
    base_url: str | None
    supports_flex: bool
    cost_log_provider: str


def routing_kwargs_for_chat_transport() -> ProviderRoutingKwargs:
    """Resolve the right ``call_provider`` kwargs for the active transport.

    Reads the platform route's row (``ChatConfig.route_row`` on the cached
    singleton in ``copilot.sdk.env``) for the static identity, then
    resolves the matching credential field. Lazy imports both ``chat_cfg`` and
    ``Settings`` to avoid the ``copilot.sdk.env`` ↔ ``util.clients``
    import cycle dev's PR #12993 already established.

    Each route dispatches with its own key to its own URL, never an
    integration key:

    - ``local`` — the server's key (``CHAT_LOCAL_API_KEY``) or a
      placeholder, which Ollama accepts. ``base_url`` is the server's
      OpenAI-compatible URL (``CHAT_LOCAL_BASE_URL``).
    - ``direct_anthropic`` — ``CHAT_ANTHROPIC_API_KEY``, sent to
      ``CHAT_ANTHROPIC_BASE_URL`` (the Messages API's root).
    - ``openrouter`` — ``CHAT_OPENROUTER_API_KEY``, sent to
      ``CHAT_OPENROUTER_BASE_URL``'s OpenAI-compatible ``/v1``.
    - ``subscription`` and ``disabled`` — no platform API: ``provider``
      is None and callers skip the call. A Claude login cannot call the
      Messages API, and an integration key is never a stand-in for it.

    The Anthropic Batch API keeps its own key policy
    (``ChatConfig.batch_anthropic_api_key``); this is the synchronous
    path only.

    ``api_key`` may be the empty string when the route's key is unset
    (startup refuses that, so only a test or a misconfigured process
    sees it); callers surface a friendly error instead of a 401. For
    ``local`` an empty key is fine (Ollama doesn't validate it).
    """
    # Lazy import: ``copilot.sdk.env`` instantiates ``ChatConfig`` at
    # import time, which runs the full validator chain; importing it at
    # module scope would force every consumer of this helper into the
    # same boot path even when they only need the dataclass.
    from backend.copilot.config import ANTHROPIC_API_URL
    from backend.copilot.sdk.env import config as chat_cfg

    row = chat_cfg.route_row
    if row is None or row.dispatch_provider is None:
        # No platform API: nothing to dispatch on, and callers skip.
        return ProviderRoutingKwargs(
            provider=None,
            api_key="",
            base_url=None,
            supports_flex=False,
            cost_log_provider="none",
        )

    if row.name == "local":
        api_key = chat_cfg.local_api_key or "local"
        base_url = chat_cfg.local_openai_base_url
    elif row.name == "direct_anthropic":
        api_key = chat_cfg.anthropic_api_key or ""
        base_url = (chat_cfg.anthropic_base_url or ANTHROPIC_API_URL).rstrip("/")
    else:  # "openrouter"
        api_key = chat_cfg.openrouter_api_key or ""
        base_url = chat_cfg.openrouter_base_url.rstrip("/") + "/v1"

    return ProviderRoutingKwargs(
        provider=row.dispatch_provider,
        api_key=api_key,
        base_url=base_url,
        supports_flex=row.supports_flex_tier,
        cost_log_provider=row.cost_log_provider,
    )


def background_model_name(raw_model: str) -> str:
    """*raw_model* as the platform route sends it, read from the same settings
    as ``routing_kwargs_for_chat_transport``: native Claude ids on the
    Anthropic routes, as configured on OpenRouter and a local server.
    ``ValueError`` for a model the route cannot serve."""
    from backend.copilot.model_normalize import normalize_model_for_transport
    from backend.copilot.sdk.env import config as chat_cfg

    return normalize_model_for_transport(raw_model, chat_cfg)
