"""Model-slug normalization for the configured transport.

Shared between the SDK (extended-thinking) path and the baseline (fast)
path so a single rule decides whether to keep the OpenRouter
``vendor/model`` slug or strip it for direct-Anthropic / subscription
transports.

Lives in its own module to avoid a circular import — both paths need
this and importing from one into the other would knot up the dependency
graph.
"""

from __future__ import annotations

from backend.copilot.config import ChatConfig


def normalize_model_for_transport(raw_model: str, cfg: ChatConfig | None = None) -> str:
    """Normalize a model name for the **actual** SDK CLI / OpenAI-compat
    transport.

    Three transports (see ``ChatConfig.effective_transport``):

    1. **OpenRouter** — return the prefixed ``vendor/model`` slug
       unchanged (``anthropic/claude-opus-4-6``,
       ``moonshotai/kimi-k2-6``, ...).  Stripping the prefix would break
       routing for non-Anthropic vendors.
    2. **Subscription / Direct Anthropic** — strip the OpenRouter
       ``anthropic/`` prefix and convert dots to hyphens
       (``claude-opus-4.6`` → ``claude-opus-4-6``).  The CLI subprocess
       (subscription mode) and the Anthropic Messages / OpenAI-compat
       APIs reject the prefix and dot-separated versions.  Raises
       ``ValueError`` when a non-Anthropic vendor slug is paired with
       these transports — silently stripping ``moonshotai/`` would send
       ``kimi-k2-6`` to the Anthropic API and produce an opaque
       ``model_not_found`` error far from the misconfiguration source.

    *cfg* is optional so call sites that already hold a per-module
    ``ChatConfig`` reference (e.g. ``copilot.sdk.service.config``) can
    pass it through and keep monkeypatch-based test fixtures working
    against that exact reference.  When omitted, a fresh ``ChatConfig``
    is read from the current environment per call — no module-level
    cache that goes stale when tests mutate env vars.
    """
    config = cfg if cfg is not None else ChatConfig()
    if config.effective_transport == "openrouter":
        return raw_model
    model = raw_model
    if "/" in model:
        vendor, model = model.split("/", 1)
        if vendor != "anthropic":
            raise ValueError(
                f"{config.effective_transport!r} transport requires an "
                f"Anthropic model, got vendor={vendor!r} from "
                f"model={raw_model!r}. Set CHAT_THINKING_STANDARD_MODEL/"
                f"CHAT_THINKING_ADVANCED_MODEL/CHAT_FAST_STANDARD_MODEL/"
                f"CHAT_FAST_ADVANCED_MODEL to an anthropic/* slug, or "
                f"enable OpenRouter."
            )
    elif model and not model.startswith("claude-"):
        # Bare slug missing the ``vendor/`` prefix — only ``claude-*``
        # forms (``claude-sonnet-4-6``, ``claude-3-5-sonnet-20241022``)
        # are accepted by the Anthropic Messages / OpenAI-compat APIs
        # and the CLI subprocess.  Anything else (``gpt-4o-mini``,
        # ``gemini-pro``, ...) would fail with an opaque
        # ``model_not_found`` at request time — surface it here.
        raise ValueError(
            f"{config.effective_transport!r} transport requires an "
            f"Anthropic model slug, got model={raw_model!r}. Use an "
            f"``anthropic/*`` or ``claude-*`` slug, or enable OpenRouter."
        )
    return model.replace(".", "-")
