"""Model names as the platform route sends them.

Shared by the SDK and the fast engine, and by the side calls that run on the
platform route; the rule itself is the route's row (``route_profile.py``).
"""

from __future__ import annotations

from backend.copilot.config import ChatConfig
from backend.copilot.route_profile import normalize_model


def normalize_model_for_transport(raw_model: str, cfg: ChatConfig | None = None) -> str:
    """*raw_model* as the platform route sends it.

    OpenRouter and a local server keep the name as configured; the Anthropic
    routes take native Claude ids and refuse other vendors (``ValueError``).
    With the platform route disabled the name passes through unchanged.

    *cfg* is optional so call sites that already hold a per-module
    ``ChatConfig`` reference can pass it through and keep monkeypatch-based
    test fixtures working against that exact reference.
    """
    config = cfg if cfg is not None else ChatConfig()
    row = config.route_row
    if row is None:
        return raw_model
    return normalize_model(row, raw_model)


def normalize_model_for_anthropic(
    raw_model: str, *, caller: str = "The native Anthropic API"
) -> str:
    """The native Anthropic API's spelling of *raw_model*, whatever the
    chat transport: the ``anthropic/`` prefix stripped and dots turned into
    hyphens (``anthropic/claude-opus-4.6`` → ``claude-opus-4-6``).

    The Anthropic branch of :func:`normalize_model_for_transport`, and the
    whole rule for calls that always reach Anthropic directly — the dream
    batch submit, which runs under the OpenRouter chat transport too, and
    the dream sync path on the ``anthropic`` provider. *caller* names the
    path in the error raised for a non-Anthropic model.
    """
    model = raw_model
    if "/" in model:
        vendor, model = model.split("/", 1)
        if vendor != "anthropic":
            raise ValueError(
                f"{caller} requires an Anthropic model, got vendor={vendor!r} "
                f"from model={raw_model!r}. Set CHAT_STANDARD_MODEL/"
                f"CHAT_ADVANCED_MODEL to an anthropic/* slug, or "
                f"choose the OpenRouter route."
            )
    elif model and not model.startswith("claude-"):
        # Bare slug missing the ``vendor/`` prefix — only ``claude-*``
        # forms (``claude-sonnet-4-6``, ``claude-3-5-sonnet-20241022``)
        # are accepted by the Anthropic Messages / OpenAI-compat APIs
        # and the CLI subprocess.  Anything else (``gpt-4o-mini``,
        # ``gemini-pro``, ...) would fail with an opaque
        # ``model_not_found`` at request time — surface it here.
        raise ValueError(
            f"{caller} requires an Anthropic model slug, got "
            f"model={raw_model!r}. Use an ``anthropic/*`` or ``claude-*`` "
            f"slug, or choose the OpenRouter route."
        )
    return model.replace(".", "-")
