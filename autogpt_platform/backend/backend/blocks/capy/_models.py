"""How a Capy model is paid for: the Capy balance, or a linked provider.

A Capy model ID names both the model and who pays for it. ``openai/gpt-6-astra``
bills the organization's Capy balance; ``codex/gpt-6-astra`` runs the same
model on a ChatGPT subscription linked in Capy's settings. Balance-billed IDs
carry the model vendor's prefix (``anthropic/``, ``meta/``, ``xai/``...), and
Capy does not resolve every bare name, so the balance route adds the prefix.
See https://docs.capy.ai/models-and-pricing.
"""

from enum import Enum

from backend.sdk import SchemaField


class ModelRoute(str, Enum):
    """Who pays for the model: as the model ID says, or a named route."""

    AS_GIVEN = "as_given"
    CAPY_BALANCE = "capy_balance"
    CODEX = "codex"
    COPILOT = "copilot"
    SUPERGROK = "supergrok"
    AZURE = "azure"


# Routes that bill a linked subscription or organization account rather than
# the Capy balance, keyed by their model ID prefix.
LINKED_ROUTES = {
    ModelRoute.CODEX.value: "Codex (ChatGPT subscription)",
    ModelRoute.COPILOT.value: "GitHub Copilot subscription",
    ModelRoute.SUPERGROK.value: "SuperGrok subscription",
    ModelRoute.AZURE.value: "Azure organization account",
}


# Model-name families and the vendor prefix their Capy-balance entries use,
# from the price list at docs.capy.ai/models-and-pricing (2026-09-28).
_BALANCE_VENDORS = (
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("gemini-", "google"),
    ("muse-", "meta"),
    ("grok-", "xai"),
    ("deepseek-", "deepseek"),
    ("longcat-", "meituan"),
    ("minimax-", "minimax"),
    ("kimi-", "moonshotai"),
    ("qwen", "qwen"),
    ("hy", "tencent"),
    ("mimo-", "xiaomi"),
    ("glm-", "zai"),
)


def resolve_model_id(model_id: str, route: ModelRoute) -> str:
    """Rewrite ``model_id`` so it runs on ``route``.

    ``gpt-6-astra`` with the Codex route becomes ``codex/gpt-6-astra``;
    ``codex/gpt-6-astra`` with the Capy balance route becomes
    ``openai/gpt-6-astra``. A family the table doesn't know is sent bare for
    Capy to resolve or reject.
    """
    if not model_id or route == ModelRoute.AS_GIVEN:
        return model_id
    if route == ModelRoute.CAPY_BALANCE:
        return _balance_model_id(model_id)
    name = model_id.split("/", 1)[-1]
    return f"{route.value}/{name}"


def _balance_model_id(model_id: str) -> str:
    if "/" in model_id and not is_linked_route(model_id):
        return model_id
    name = model_id.split("/", 1)[-1]
    vendor = next((v for prefix, v in _BALANCE_VENDORS if name.startswith(prefix)), "")
    return f"{vendor}/{name}" if vendor else name


def billed_via(model_id: str | None) -> str:
    """Name who pays for a model ID, for reporting back to the user."""
    if not model_id:
        return ""
    prefix = model_id.split("/", 1)[0] if "/" in model_id else ""
    return LINKED_ROUTES.get(prefix, "Capy balance")


def is_linked_route(model_id: str) -> bool:
    return "/" in model_id and model_id.split("/", 1)[0] in LINKED_ROUTES


def rejection_hint(rejection: str | None, service: str | None, model_id: str) -> str:
    """Explain a ModelSelection.Rejected answer in terms the user can act on."""
    label = LINKED_ROUTES.get(service or "", service or "")
    if rejection == "disconnected":
        return (
            f"the {label} linked in Capy is disconnected; reconnect it in Capy "
            "under Settings > Models, or run the model on the Capy balance"
        )
    if rejection == "not_connected":
        return (
            f"no {label} is linked to this Capy account (a service-user key also "
            "needs a member to grant it one); link it in Capy under Settings > "
            "Models, or run the model on the Capy balance"
        )
    if is_linked_route(model_id):
        prefix = model_id.split("/", 1)[0]
        return (
            f"the {LINKED_ROUTES[prefix]} this model runs through is not set up "
            "for this Capy organization"
        )
    return "Capy rejected the model selection"


def model_route_field() -> ModelRoute:
    return SchemaField(
        description=(
            "Who pays for the model. as_given uses model_id exactly as written. "
            "capy_balance bills the Capy balance. codex, copilot, supergrok and "
            "azure run the model through that provider linked in Capy's "
            "settings, so it bills the subscription instead."
        ),
        default=ModelRoute.AS_GIVEN,
        advanced=True,
    )


def capy_balance_fallback_field() -> bool:
    return SchemaField(
        description=(
            "If the linked provider is disconnected or not linked, run the same "
            "model on the Capy balance instead of failing. Off by default, "
            "because it moves the cost from the subscription to the balance."
        ),
        default=False,
        advanced=True,
    )
