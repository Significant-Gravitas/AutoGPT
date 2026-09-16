"""What each quality tier resolves to, per provider, independent of access.

``offers`` answers "which connections can this user pick, and what are they".
That is the right question for the composer and for settings, and the wrong
one for every surface that has to *describe* a provider before the user has
one: the connect dialog, and the plan cards that sell a plan the user has not
bought yet.

Those surfaces need "ChatGPT's Advanced tier is 5.6 Sol" — a statement about
the catalog, not about the user's entitlements. Asking ``offers`` for it does
not work, because a connection nobody can select is deliberately absent from
that list, and hardcoding it in the client reintroduces exactly the drift the
offers endpoint exists to remove.

So this module owns tier resolution, and ``offers`` consumes it.
"""

import logging

from pydantic import BaseModel

from backend.copilot.config import ChatConfig, CopilotLLMModel
from backend.copilot.model_router import (
    ROUTE_SURFACE_CODEX,
    catalog_lookup,
    resolve_model_route,
)
from backend.data import llm_registry

logger = logging.getLogger(__name__)

# Product vocabulary. "Fast" and "Thinking" are execution paths, not tiers a
# user picks between; the two labels below are what the product exposes.
TIER_LABELS: dict[CopilotLLMModel, str] = {
    "standard": "Balanced",
    "advanced": "Advanced",
}


class ProviderTier(BaseModel):
    """A tier and the model it resolves to, with no claim about access.

    Deliberately not ``ConnectionTier``: that type carries ``selectable`` and
    ``lock_reason``, which are answers about a user. This one is a statement
    about the catalog and is true whether or not anyone can pick it.
    """

    tier: CopilotLLMModel
    label: str
    display_model: str | None = None


class ProviderTiers(BaseModel):
    provider_family: str
    display_name: str
    tiers: list[ProviderTier]


class ProviderTiersResponse(BaseModel):
    providers: list[ProviderTiers]


async def describe_provider_tiers(user_id: str) -> list[ProviderTiers]:
    """What every provider's tiers resolve to for this user.

    User-scoped only because LaunchDarkly routing is per-user. It says
    nothing about whether the user may use either provider.
    """
    config = ChatConfig()
    platform = await platform_tier_models(user_id, config)
    codex = codex_tier_models()
    return [
        ProviderTiers(
            provider_family="autogpt",
            display_name="AutoGPT Platform",
            tiers=_as_tiers(platform),
        ),
        ProviderTiers(
            provider_family="openai",
            display_name="ChatGPT",
            tiers=_as_tiers(codex),
        ),
    ]


async def platform_tier_models(
    user_id: str, config: ChatConfig
) -> dict[CopilotLLMModel, str | None]:
    """Resolve each tier the way a turn for this user would.

    Answerable before a turn exists because routing no longer takes a
    per-request engine — the tier alone decides.
    """
    resolved: dict[CopilotLLMModel, str | None] = {}
    for tier in TIER_LABELS:
        try:
            route = await resolve_model_route(tier, user_id, config=config)
            resolved[tier] = display_name(route.model)
        except Exception:
            # A tier that cannot be resolved is described without a name
            # rather than failing the whole list.
            logger.warning(
                "Could not resolve the %s model for the platform connection",
                tier,
                exc_info=True,
            )
            resolved[tier] = None
    return resolved


def codex_tier_models() -> dict[CopilotLLMModel, str | None]:
    """The models the catalog pins for each Codex tier.

    A registry read, so it costs nothing and needs no credential. The router
    validates the pinned slug against what the account actually advertises
    when the turn runs, and falls back if it is gone -- so this names the
    routed model, not a promise about the account.
    """
    return {
        tier: display_name(llm_registry.get_route(ROUTE_SURFACE_CODEX, tier))
        for tier in TIER_LABELS
    }


def display_name(slug: str | None) -> str | None:
    """What the catalog calls this model, rather than its slug.

    The PRD labels tiers "Balanced · 5.6 Terra", not
    "Balanced · gpt-5.6-terra": the slug is a routing key and reads as one.

    Goes through the router's lookup rather than the catalog directly,
    because the slug arrives in whatever spelling configured it. The default
    configuration routes through OpenRouter, whose provider-prefixed forms
    (``anthropic/claude-sonnet-5``) the catalog does not use as keys -- so a
    direct read misses on exactly the setup most deployments run.

    Falls back to the slug when nothing resolves, which is better than
    showing nothing for a model the catalog has never heard of -- minus the
    vendor prefix, which is how the transport addresses the model and carries
    nothing for the reader. Keeping it costs ten characters in a control that
    has to fit two of these side by side, and spends them on the one part of
    the string that cannot tell anyone anything.
    """
    if not slug:
        return None
    model = catalog_lookup(slug)
    if model:
        return model.display_name
    return slug.split("/", 1)[1] if "/" in slug else slug


def _as_tiers(models: dict[CopilotLLMModel, str | None]) -> list[ProviderTier]:
    return [
        ProviderTier(tier=tier, label=label, display_model=models.get(tier))
        for tier, label in TIER_LABELS.items()
    ]
