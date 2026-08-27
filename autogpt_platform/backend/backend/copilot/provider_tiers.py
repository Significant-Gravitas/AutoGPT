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

from backend.copilot.subscription_providers import (
    is_enabled,
    known_profiles,
)
from backend.copilot.config import ChatConfig, CopilotLLMModel
from backend.copilot.engine import resolve_use_sdk
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
    # The provider key, so a client can tie this record to a connection
    # without matching on the display name.
    auth_provider: str | None = None
    # What the sign-in button says and whose terms it sends the user to.
    # Here rather than in the client so adding a provider does not mean
    # adding a branch to whatever renders the button.
    connect_button_label: str | None = None
    terms_company: str | None = None
    # How long the client should wait on the sign-in window before giving up.
    # A subscription sign-in hands the user to a third party and, for the
    # device-code providers, to a CLI that polls -- far longer than a plain
    # redirect, and long enough that the client's default cuts it off partway
    # through. The server owns the number because the server knows which
    # strategy the provider uses.
    login_timeout_seconds: int | None = None


class ProviderTiersResponse(BaseModel):
    providers: list[ProviderTiers]


async def describe_provider_tiers(user_id: str) -> list[ProviderTiers]:
    """What every provider's tiers resolve to for this user's engine.

    User-scoped only because the engine is: which models a tier maps to
    depends on whether the turn will run on the SDK path. It says nothing
    about whether the user may use either provider.
    """
    config = ChatConfig()
    mode = await resolve_engine_mode(user_id, config)
    platform = await platform_tier_models(mode, user_id, config)
    # Only codex can name its models today; the others resolve through
    # surfaces the catalog does not carry yet, and an empty tier list is the
    # honest answer rather than a guess.
    per_provider = {"codex": codex_tier_models(mode)}

    described: list[ProviderTiers] = []
    for profile in known_profiles():
        if profile.key == "platform":
            models = platform
        elif not is_enabled(profile):
            # Not offered on this deployment, so not described either. This
            # test comes first: a provider that is off must stay absent
            # whatever else is true of it.
            continue
        elif not profile.serves_named_models:
            # Not "we could not resolve it" -- the provider has no model to
            # name. An empty list keeps the connection describable without
            # claiming a tier structure it does not have.
            models = {}
        else:
            models = per_provider.get(profile.key, {})
        described.append(
            ProviderTiers(
                provider_family=profile.provider_family,
                display_name=profile.display_name,
                auth_provider=profile.key,
                connect_button_label=profile.connect_button_label,
                terms_company=profile.terms_company,
                login_timeout_seconds=profile.login_timeout_seconds,
                tiers=_as_tiers(models),
            )
        )
    return described


async def resolve_engine_mode(user_id: str, config: ChatConfig) -> str:
    """One engine decision per response.

    It is a property of the deployment and the user, not of which connection
    they pick, so it is resolved once rather than per provider.
    """
    use_sdk = await resolve_use_sdk(
        user_id,
        use_claude_code_subscription=config.use_claude_code_subscription,
        config_default=config.use_claude_agent_sdk,
        thinking_available=config.thinking_available,
    )
    return "thinking" if use_sdk else "fast"


async def platform_tier_models(
    mode: str, user_id: str, config: ChatConfig
) -> dict[CopilotLLMModel, str | None]:
    """Resolve each tier against the engine this user's turns will run on.

    Answerable at all only because nothing can name an engine per request
    any more — the decision is the server's, so it can be made before a turn
    exists rather than during one.
    """
    resolved: dict[CopilotLLMModel, str | None] = {}
    for tier in TIER_LABELS:
        try:
            route = await resolve_model_route(mode, tier, user_id, config=config)
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


def codex_tier_models(mode: str) -> dict[CopilotLLMModel, str | None]:
    """The models the catalog pins for the Codex cells of this engine.

    A registry read, so it costs nothing and needs no credential. The router
    validates the pinned slug against what the account actually advertises
    when the turn runs, and falls back if it is gone -- so this names the
    routed model, not a promise about the account.
    """
    return {
        tier: display_name(llm_registry.get_route(ROUTE_SURFACE_CODEX, mode, tier))
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
