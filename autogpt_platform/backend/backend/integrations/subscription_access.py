"""Which subscription providers a user may even see.

``codex/access.py`` answers this for one provider. The question is the same
for every one of them -- is this deployment offering it, and is this user
entitled to it -- and the provider table already holds both answers, so this
module asks it generically rather than growing a second, third and fourth
near-copy of the codex helper.

Discovery only. Every function here fails closed and never raises: a
provider wrongly hidden for one render costs a page nothing, where one
wrongly shown produces a connection that refuses on first use. Enforcement
asks a different question, and lives with the route that spends.
"""

import logging

from backend.copilot.subscription_providers import (
    is_enabled,
    known_profiles,
    linked_profiles,
    profile_for,
)
from backend.integrations.providers import ProviderName
from backend.util.entitlements import (
    EntitlementRequiredError,
    has_entitlement_for_discovery,
    require_entitlement,
)
from backend.util.exceptions import UserPaywalledError

logger = logging.getLogger(__name__)


async def visible_subscription_providers(user_id: str | None) -> set[ProviderName]:
    """The subscription credential providers this user may be offered.

    An anonymous caller gets none: entitlements are per-user, and a provider
    shown to someone we cannot check is one that will refuse when they try
    to use it.
    """
    if user_id is None:
        return set()

    visible: set[ProviderName] = set()
    for profile in linked_profiles():
        if profile.credential_provider is None:
            continue
        if profile.entitlement is not None and not await has_entitlement_for_discovery(
            user_id, profile.entitlement
        ):
            continue
        visible.add(profile.credential_provider)
    return visible


async def hidden_subscription_providers(user_id: str | None) -> set[ProviderName]:
    """The inverse, for callers filtering a list they already built.

    Derived from every known provider rather than from the offered ones,
    because a provider the operator has not opted into is absent from
    ``linked_profiles`` entirely -- and must still be removed from a list
    that was assembled without asking.
    """
    return _all_subscription_providers() - await visible_subscription_providers(user_id)


def _all_subscription_providers() -> set[ProviderName]:
    return {
        profile.credential_provider
        for profile in known_profiles()
        if profile.credential_strategy != "platform"
        and profile.credential_provider is not None
    }


async def enforce_subscription_access(user_id: str, auth_provider: str) -> None:
    """Refuse a run on a subscription the user is not entitled to.

    Enforcement, not discovery: this raises. The distinction is the same one
    ``has_entitlement_for_discovery`` documents -- hiding an offer on a bad
    lookup is polite, letting a run through on one is not.
    """
    profile = profile_for(auth_provider)
    if profile.entitlement is None:
        return
    if not is_enabled(profile):
        raise UserPaywalledError(
            f"{profile.display_name} is not enabled on this deployment."
        )
    try:
        await require_entitlement(user_id, profile.entitlement)
    except EntitlementRequiredError:
        raise UserPaywalledError(
            profile.lock_reason or f"Your plan does not include {profile.display_name}."
        ) from None


async def has_subscription_access(user_id: str, auth_provider: str) -> bool:
    """Whether a queued run on this provider may be dispatched.

    Fails closed so a lookup that errors leaves the turn queued rather than
    running it on an entitlement nobody could confirm.

    A provider this build does not know is refused rather than raised on: the
    caller is the queue dispatcher, and a session that names a provider we
    have since removed must not take down the dispatch loop for every other
    turn that user has waiting.
    """
    try:
        profile = profile_for(auth_provider)
    except ValueError:
        logger.warning(
            "Queued turn names unknown chat provider %r; leaving it queued",
            auth_provider,
        )
        return False
    if not is_enabled(profile):
        return False
    if profile.entitlement is None:
        return True
    return await has_entitlement_for_discovery(user_id, profile.entitlement)
