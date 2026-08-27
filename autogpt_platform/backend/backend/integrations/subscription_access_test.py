"""Who gets offered a subscription provider.

The gate is a security-shaped question wearing presentation clothes: the
connections list is where a user first learns a provider exists, so listing
one they cannot use produces a sign-in that fails at the very end, after
they have already handed credentials to a third party. Fails closed, and
these are the assertions that keep it that way.
"""

import pytest

from backend.integrations.providers import ProviderName
from backend.integrations.subscription_access import (
    hidden_subscription_providers,
    visible_subscription_providers,
)


@pytest.fixture
def entitled(monkeypatch):
    """Everyone is entitled to everything."""

    async def _yes(user_id: str, entitlement) -> bool:
        return True

    monkeypatch.setattr(
        "backend.integrations.subscription_access.has_entitlement_for_discovery", _yes
    )


@pytest.fixture
def unentitled(monkeypatch):
    async def _no(user_id: str, entitlement) -> bool:
        return False

    monkeypatch.setattr(
        "backend.integrations.subscription_access.has_entitlement_for_discovery", _no
    )


@pytest.mark.asyncio
async def test_an_anonymous_caller_is_offered_nothing(entitled) -> None:
    """No user id means no entitlement to check, and an unverifiable claim
    is not a reason to show the provider -- the sign-in would refuse."""
    assert await visible_subscription_providers(None) == set()
    assert ProviderName.CODEX in await hidden_subscription_providers(None)


@pytest.mark.asyncio
async def test_an_entitled_user_is_offered_what_the_deployment_enables(
    entitled,
) -> None:
    visible = await visible_subscription_providers("user-1")
    assert ProviderName.CODEX in visible
    # Grok is behind an operator opt-in that is off by default, so entitlement
    # alone must not surface it.
    assert ProviderName.GROK not in visible


@pytest.mark.asyncio
async def test_the_operator_opt_in_is_the_outer_gate(entitled, monkeypatch) -> None:
    monkeypatch.setenv("CHAT_ENABLE_GROK_SUBSCRIPTION", "true")
    assert ProviderName.GROK in await visible_subscription_providers("user-1")


@pytest.mark.asyncio
async def test_an_unentitled_user_sees_none_of_them(unentitled) -> None:
    assert await visible_subscription_providers("user-1") == set()
    hidden = await hidden_subscription_providers("user-1")
    assert {ProviderName.CODEX, ProviderName.GITHUB_COPILOT} <= hidden


@pytest.mark.asyncio
async def test_a_lookup_that_blows_up_hides_rather_than_shows(monkeypatch) -> None:
    """`has_entitlement_for_discovery` already swallows its own errors; this
    asserts the caller does not reintroduce a fail-open path around it."""

    async def _explode(user_id: str, entitlement) -> bool:
        raise RuntimeError("billing is down")

    monkeypatch.setattr(
        "backend.util.entitlements.has_entitlement",
        _explode,
    )
    assert await visible_subscription_providers("user-1") == set()


@pytest.mark.asyncio
async def test_hidden_and_visible_are_complements(entitled) -> None:
    """Callers filter with one and build with the other; a provider that
    slipped out of both would be listed to everyone."""
    visible = await visible_subscription_providers("user-1")
    hidden = await hidden_subscription_providers("user-1")
    assert not (visible & hidden)
    assert ProviderName.CODEX in visible | hidden
    # An ordinary integration is in neither: this gate decides whether to
    # show a *subscription*, and a set that quietly swept up every provider
    # would empty the connections list for everyone.
    assert ProviderName.GITHUB not in visible | hidden
