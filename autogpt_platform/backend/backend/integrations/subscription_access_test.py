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
    enforce_subscription_access,
    has_subscription_access,
    hidden_subscription_providers,
    visible_subscription_providers,
)
from backend.util.entitlements import EntitlementRequiredError
from prisma.enums import SubscriptionTier
from backend.util.exceptions import UserPaywalledError


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


class TestRunningOnASubscription:
    """The gate on dispatching a queued turn, which is not the paywall.

    A run on a linked subscription is paid for by the user's own plan. Asking
    "has this user paid AutoGPT" there blocks someone for owing us nothing --
    which is what happened to every provider but the first, because the check
    was written as `== "codex"` with the paywall in the else.
    """

    @pytest.mark.asyncio
    async def test_an_entitled_user_may_run(self, entitled) -> None:
        assert await has_subscription_access("user-1", "codex")

    @pytest.mark.asyncio
    async def test_an_unentitled_user_may_not(self, unentitled) -> None:
        assert not await has_subscription_access("user-1", "codex")

    @pytest.mark.asyncio
    async def test_a_provider_the_operator_disabled_may_not(
        self, entitled, monkeypatch
    ) -> None:
        monkeypatch.delenv("CHAT_ENABLE_GROK_SUBSCRIPTION", raising=False)
        assert not await has_subscription_access("user-1", "grok")

    @pytest.mark.asyncio
    async def test_enforcement_says_what_would_change_it(self, monkeypatch) -> None:
        """The message reaches a user mid-run, so it has to be the one the
        offer would have shown -- not a bare "forbidden".

        Enforcement asks `require_entitlement`, not the discovery helper: it
        must raise rather than answer False, so that a lookup failure stops a
        run instead of quietly allowing one.
        """

        async def _refuse(user_id: str, entitlement) -> None:
            raise EntitlementRequiredError(entitlement, SubscriptionTier.MAX)

        monkeypatch.setattr(
            "backend.integrations.subscription_access.require_entitlement", _refuse
        )
        with pytest.raises(UserPaywalledError) as raised:
            await enforce_subscription_access("user-1", "codex")
        assert "Max plan" in str(raised.value)

    @pytest.mark.asyncio
    async def test_enforcement_lets_an_entitled_user_through(self, monkeypatch) -> None:
        async def _allow(user_id: str, entitlement) -> None:
            return None

        monkeypatch.setattr(
            "backend.integrations.subscription_access.require_entitlement", _allow
        )
        await enforce_subscription_access("user-1", "codex")

    @pytest.mark.asyncio
    async def test_enforcement_refuses_a_provider_the_operator_disabled(
        self, monkeypatch
    ) -> None:
        """Entitlement is not the only gate: an operator who has not opted in
        has not agreed to run against that vendor at all."""

        async def _allow(user_id: str, entitlement) -> None:
            return None

        monkeypatch.setattr(
            "backend.integrations.subscription_access.require_entitlement", _allow
        )
        monkeypatch.delenv("CHAT_ENABLE_GROK_SUBSCRIPTION", raising=False)
        with pytest.raises(UserPaywalledError):
            await enforce_subscription_access("user-1", "grok")
