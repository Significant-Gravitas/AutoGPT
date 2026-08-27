"""A user's subscription token must not outlive their turn.

The baseline path's normal client is a process-global singleton, which is
right for a deployment key and would be a real incident for a user's: an
OpenAI client carries the credential it was built with, so a cached one
built from someone's ChatGPT or Grok token would quietly pay for every turn
that followed it in the same worker.

These are the assertions that keep that from being reintroduced by an
optimisation that looks reasonable in isolation.
"""

import pytest

from backend.copilot.subscription_endpoint import (
    SubscriptionEndpoint,
    SubscriptionEndpointUnavailable,
    build_subscription_client,
    register_endpoint_resolver,
    resolve_subscription_endpoint,
)


def _endpoint(token: str = "user-token", provider: str = "grok"):
    return SubscriptionEndpoint(
        auth_provider=provider,
        base_url="https://example.invalid/v1",
        api_key=token,
        model="a-model",
    )


class TestTheTokenStaysWithTheTurn:
    def test_every_call_builds_its_own_client(self) -> None:
        """Not an optimisation left on the table -- the thing that keeps one
        person's subscription from paying for the next person's chat."""
        endpoint = _endpoint()
        assert build_subscription_client(endpoint) is not build_subscription_client(
            endpoint
        )

    def test_two_users_never_share_a_client(self) -> None:
        first = build_subscription_client(_endpoint("token-a"))
        second = build_subscription_client(_endpoint("token-b"))
        assert first is not second
        assert first.api_key != second.api_key

    def test_the_token_does_not_show_up_in_a_log_line(self) -> None:
        """These reach the same error paths as everything else, and a
        subscription token in a Sentry payload is the user's account."""
        endpoint = _endpoint("super-secret-token")
        assert "super-secret-token" not in repr(endpoint)
        assert "super-secret-token" not in str(endpoint)
        assert "super-secret-token" not in str(endpoint.model_dump())


class TestResolvingWhereATurnGoes:
    @pytest.mark.asyncio
    async def test_the_platform_route_has_no_endpoint(self) -> None:
        """``None`` means the deployment's own key -- an actual absence, not
        a failure to find one."""
        assert (
            await resolve_subscription_endpoint(
                "platform", None, user_id="u1", model="m"
            )
            is None
        )
        assert (
            await resolve_subscription_endpoint(None, None, user_id="u1", model="m")
            is None
        )

    @pytest.mark.asyncio
    async def test_a_provider_with_no_runtime_raises_rather_than_falling_back(
        self,
    ) -> None:
        """The caller's fallback is the deployment key. Taking it silently
        would bill us for a turn the user routed to their own account, and
        tell them their subscription ran it -- so this is the one case that
        must never answer ``None``."""
        with pytest.raises(SubscriptionEndpointUnavailable):
            await resolve_subscription_endpoint(
                "not-built-yet", "cred-1", user_id="u1", model="m"
            )

    @pytest.mark.asyncio
    async def test_a_linked_route_without_a_credential_raises(self) -> None:
        with pytest.raises(SubscriptionEndpointUnavailable):
            await resolve_subscription_endpoint("grok", None, user_id="u1", model="m")

    @pytest.mark.asyncio
    async def test_a_linked_route_without_a_user_raises(self) -> None:
        """A credential is only meaningful against the user who owns it."""
        with pytest.raises(SubscriptionEndpointUnavailable):
            await resolve_subscription_endpoint(
                "grok", "cred-1", user_id=None, model="m"
            )

    @pytest.mark.asyncio
    async def test_a_registered_provider_is_asked_for_its_endpoint(self) -> None:
        seen: dict = {}

        async def _resolver(*, user_id: str, credential_id: str, model: str):
            seen.update(
                {"user_id": user_id, "credential_id": credential_id, "model": model}
            )
            return _endpoint(provider="test-provider")

        register_endpoint_resolver("test-provider", _resolver)
        endpoint = await resolve_subscription_endpoint(
            "test-provider", "cred-7", user_id="u1", model="a-model"
        )

        assert endpoint is not None
        assert endpoint.auth_provider == "test-provider"
        # The resolver is given everything it needs to fetch a per-user,
        # possibly short-lived token, which is why this is resolved per turn
        # rather than stored alongside the credential.
        assert seen == {
            "user_id": "u1",
            "credential_id": "cred-7",
            "model": "a-model",
        }
