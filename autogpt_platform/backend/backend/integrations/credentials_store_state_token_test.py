"""State-token lifecycle for the device-code flow.

`peek` and `consume` exist as a pair specifically so a poll loop can read the
same token many times while exactly one caller may retire it. Both halves are
pinned here: without the non-consuming read the second poll 400s, and without
single-use consumption an approval stores duplicate credentials.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from backend.data.model import OAuthState, UserIntegrations
from backend.integrations.credentials_store import IntegrationCredentialsStore


class _NoopLockContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        return None


def _state(
    token: str = "tok",
    provider: str = "stripe_link",
    expires_in: int = 600,
) -> OAuthState:
    return OAuthState(
        token=token,
        provider=provider,
        expires_at=int(
            (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).timestamp()
        ),
        scopes=["userinfo:read"],
        state_metadata={"flow_type": "device_code", "device_code": "dev-code"},
    )


def _store_with(mocker, states: list[OAuthState]):
    store = IntegrationCredentialsStore()
    integrations = UserIntegrations(oauth_states=states)
    mocker.patch.object(
        store, "locked_user_integrations", AsyncMock(return_value=_NoopLockContext())
    )
    mocker.patch.object(
        store, "_get_user_integrations", AsyncMock(return_value=integrations)
    )
    # `db_manager` is a property returning the real DatabaseManager client; left
    # unpatched, `consume_state_token` makes an RPC whose socket read never
    # returns and the test hangs rather than fails.
    db_manager = MagicMock()
    db_manager.update_user_integrations = AsyncMock()
    mocker.patch.object(
        IntegrationCredentialsStore,
        "db_manager",
        new_callable=PropertyMock,
        return_value=db_manager,
    )
    return store, integrations


@pytest.mark.asyncio
async def test_peek_returns_the_state_without_removing_it(mocker):
    store, integrations = _store_with(mocker, [_state()])

    for _ in range(3):
        found = await store.peek_state_token("user-a", "tok", "stripe_link")
        assert found is not None
        assert found.state_metadata["device_code"] == "dev-code"

    assert len(integrations.oauth_states) == 1


@pytest.mark.asyncio
async def test_consume_removes_the_state_and_only_succeeds_once(mocker):
    store, integrations = _store_with(mocker, [_state()])

    first = await store.consume_state_token("user-a", "tok", "stripe_link")
    assert first is not None
    assert integrations.oauth_states == []

    # The race loser: same token, already retired.
    second = await store.consume_state_token("user-a", "tok", "stripe_link")
    assert second is None


@pytest.mark.asyncio
async def test_peek_rejects_a_token_from_another_provider(mocker):
    store, _ = _store_with(mocker, [_state(provider="github")])

    assert await store.peek_state_token("user-a", "tok", "stripe_link") is None


@pytest.mark.asyncio
async def test_peek_and_consume_reject_an_expired_token(mocker):
    store, integrations = _store_with(mocker, [_state(expires_in=-1)])

    assert await store.peek_state_token("user-a", "tok", "stripe_link") is None
    assert await store.consume_state_token("user-a", "tok", "stripe_link") is None
    # An expired token is rejected, not silently swept.
    assert len(integrations.oauth_states) == 1


@pytest.mark.asyncio
async def test_peek_rejects_an_unknown_token(mocker):
    store, _ = _store_with(mocker, [_state()])

    assert await store.peek_state_token("user-a", "wrong", "stripe_link") is None
