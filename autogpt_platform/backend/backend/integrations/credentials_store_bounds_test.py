"""What one account may keep in the integrations store is bounded.

The bound sits in the store, which the REST route, the external API, the MCP
routes and the OAuth callbacks all reach.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, OAuthState, UserIntegrations
from backend.integrations.credentials_store import (
    MAX_CREDENTIALS_PER_USER,
    MAX_PENDING_OAUTH_STATES_PER_USER,
    MAX_STORED_ITEM_BYTES,
    IntegrationCredentialsStore,
    check_storable_size,
    stored_size,
)
from backend.util.exceptions import InvalidInputError


def _credentials(**overrides) -> APIKeyCredentials:
    fields: dict = {
        "provider": "github",
        "api_key": SecretStr("a-key"),
        "title": "a title",
    }
    fields.update(overrides)
    return APIKeyCredentials(**fields)


def _state(expires_at: datetime) -> OAuthState:
    return OAuthState(
        token="t",
        provider="github",
        expires_at=int(expires_at.timestamp()),
        scopes=[],
    )


def test_an_item_at_the_size_bound_is_accepted():
    base = stored_size(_credentials(metadata={"pad": ""}))
    at_bound = _credentials(metadata={"pad": "x" * (MAX_STORED_ITEM_BYTES - base)})

    assert stored_size(at_bound) == MAX_STORED_ITEM_BYTES
    check_storable_size(at_bound)


def test_an_item_over_the_size_bound_is_refused():
    padded = _credentials(metadata={"pad": "x" * MAX_STORED_ITEM_BYTES})

    with pytest.raises(InvalidInputError, match="is too large"):
        check_storable_size(padded)


def test_the_size_is_measured_on_the_secret_too():
    """The check reads what is stored, not a masked rendering of it.

    `Credentials` serialises its `SecretStr` fields in full, because that is
    what is persisted. If that ever changed, measuring the serialised model
    would stop seeing the bytes that matter, so this pins it.
    """
    padded = _credentials(api_key=SecretStr("x" * (MAX_STORED_ITEM_BYTES + 1)))

    with pytest.raises(InvalidInputError, match="is too large"):
        check_storable_size(padded)


@pytest.fixture
def store(mocker):
    store = IntegrationCredentialsStore()
    mocker.patch.object(store, "locked_user_integrations", mocker.AsyncMock())
    mocker.patch.object(store, "_set_user_integration_creds", mocker.AsyncMock())
    return store


@pytest.mark.asyncio
async def test_a_user_at_the_credential_count_bound_cannot_add_another(store, mocker):
    held = [_credentials(id=f"id-{i}") for i in range(MAX_CREDENTIALS_PER_USER)]
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        mocker.AsyncMock(return_value=held),
    )

    with pytest.raises(InvalidInputError, match="at most"):
        await store.add_creds("user-1", _credentials(id="one-too-many"))
    store._set_user_integration_creds.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_user_below_the_count_bound_can_add_one(store, mocker):
    held = [_credentials(id=f"id-{i}") for i in range(MAX_CREDENTIALS_PER_USER - 1)]
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        mocker.AsyncMock(return_value=held),
    )

    await store.add_creds("user-1", _credentials(id="the-last-one"))

    written = store._set_user_integration_creds.await_args.args[1]
    assert [c.id for c in written][-1] == "the-last-one"


@pytest.mark.asyncio
async def test_an_upsert_adding_a_provider_at_the_count_bound_is_refused(store, mocker):
    held = [_credentials(id=f"id-{i}") for i in range(MAX_CREDENTIALS_PER_USER)]
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        mocker.AsyncMock(return_value=held),
    )

    with pytest.raises(InvalidInputError, match="at most"):
        await store.upsert_single_provider_creds(
            "user-1", _credentials(provider="linear")
        )
    store._set_user_integration_creds.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_upsert_replacing_a_provider_at_the_count_bound_goes_through(
    store, mocker
):
    held = [_credentials(id=f"id-{i}") for i in range(MAX_CREDENTIALS_PER_USER - 1)]
    held.append(_credentials(id="linear-1", provider="linear"))
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        mocker.AsyncMock(return_value=held),
    )

    await store.upsert_single_provider_creds("user-1", _credentials(provider="linear"))

    written = store._set_user_integration_creds.await_args.args[1]
    assert len(written) == MAX_CREDENTIALS_PER_USER


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["add_creds", "upsert_single_provider_creds"])
async def test_managed_credentials_do_not_count_toward_the_bound(
    store, mocker, method_name: str
):
    held = [_credentials(id=f"id-{i}") for i in range(MAX_CREDENTIALS_PER_USER - 1)]
    held.append(_credentials(id="managed", provider="openai", is_managed=True))
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        mocker.AsyncMock(return_value=held),
    )

    await getattr(store, method_name)("user-1", _credentials(provider="linear"))

    written = store._set_user_integration_creds.await_args.args[1]
    assert len(written) == MAX_CREDENTIALS_PER_USER + 1


@pytest.mark.asyncio
async def test_adding_a_credential_marked_managed_is_refused(store, mocker):
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        mocker.AsyncMock(return_value=[]),
    )

    with pytest.raises(InvalidInputError, match="user-owned"):
        await store.add_creds("user-1", _credentials(is_managed=True))
    store._set_user_integration_creds.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_upsert_that_keeps_a_long_stored_title_is_measured_after(
    store, mocker
):
    half = MAX_STORED_ITEM_BYTES // 2
    stored = _credentials(id="linear-1", provider="linear", title="t" * half)
    incoming = _credentials(provider="linear", metadata={"pad": "x" * half})
    check_storable_size(stored)
    check_storable_size(incoming)
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        mocker.AsyncMock(return_value=[stored]),
    )

    with pytest.raises(InvalidInputError, match="is too large"):
        await store.upsert_single_provider_creds("user-1", incoming)
    store._set_user_integration_creds.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["add_creds", "upsert_single_provider_creds"])
async def test_every_credential_write_refuses_an_oversized_item(
    store, mocker, method_name: str
):
    """The checks are at the writes, not only in the helper they call."""
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        mocker.AsyncMock(return_value=[]),
    )
    padded = _credentials(metadata={"pad": "x" * MAX_STORED_ITEM_BYTES})

    with pytest.raises(InvalidInputError, match="is too large"):
        await getattr(store, method_name)("user-1", padded)
    store._set_user_integration_creds.assert_not_awaited()


@pytest.mark.asyncio
async def test_updating_a_credential_to_an_oversized_one_is_refused(store, mocker):
    """`update_creds` measures against what is already stored, so its check
    runs once the current item is known rather than on the way in."""
    held = _credentials(id="held")
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        mocker.AsyncMock(return_value=[held]),
    )
    grown = held.model_copy(
        update={"api_key": SecretStr("y" * (MAX_STORED_ITEM_BYTES + 10))}
    )

    with pytest.raises(InvalidInputError, match="is too large"):
        await store.update_creds("user-1", grown)
    store._set_user_integration_creds.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_oversized_credential_can_still_be_refreshed(store, mocker):
    """A token refresh goes through `update_creds`.

    An item stored before this bound existed would otherwise be stuck: the
    refresh could never be written, and the credential would stop working.
    """
    held = _credentials(
        id="held", api_key=SecretStr("x" * (MAX_STORED_ITEM_BYTES + 10))
    )
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        mocker.AsyncMock(return_value=[held]),
    )
    refreshed = held.model_copy(
        update={"api_key": SecretStr("y" * (MAX_STORED_ITEM_BYTES + 10))}
    )

    await store.update_creds("user-1", refreshed)

    written = store._set_user_integration_creds.await_args.args[1]
    assert written[0].api_key.get_secret_value().startswith("y")


@pytest.mark.asyncio
async def test_an_oversized_credential_cannot_grow_further(store, mocker):
    held = _credentials(
        id="held", api_key=SecretStr("x" * (MAX_STORED_ITEM_BYTES + 10))
    )
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        mocker.AsyncMock(return_value=[held]),
    )
    grown = held.model_copy(
        update={"api_key": SecretStr("y" * (MAX_STORED_ITEM_BYTES + 2000))}
    )

    with pytest.raises(InvalidInputError, match="is too large"):
        await store.update_creds("user-1", grown)
    store._set_user_integration_creds.assert_not_awaited()


@pytest.mark.asyncio
async def test_storing_a_state_refuses_an_oversized_one(store, mocker):
    edit = mocker.patch.object(store, "edit_user_integrations")

    with pytest.raises(InvalidInputError, match="is too large"):
        await store.store_state_token(
            "user-1", "github", [], state_metadata={"pad": "x" * MAX_STORED_ITEM_BYTES}
        )
    edit.assert_not_called()


@pytest.mark.asyncio
async def test_storing_a_state_drops_the_expired_ones(store, mocker):
    now = datetime.now(UTC)
    integrations = UserIntegrations(
        oauth_states=[
            _state(now - timedelta(minutes=5)),
            _state(now + timedelta(minutes=5)),
        ]
    )
    mocker.patch.object(
        store,
        "edit_user_integrations",
        lambda _user_id: _yielding(integrations),
    )

    await store.store_state_token("user-1", "github", [])

    expiries = [s.expires_at for s in integrations.oauth_states]
    assert len(expiries) == 2
    assert all(e > now.timestamp() for e in expiries)


@pytest.mark.asyncio
async def test_a_user_at_the_pending_state_bound_cannot_start_another(store, mocker):
    now = datetime.now(UTC)
    integrations = UserIntegrations(
        oauth_states=[
            _state(now + timedelta(minutes=5))
            for _ in range(MAX_PENDING_OAUTH_STATES_PER_USER)
        ]
    )
    mocker.patch.object(
        store,
        "edit_user_integrations",
        lambda _user_id: _yielding(integrations),
    )

    with pytest.raises(InvalidInputError, match="pending sign-ins"):
        await store.store_state_token("user-1", "github", [])


def _yielding(value):
    """A minimal stand-in for the store's `edit_user_integrations` context."""

    class _Ctx:
        async def __aenter__(self):
            return value

        async def __aexit__(self, *_exc):
            return False

    return _Ctx()
