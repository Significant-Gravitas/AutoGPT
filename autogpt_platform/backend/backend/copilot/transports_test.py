"""Chat transport discovery and the connection a user made their default."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock
from pydantic import SecretStr

from backend.copilot import transports
from backend.copilot.transports import (
    DefaultChatRoute,
    InvalidDefaultChatRoute,
    get_chat_transports,
    resolve_default_chat_route,
    save_default_chat_route,
)
from backend.data.model import OAuth2Credentials
from backend.integrations.codex.auth_bundle import (
    CodexAuthBundleV1,
    CodexAuthTokensV1,
    encode_provider_state,
)
from backend.util.settings import BehaveAs

USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"
VALID_CODEX_PROVIDER_STATE = encode_provider_state(
    CodexAuthBundleV1(
        tokens=CodexAuthTokensV1(
            id_token=SecretStr("id-token"),
            access_token=SecretStr("access-token"),
            refresh_token=SecretStr("refresh-token"),
        ),
        codex_runtime_version="0.144.4",
    )
)


def _codex_credentials(credential_id: str = "cred-codex") -> OAuth2Credentials:
    return OAuth2Credentials(
        id=credential_id,
        provider="codex",
        access_token=SecretStr("access"),
        refresh_token=SecretStr("refresh"),
        scopes=[],
        refresh_strategy="provider_runtime",
        provider_state=SecretStr(VALID_CODEX_PROVIDER_STATE),
        provider_state_version=1,
    )


@pytest.fixture(autouse=True)
def transport_env(mocker: pytest_mock.MockerFixture):
    """Hosted deployment, entitled user, no connections, nothing saved."""
    mocker.patch.object(transports.settings.config, "behave_as", BehaveAs.CLOUD)
    mocker.patch.object(
        transports,
        "has_codex_access_for_discovery",
        new=AsyncMock(return_value=True),
    )
    mocker.patch.object(
        transports.credentials_manager.store,
        "get_creds_by_provider",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch.object(
        transports,
        "get_user_default_chat_route",
        new=AsyncMock(return_value=(None, None)),
    )
    mocker.patch.object(transports, "set_user_default_chat_route", new=AsyncMock())


def _connect(*credential_ids: str) -> None:
    transports.credentials_manager.store.get_creds_by_provider.return_value = [
        _codex_credentials(credential_id) for credential_id in credential_ids
    ]


def _saved(auth_provider: str | None, credential_id: str | None = None) -> None:
    transports.get_user_default_chat_route.return_value = (
        auth_provider,
        credential_id,
    )


def _self_hosted_without_deployment(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch.object(transports.settings.config, "behave_as", BehaveAs.LOCAL)
    chat_config = MagicMock()
    chat_config.test_mode = False
    chat_config.use_claude_code_subscription = False
    chat_config.main_client_credentials = (None, "https://api.anthropic.com/v1/")
    mocker.patch.object(transports, "config", chat_config)


def _default_of(transport_list) -> tuple[str, str | None] | None:
    default = next((t for t in transport_list if t.default), None)
    return None if default is None else (default.auth_provider, default.credential_id)


# ─── which transport comes back marked default ─────────────────────────


@pytest.mark.asyncio
async def test_nothing_saved_keeps_the_server_pick() -> None:
    _connect("cred-codex")

    assert _default_of(await get_chat_transports(USER_ID)) == ("platform", None)


@pytest.mark.asyncio
async def test_saved_choice_beats_the_server_pick() -> None:
    _connect("cred-codex")
    _saved("codex", "cred-codex")

    assert _default_of(await get_chat_transports(USER_ID)) == ("codex", "cred-codex")


@pytest.mark.asyncio
async def test_exactly_one_transport_is_ever_default() -> None:
    _connect("cred-a", "cred-b")
    _saved("codex", "cred-b")

    transport_list = await get_chat_transports(USER_ID)

    assert [t.default for t in transport_list].count(True) == 1


@pytest.mark.asyncio
async def test_saved_choice_names_the_account_not_just_the_provider() -> None:
    _connect("cred-a", "cred-b")
    _saved("codex", "cred-b")

    assert _default_of(await get_chat_transports(USER_ID)) == ("codex", "cred-b")


# ─── healing, when the saved choice can't be honoured ──────────────────


@pytest.mark.asyncio
async def test_disconnected_account_heals_to_the_server_pick() -> None:
    _connect("cred-other")
    _saved("codex", "cred-gone")

    assert _default_of(await get_chat_transports(USER_ID)) == ("platform", None)


@pytest.mark.asyncio
async def test_lost_entitlement_heals_to_the_server_pick() -> None:
    _connect("cred-codex")
    _saved("codex", "cred-codex")
    transports.has_codex_access_for_discovery.return_value = False

    assert _default_of(await get_chat_transports(USER_ID)) == ("platform", None)


@pytest.mark.asyncio
async def test_healing_lands_on_the_remaining_connection_when_there_is_no_platform(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Self-host has no platform route to fall back to.

    Healing to ``platform`` would strand exactly the users who have no
    platform credentials — so it heals to whatever the server would have
    picked on its own, which here is the one connection that still exists.
    """
    _self_hosted_without_deployment(mocker)
    _connect("cred-other")
    _saved("codex", "cred-gone")

    assert _default_of(await get_chat_transports(USER_ID)) == ("codex", "cred-other")


@pytest.mark.asyncio
async def test_the_saved_row_survives_a_heal() -> None:
    _connect()
    _saved("codex", "cred-gone")

    await get_chat_transports(USER_ID)

    transports.set_user_default_chat_route.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_provider_this_version_does_not_know_reads_as_automatic() -> None:
    _connect("cred-codex")
    _saved("github_copilot", "cred-gh")

    assert _default_of(await get_chat_transports(USER_ID)) == ("platform", None)


# ─── resolve_default_chat_route: the unrouted callers ──────────────────


@pytest.mark.asyncio
async def test_unrouted_callers_get_the_saved_choice() -> None:
    _connect("cred-codex")
    _saved("codex", "cred-codex")

    assert await resolve_default_chat_route(USER_ID) == ("codex", "cred-codex")


@pytest.mark.asyncio
async def test_unrouted_callers_fall_back_to_platform_when_nothing_resolves(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Two connections and no platform route is a question, not an answer.

    The HTTP path answers it with a 409 and asks the user. A Discord message
    has nobody to ask, so it keeps the behaviour it had before the setting
    existed rather than failing.
    """
    _self_hosted_without_deployment(mocker)
    _connect("cred-a", "cred-b")

    assert await resolve_default_chat_route(USER_ID) == ("platform", None)


@pytest.mark.asyncio
async def test_unrouted_callers_survive_a_broken_lookup(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(
        transports,
        "get_chat_transports",
        new=AsyncMock(side_effect=RuntimeError("credential store down")),
    )

    assert await resolve_default_chat_route(USER_ID) == ("platform", None)


# ─── saving ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_saving_persists_and_returns_the_new_default() -> None:
    _connect("cred-codex")

    transport_list = await save_default_chat_route(
        USER_ID, DefaultChatRoute(auth_provider="codex", credential_id="cred-codex")
    )

    transports.set_user_default_chat_route.assert_awaited_once_with(
        USER_ID, "codex", "cred-codex"
    )
    assert _default_of(transport_list) == ("codex", "cred-codex")


@pytest.mark.asyncio
async def test_clearing_hands_the_decision_back_to_the_server() -> None:
    _connect("cred-codex")
    _saved("codex", "cred-codex")

    transport_list = await save_default_chat_route(USER_ID, DefaultChatRoute())

    transports.set_user_default_chat_route.assert_awaited_once_with(USER_ID, None, None)
    # The stub still reports the old saved value, so asserting the refreshed
    # list would only prove the stub. The write is the contract here.
    assert transport_list is not None


@pytest.mark.asyncio
async def test_chatgpt_without_an_account_is_rejected() -> None:
    _connect("cred-codex")

    with pytest.raises(InvalidDefaultChatRoute) as error:
        await save_default_chat_route(USER_ID, DefaultChatRoute(auth_provider="codex"))

    assert error.value.detail == "codex_credential_required"
    transports.set_user_default_chat_route.assert_not_awaited()


@pytest.mark.asyncio
async def test_platform_with_an_account_is_rejected() -> None:
    with pytest.raises(InvalidDefaultChatRoute) as error:
        await save_default_chat_route(
            USER_ID,
            DefaultChatRoute(auth_provider="platform", credential_id="cred-codex"),
        )

    assert error.value.detail == "codex_credential_not_allowed"
    transports.set_user_default_chat_route.assert_not_awaited()


@pytest.mark.asyncio
async def test_clearing_while_naming_an_account_is_rejected() -> None:
    with pytest.raises(InvalidDefaultChatRoute) as error:
        await save_default_chat_route(
            USER_ID, DefaultChatRoute(credential_id="cred-codex")
        )

    assert error.value.detail == "codex_credential_not_allowed"
    transports.set_user_default_chat_route.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_account_the_user_does_not_own_is_rejected() -> None:
    _connect("cred-mine")

    with pytest.raises(InvalidDefaultChatRoute) as error:
        await save_default_chat_route(
            USER_ID,
            DefaultChatRoute(auth_provider="codex", credential_id="cred-someone-else"),
        )

    assert error.value.detail == "codex_credential_not_found"
    transports.set_user_default_chat_route.assert_not_awaited()


@pytest.mark.asyncio
async def test_chatgpt_is_not_saveable_without_the_entitlement() -> None:
    _connect("cred-codex")
    transports.has_codex_access_for_discovery.return_value = False

    with pytest.raises(InvalidDefaultChatRoute) as error:
        await save_default_chat_route(
            USER_ID,
            DefaultChatRoute(auth_provider="codex", credential_id="cred-codex"),
        )

    assert error.value.detail == "codex_credential_not_found"
    transports.set_user_default_chat_route.assert_not_awaited()


@pytest.mark.asyncio
async def test_platform_is_not_saveable_where_it_is_unavailable(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _self_hosted_without_deployment(mocker)

    with pytest.raises(InvalidDefaultChatRoute) as error:
        await save_default_chat_route(
            USER_ID, DefaultChatRoute(auth_provider="platform")
        )

    assert error.value.detail == "chat_transport_not_configured"
    transports.set_user_default_chat_route.assert_not_awaited()
