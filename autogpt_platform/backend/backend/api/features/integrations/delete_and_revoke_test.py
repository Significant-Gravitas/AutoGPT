"""The shared delete-then-revoke path behind credential deletion.

The DELETE route and account deletion both go through
`delete_and_revoke_credentials`, so its order matters on its own: the local
copy goes first, then the token is revoked at the provider.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from pydantic import SecretStr

from backend.api.features.integrations.router import (
    delete_and_revoke_credentials,
    router,
)
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.providers import ProviderName
from backend.util.exceptions import NeedConfirmation
from backend.util.request import HTTPClientError

ROUTER = "backend.api.features.integrations.router"
USER_ID = "user-1"

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture
def authenticated(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _oauth2_cred(provider: str = "github") -> OAuth2Credentials:
    return OAuth2Credentials(
        id="cred-oauth",
        provider=provider,
        title="GitHub",
        access_token=SecretStr("ghp_access_secret"),
        refresh_token=SecretStr("ghp_refresh_secret"),
        scopes=["repo"],
        username="octocat",
    )


@pytest.mark.asyncio
async def test_oauth2_tokens_are_revoked_after_the_local_delete():
    creds = _oauth2_cred()
    handler = MagicMock()
    handler.revoke_tokens = AsyncMock(return_value=True)
    with (
        patch(
            f"{ROUTER}.remove_all_webhooks_for_credentials", new=AsyncMock()
        ) as remove_webhooks,
        patch(f"{ROUTER}.creds_manager") as manager,
        patch(f"{ROUTER}._get_provider_oauth_handler", return_value=handler),
    ):
        manager.delete = AsyncMock()
        order = MagicMock()
        order.attach_mock(manager.delete, "delete")
        order.attach_mock(handler.revoke_tokens, "revoke_tokens")

        revoked = await delete_and_revoke_credentials(
            USER_ID, creds, ProviderName.GITHUB, force=True
        )

    assert revoked is True
    remove_webhooks.assert_awaited_once_with(
        USER_ID, creds, True, skip_failed_prunes=False
    )
    manager.delete.assert_awaited_once_with(USER_ID, "cred-oauth")
    handler.revoke_tokens.assert_awaited_once_with(creds)
    assert [name for name, _, _ in order.mock_calls] == ["delete", "revoke_tokens"]


@pytest.mark.asyncio
async def test_webhooks_in_use_need_confirmation_and_nothing_is_deleted():
    creds = _oauth2_cred()
    in_use = MagicMock(triggered_nodes=[MagicMock()], triggered_presets=[])
    with (
        patch(
            f"{ROUTER}.get_all_webhooks_by_creds",
            new=AsyncMock(return_value=[in_use]),
        ),
        patch(f"{ROUTER}.creds_manager") as manager,
        patch(f"{ROUTER}._get_provider_oauth_handler") as get_handler,
    ):
        manager.delete = AsyncMock()
        with pytest.raises(NeedConfirmation):
            await delete_and_revoke_credentials(
                USER_ID, creds, ProviderName.GITHUB, force=False
            )

    manager.delete.assert_not_awaited()
    get_handler.assert_not_called()


def _hook_the_provider_refuses_to_delete() -> MagicMock:
    """A repo webhook GitHub answers with a 404 (repo gone) or 403 (admin lost)
    on every attempt."""
    manager = MagicMock()
    manager.prune_webhook_if_dangling = AsyncMock(
        side_effect=HTTPClientError("Not Found", 404)
    )
    return manager


@pytest.mark.asyncio
async def test_skipping_failed_prunes_still_deletes_and_revokes():
    creds = _oauth2_cred()
    hook = MagicMock(id="hook-1", triggered_nodes=[], triggered_presets=[])
    handler = MagicMock()
    handler.revoke_tokens = AsyncMock(return_value=True)
    with (
        patch(
            f"{ROUTER}.get_all_webhooks_by_creds", new=AsyncMock(return_value=[hook])
        ),
        patch(
            f"{ROUTER}.get_webhook_manager",
            return_value=_hook_the_provider_refuses_to_delete(),
        ),
        patch(f"{ROUTER}.creds_manager") as manager,
        patch(f"{ROUTER}._get_provider_oauth_handler", return_value=handler),
    ):
        manager.delete = AsyncMock()

        revoked = await delete_and_revoke_credentials(
            USER_ID, creds, ProviderName.GITHUB, force=True, skip_failed_prunes=True
        )

    assert revoked is True
    manager.delete.assert_awaited_once_with(USER_ID, "cred-oauth")
    handler.revoke_tokens.assert_awaited_once_with(creds)


@pytest.mark.parametrize("force", [True, False])
@pytest.mark.asyncio
async def test_the_disconnect_route_still_stops_on_a_prune_error(force: bool):
    creds = _oauth2_cred()
    hook = MagicMock(id="hook-1", triggered_nodes=[], triggered_presets=[])
    with (
        patch(
            f"{ROUTER}.get_all_webhooks_by_creds", new=AsyncMock(return_value=[hook])
        ),
        patch(
            f"{ROUTER}.get_webhook_manager",
            return_value=_hook_the_provider_refuses_to_delete(),
        ),
        patch(f"{ROUTER}.creds_manager") as manager,
        patch(f"{ROUTER}._get_provider_oauth_handler") as get_handler,
    ):
        manager.delete = AsyncMock()
        with pytest.raises(HTTPClientError):
            await delete_and_revoke_credentials(
                USER_ID, creds, ProviderName.GITHUB, force=force
            )

    manager.delete.assert_not_awaited()
    get_handler.assert_not_called()


@pytest.mark.asyncio
async def test_an_api_key_is_deleted_with_nothing_to_revoke():
    creds = APIKeyCredentials(
        id="cred-key",
        provider="openai",
        title="OpenAI",
        api_key=SecretStr("sk-secret"),
    )
    with (
        patch(f"{ROUTER}.remove_all_webhooks_for_credentials", new=AsyncMock()),
        patch(f"{ROUTER}.creds_manager") as manager,
        patch(f"{ROUTER}._get_provider_oauth_handler") as get_handler,
    ):
        manager.delete = AsyncMock()

        revoked = await delete_and_revoke_credentials(
            USER_ID, creds, ProviderName.OPENAI, force=False
        )

    assert revoked is None
    manager.delete.assert_awaited_once_with(USER_ID, "cred-key")
    get_handler.assert_not_called()


@pytest.mark.asyncio
async def test_device_code_credentials_are_revoked_with_the_device_handler():
    creds = _oauth2_cred(provider="stripe_link")
    device_handler_cls = MagicMock()
    device_handler_cls.return_value.revoke_tokens = AsyncMock(return_value=True)
    with (
        patch(f"{ROUTER}.remove_all_webhooks_for_credentials", new=AsyncMock()),
        patch(f"{ROUTER}.creds_manager") as manager,
        patch(f"{ROUTER}.DEVICE_HANDLERS_BY_NAME", {"stripe_link": device_handler_cls}),
        patch(f"{ROUTER}._get_provider_oauth_handler") as get_handler,
    ):
        manager.delete = AsyncMock()

        revoked = await delete_and_revoke_credentials(
            USER_ID, creds, ProviderName("stripe_link"), force=True
        )

    assert revoked is True
    device_handler_cls.return_value.revoke_tokens.assert_awaited_once_with(creds)
    get_handler.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_credentials_are_revoked_with_a_handler_from_their_metadata():
    creds = _oauth2_cred(provider="mcp")
    handler = MagicMock()
    handler.revoke_tokens = AsyncMock(return_value=False)
    with (
        patch(f"{ROUTER}.remove_all_webhooks_for_credentials", new=AsyncMock()),
        patch(f"{ROUTER}.creds_manager") as manager,
        patch(f"{ROUTER}.create_mcp_oauth_handler", return_value=handler) as create_mcp,
        patch(f"{ROUTER}._get_provider_oauth_handler") as get_handler,
    ):
        manager.delete = AsyncMock()

        revoked = await delete_and_revoke_credentials(
            USER_ID, creds, ProviderName.MCP, force=True
        )

    assert revoked is False
    create_mcp.assert_called_once_with(creds)
    handler.revoke_tokens.assert_awaited_once_with(creds)
    get_handler.assert_not_called()


def test_route_still_asks_for_confirmation_when_webhooks_are_in_use(
    authenticated: None,
):
    in_use = MagicMock(triggered_nodes=[MagicMock()], triggered_presets=[])
    with (
        patch(
            f"{ROUTER}.get_all_webhooks_by_creds",
            new=AsyncMock(return_value=[in_use]),
        ),
        patch(f"{ROUTER}.creds_manager") as manager,
    ):
        manager.store.get_creds_by_id = AsyncMock(return_value=_oauth2_cred())
        manager.delete = AsyncMock()

        resp = client.delete("/github/credentials/cred-oauth")

    assert resp.status_code == 200
    assert resp.json() == {
        "deleted": False,
        "need_confirmation": True,
        "message": "Some webhooks linked to these credentials are still in use "
        "by a workflow",
    }
    manager.delete.assert_not_awaited()


def test_route_reports_the_provider_revocation_result(
    authenticated: None, test_user_id: str
):
    creds = _oauth2_cred()
    handler = MagicMock()
    handler.revoke_tokens = AsyncMock(return_value=True)
    with (
        patch(
            f"{ROUTER}.remove_all_webhooks_for_credentials", new=AsyncMock()
        ) as remove_webhooks,
        patch(f"{ROUTER}.creds_manager") as manager,
        patch(
            f"{ROUTER}._get_provider_oauth_handler", return_value=handler
        ) as get_handler,
    ):
        manager.store.get_creds_by_id = AsyncMock(return_value=creds)
        manager.delete = AsyncMock()

        resp = client.delete("/github/credentials/cred-oauth?force=true")

    assert resp.status_code == 200
    assert resp.json() == {"deleted": True, "revoked": True}
    remove_webhooks.assert_awaited_once_with(
        test_user_id, creds, True, skip_failed_prunes=False
    )
    manager.delete.assert_awaited_once_with(test_user_id, "cred-oauth")
    get_handler.assert_called_once_with(ProviderName.GITHUB)
    handler.revoke_tokens.assert_awaited_once_with(creds)


def _pasted_mcp_token(metadata: dict) -> OAuth2Credentials:
    """An MCP credential the user pasted (POST /api/mcp/token): stored like an
    OAuth grant, with no refresh token and no OAuth endpoints."""
    return OAuth2Credentials(
        id="cred-mcp",
        provider="mcp",
        title="MCP: api.example.com",
        access_token=SecretStr("Bearer pasted-secret"),
        scopes=[],
        metadata=metadata,
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"mcp_server_url": "https://api.example.com/mcp", "mcp_auth_scheme": "bearer"},
        {"mcp_server_url": "https://api.example.com/mcp", "mcp_auth_scheme": "basic"},
        # Stored before the scheme was recorded: a bare token, sent as Bearer.
        {"mcp_server_url": "https://api.example.com/mcp"},
    ],
)
def test_route_deletes_a_pasted_mcp_token_with_nothing_to_revoke(
    authenticated: None, test_user_id: str, metadata: dict
):
    # The real handler factory stays in place: it can't build an OAuth handler
    # for a pasted token, and the delete must not need one.
    creds = _pasted_mcp_token(metadata)
    with (
        patch(f"{ROUTER}.remove_all_webhooks_for_credentials", new=AsyncMock()),
        patch(f"{ROUTER}.creds_manager") as manager,
    ):
        manager.store.get_creds_by_id = AsyncMock(return_value=creds)
        manager.delete = AsyncMock()

        resp = client.delete("/mcp/credentials/cred-mcp")

    assert resp.status_code == 200
    assert resp.json() == {"deleted": True, "revoked": None}
    manager.delete.assert_awaited_once_with(test_user_id, "cred-mcp")


@pytest.mark.parametrize(
    "oauth_metadata",
    [
        {"mcp_token_url": "https://auth.example.com/token"},
        {"mcp_client_id": "client-1"},
    ],
)
@pytest.mark.asyncio
async def test_an_mcp_oauth_grant_without_a_refresh_token_is_still_revoked(
    oauth_metadata: dict,
):
    # Some servers grant no refresh token: the OAuth metadata, not the refresh
    # token, is what tells the grant apart from a pasted token.
    creds = _pasted_mcp_token(
        {"mcp_server_url": "https://api.example.com/mcp", **oauth_metadata}
    )
    handler = MagicMock()
    handler.revoke_tokens = AsyncMock(return_value=True)
    with (
        patch(f"{ROUTER}.remove_all_webhooks_for_credentials", new=AsyncMock()),
        patch(f"{ROUTER}.creds_manager") as manager,
        patch(f"{ROUTER}.create_mcp_oauth_handler", return_value=handler) as create_mcp,
    ):
        manager.delete = AsyncMock()

        revoked = await delete_and_revoke_credentials(
            USER_ID, creds, ProviderName.MCP, force=True
        )

    assert revoked is True
    manager.delete.assert_awaited_once_with(USER_ID, "cred-mcp")
    create_mcp.assert_called_once_with(creds)
    handler.revoke_tokens.assert_awaited_once_with(creds)
