from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.api.external.v1 import integrations as external
from backend.api.features.integrations import router as routes
from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.microsoft import MicrosoftDeviceAuthHandler
from backend.integrations.oauth.microsoft_recovery import (
    MicrosoftAuthError,
    microsoft_recovery,
)
from backend.integrations.oauth.microsoft_tokens import issuing_client_id
from backend.integrations.providers import ProviderName


def credential(account_id, tenant_id="tenant", username="same@company.example"):
    return OAuth2Credentials(
        provider="microsoft",
        access_token="opaque",
        scopes=["User.Read"],
        username=username,
        metadata={
            "microsoft_identity": {
                "account_id": account_id,
                "tenant_id": tenant_id,
                "account_type": "work",
            }
        },
    )


@pytest.mark.asyncio
async def test_same_email_different_account_does_not_replace_existing(mocker):
    old, new = credential("personal", "consumer"), credential("work", "company")
    mocker.patch.object(
        routes.creds_manager.store,
        "get_creds_by_provider",
        AsyncMock(return_value=[old]),
    )
    create = mocker.patch.object(routes.creds_manager, "create", AsyncMock())
    update = mocker.patch.object(routes.creds_manager, "update", AsyncMock())
    result = await routes._merge_or_create_credential(
        "owner", ProviderName.MICROSOFT, new, None
    )
    assert result.id == new.id
    create.assert_awaited_once_with("owner", new)
    update.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_upgrade_cannot_switch_same_email_identity(mocker):
    old, new = credential("personal", "consumer"), credential("work", "company")
    mocker.patch.object(
        routes.creds_manager.store, "get_creds_by_id", AsyncMock(return_value=old)
    )
    update = mocker.patch.object(routes.creds_manager, "update", AsyncMock())
    with pytest.raises(HTTPException, match="different Microsoft account"):
        await routes._upgrade_existing_credential("owner", old.id, new)
    update.assert_not_awaited()


@pytest.mark.asyncio
async def test_upgrade_with_fewer_permissions_cannot_switch_identity(mocker):
    # The upgrade names a connection; another account signed in on it, with
    # less granted than the connection has, is still another account.
    old, new = credential("personal", "consumer"), credential("work", "company")
    old.scopes = ["User.Read", "Mail.ReadWrite"]
    mocker.patch.object(
        routes.creds_manager.store, "get_creds_by_id", AsyncMock(return_value=old)
    )
    create = mocker.patch.object(routes.creds_manager, "create", AsyncMock())
    update = mocker.patch.object(routes.creds_manager, "update", AsyncMock())

    with pytest.raises(HTTPException, match="different Microsoft account"):
        await routes._merge_or_create_credential(
            "owner", ProviderName.MICROSOFT, new, old.id
        )

    create.assert_not_awaited()
    update.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_renamed_account_signs_in_to_its_own_connection(mocker):
    # Microsoft's preferred_username can change; the tenant and object id
    # are what name the account.
    old = credential("work", "company", username="pat@company.example")
    new = credential("work", "company", username="pat.lee@company.example")
    mocker.patch.object(
        routes.creds_manager.store,
        "get_creds_by_provider",
        AsyncMock(return_value=[old]),
    )
    mocker.patch.object(
        routes.creds_manager.store, "get_creds_by_id", AsyncMock(return_value=old)
    )
    create = mocker.patch.object(routes.creds_manager, "create", AsyncMock())
    update = mocker.patch.object(routes.creds_manager, "update", AsyncMock())

    result = await routes._merge_or_create_credential(
        "owner", ProviderName.MICROSOFT, new, None
    )

    assert result.id == old.id
    assert result.username == "pat.lee@company.example"
    create.assert_not_awaited()
    update.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_renamed_account_can_still_add_permissions(mocker):
    old = credential("work", "company", username="pat@company.example")
    new = credential("work", "company", username="pat.lee@company.example")
    mocker.patch.object(
        routes.creds_manager.store, "get_creds_by_id", AsyncMock(return_value=old)
    )
    update = mocker.patch.object(routes.creds_manager, "update", AsyncMock())

    result = await routes._upgrade_existing_credential("owner", old.id, new)

    assert result.id == old.id
    update.assert_awaited_once()


@pytest.mark.asyncio
async def test_verification_scopes_credential_lookup_to_signed_in_user(mocker):
    lookup = mocker.patch.object(
        routes.creds_manager, "get", AsyncMock(return_value=None)
    )
    probe = mocker.patch.object(routes, "check_microsoft_access", AsyncMock())
    with pytest.raises(HTTPException) as error:
        await routes.verify_microsoft_connection(
            "someone-elses-credential", "signed-in-user"
        )
    assert error.value.status_code == 404
    lookup.assert_awaited_once_with("signed-in-user", "someone-elses-credential")
    probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_sign_in_code_upgrade_refreshes_with_the_app_that_issued_it(mocker):
    browser = credential("account")
    browser.metadata["microsoft_client_id"] = "deployment-app"
    browser.metadata["microsoft_auth_flow"] = "authorization_code"
    device = credential("account")
    device.metadata["microsoft_client_id"] = "public-app"
    device.metadata["microsoft_auth_flow"] = "device_code"
    mocker.patch.object(
        routes.creds_manager.store, "get_creds_by_id", AsyncMock(return_value=browser)
    )
    mocker.patch.object(routes.creds_manager, "update", AsyncMock())

    merged = await routes._upgrade_existing_credential("owner", browser.id, device)

    assert merged.id == browser.id
    assert issuing_client_id(merged) == "public-app"
    assert isinstance(
        await routes.creds_manager._get_oauth_handler(merged),
        MicrosoftDeviceAuthHandler,
    )


def test_browser_sign_in_is_refused_until_the_deployment_turns_it_on(mocker):
    mocker.patch.object(routes, "browser_sign_in_available", return_value=False)

    with pytest.raises(HTTPException) as refused:
        routes._get_provider_oauth_handler(mocker.Mock(), ProviderName.MICROSOFT)

    assert refused.value.status_code == 400
    assert "sign-in code" in refused.value.detail


def test_external_api_points_microsoft_to_settings():
    with pytest.raises(HTTPException) as refused:
        external._get_oauth_handler_for_external(
            "microsoft", "https://client.example/callback"
        )

    assert refused.value.status_code == 400
    assert "Settings" in refused.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        MicrosoftAuthError(microsoft_recovery(description="AADSTS70043: expired")),
        ValueError("Reconnect Microsoft to renew access."),
    ],
)
async def test_check_answers_a_connection_that_can_no_longer_refresh(mocker, error):
    mocker.patch.object(routes.creds_manager, "get", AsyncMock(side_effect=error))

    with pytest.raises(HTTPException) as refused:
        await routes.verify_microsoft_connection("work-id", "user-1")

    assert refused.value.status_code == 400


@pytest.mark.asyncio
async def test_work_iq_scopes_a_graph_token_cannot_carry_are_not_reported_missing(
    mocker,
):
    failure = mocker.patch.object(routes, "report_credential_failure")
    handler = mocker.Mock()
    handler.handle_default_scopes = lambda scopes: scopes
    granted = credential("work-id")
    granted.scopes = ["Chat.ReadWrite"]
    handler.exchange_code_for_tokens = AsyncMock(return_value=granted)
    state = mocker.Mock(
        scopes=["Chat.ReadWrite", "api://workiq.svc.cloud.microsoft/WorkIQAgent.Ask"],
        code_verifier=None,
    )

    await routes._exchange_code_for_credentials(
        handler, ProviderName.MICROSOFT, "code", state
    )

    failure.assert_not_called()
