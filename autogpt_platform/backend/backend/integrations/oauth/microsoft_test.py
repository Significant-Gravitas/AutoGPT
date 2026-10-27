from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.microsoft import (
    MicrosoftDeviceAuthHandler,
    MicrosoftOAuthHandler,
)
from backend.integrations.oauth.microsoft_365_copilot import (
    MICROSOFT_365_COPILOT_PUBLIC_CLIENT_ID,
)
from backend.integrations.oauth.microsoft_identity import same_microsoft_account
from backend.integrations.oauth.microsoft_recovery import (
    MicrosoftAuthError,
    microsoft_recovery,
)
from backend.integrations.oauth.microsoft_services import (
    COPILOT_SCOPES,
    WORK_IQ_SCOPE,
    is_microsoft_365_copilot_credential,
)
from backend.integrations.oauth.microsoft_tokens import (
    microsoft_resource_token,
    refresh_microsoft_credentials,
)

PERSONAL_TENANT = "9188040d-6c67-4c5b-b112-36a304b66dad"
TOKENS = {
    "access_token": "graph-access",
    "refresh_token": "refresh",
    "expires_in": 3600,
    "scope": "https://graph.microsoft.com/Mail.ReadWrite User.Read openid profile",
    "id_token": "id-token",
}


def _app(mocker, client_id: str = "", secret: str = "") -> None:
    secrets = mocker.patch(
        "backend.integrations.oauth.microsoft_tokens.Settings"
    ).return_value.secrets
    secrets.microsoft_client_id = client_id
    secrets.microsoft_client_secret = secret


def _http(mocker, *payloads: dict, status: int = 200) -> AsyncMock:
    responses = []
    for payload in payloads:
        response = MagicMock(status_code=status, is_success=status < 400)
        response.json.return_value = payload
        responses.append(response)
    client = AsyncMock()
    client.post = AsyncMock(side_effect=responses)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    mocker.patch("httpx.AsyncClient", return_value=context)
    return client


def _identity(mocker, tenant: str = "company", account: str = "work-id") -> None:
    async def populate(credentials, id_token, client_id):
        credentials.metadata["microsoft_identity"] = {
            "tenant_id": tenant,
            "account_id": account,
            "account_type": "personal" if tenant == PERSONAL_TENANT else "work",
        }

    mocker.patch(
        "backend.integrations.oauth.microsoft_tokens.populate_microsoft_identity",
        side_effect=populate,
    )


def _connection(
    tenant: str = "company", client_id: str = "deployment-app", scopes=None
) -> OAuth2Credentials:
    return OAuth2Credentials(
        provider="microsoft",
        access_token=SecretStr("graph-access"),
        refresh_token=SecretStr("refresh"),
        scopes=scopes or ["User.Read", "Mail.ReadWrite"],
        metadata={
            "microsoft_client_id": client_id,
            "microsoft_auth_flow": "authorization_code",
            "microsoft_identity": {
                "tenant_id": tenant,
                "account_id": "account",
                "account_type": "personal" if tenant == PERSONAL_TENANT else "work",
            },
        },
    )


def test_browser_sign_in_asks_consent_for_every_requested_resource() -> None:
    handler = MicrosoftOAuthHandler("deployment-app", "secret", "https://app/cb")
    url = urlsplit(
        handler.get_login_url(["Mail.ReadWrite", WORK_IQ_SCOPE], "state", "pkce")
    )
    query = parse_qs(url.query)
    scopes = query["scope"][0].split()

    assert url.path == "/common/oauth2/v2.0/authorize"
    assert query["client_id"] == ["deployment-app"]
    assert query["prompt"] == ["select_account"]
    assert query["code_challenge_method"] == ["S256"]
    assert {"openid", "offline_access", "User.Read", WORK_IQ_SCOPE} <= set(scopes)
    assert "client_secret" not in query
    with pytest.raises(ValueError, match="PKCE"):
        handler.get_login_url([], "state", None)


@pytest.mark.asyncio
async def test_code_exchange_redeems_graph_only_and_records_the_issuing_app(
    mocker,
) -> None:
    _app(mocker, "deployment-app", "secret")
    _identity(mocker)
    client = _http(mocker, TOKENS)
    handler = MicrosoftOAuthHandler("deployment-app", "secret", "https://app/cb")

    credentials = await handler.exchange_code_for_tokens(
        "code", ["Mail.ReadWrite", WORK_IQ_SCOPE], "verifier"
    )

    sent = client.post.await_args.kwargs["data"]
    assert WORK_IQ_SCOPE not in sent["scope"].split()
    assert sent["client_secret"] == "secret"
    assert credentials.provider == "microsoft"
    assert credentials.metadata["microsoft_client_id"] == "deployment-app"
    assert credentials.metadata["microsoft_auth_flow"] == "authorization_code"
    assert "Mail.ReadWrite" in credentials.scopes


@pytest.mark.asyncio
async def test_sign_in_codes_use_the_public_client_without_a_secret(mocker) -> None:
    _app(mocker)
    _identity(mocker, PERSONAL_TENANT, "personal-id")
    initiation = {
        "device_code": "device",
        "user_code": "ABCD",
        "verification_uri": "https://microsoft.com/devicelogin",
        "expires_in": 900,
    }
    client = _http(mocker, initiation, TOKENS)
    handler = MicrosoftDeviceAuthHandler()

    await handler.initiate_device_auth([])
    result = await handler.poll_for_tokens("device")

    for call in client.post.await_args_list:
        assert (
            call.kwargs["data"]["client_id"] == MICROSOFT_365_COPILOT_PUBLIC_CLIENT_ID
        )
        assert "client_secret" not in call.kwargs["data"]
    assert result.status == "approved" and result.credentials is not None
    assert result.credentials.metadata["microsoft_auth_flow"] == "device_code"


@pytest.mark.asyncio
async def test_sign_in_codes_with_a_deployment_app_send_its_secret(mocker) -> None:
    _app(mocker, "deployment-app", "secret")
    _identity(mocker)
    client = _http(mocker, TOKENS)

    await MicrosoftDeviceAuthHandler().poll_for_tokens("device")

    assert client.post.await_args.kwargs["data"]["client_secret"] == "secret"


@pytest.mark.asyncio
async def test_refresh_uses_the_app_that_issued_the_grant(mocker) -> None:
    _app(mocker, "deployment-app", "secret")
    client = _http(mocker, {**TOKENS, "id_token": None}, {**TOKENS, "id_token": None})

    await refresh_microsoft_credentials(_connection(client_id="deployment-app"))
    await refresh_microsoft_credentials(
        _connection(PERSONAL_TENANT, MICROSOFT_365_COPILOT_PUBLIC_CLIENT_ID)
    )

    confidential, public = client.post.await_args_list
    assert confidential.args[0].endswith("/company/oauth2/v2.0/token")
    assert confidential.kwargs["data"]["client_secret"] == "secret"
    assert public.args[0].endswith("/consumers/oauth2/v2.0/token")
    assert "client_secret" not in public.kwargs["data"]


@pytest.mark.asyncio
async def test_work_iq_token_comes_from_the_same_grant(mocker) -> None:
    _app(mocker, "deployment-app", "secret")
    client = _http(mocker, {"access_token": "work-iq-access"})

    token = await microsoft_resource_token(_connection(), WORK_IQ_SCOPE)

    assert token == "work-iq-access"
    sent = client.post.await_args.kwargs["data"]
    assert sent["scope"] == WORK_IQ_SCOPE
    assert sent["grant_type"] == "refresh_token"
    with pytest.raises(ValueError, match="work or school"):
        await microsoft_resource_token(_connection(PERSONAL_TENANT), WORK_IQ_SCOPE)


@pytest.mark.asyncio
async def test_token_errors_carry_a_safe_recovery(mocker) -> None:
    _app(mocker, "deployment-app", "secret")
    _http(
        mocker,
        {"error": "invalid_grant", "error_description": "AADSTS650052: secret-x"},
        status=400,
    )
    with pytest.raises(MicrosoftAuthError) as error:
        await microsoft_resource_token(_connection(), WORK_IQ_SCOPE)
    assert error.value.recovery.reason == "tenant_setup_required"
    assert "secret-x" not in error.value.recovery.model_dump_json()


def test_copilot_runs_on_work_connections_and_earlier_copilot_connections() -> None:
    work = _connection(scopes=["User.Read", *COPILOT_SCOPES])
    personal = _connection(PERSONAL_TENANT, scopes=["User.Read", *COPILOT_SCOPES])
    earlier = work.model_copy(update={"provider": "microsoft_365_copilot"})

    assert is_microsoft_365_copilot_credential(work)
    assert is_microsoft_365_copilot_credential(earlier)
    assert not is_microsoft_365_copilot_credential(personal)
    assert not is_microsoft_365_copilot_credential(_connection())


@pytest.mark.parametrize(
    "code,reason",
    [
        (650052, "tenant_setup_required"),
        (65001, "admin_consent_required"),
        (65004, "access_denied"),
        (90094, "admin_consent_required"),
        (7000218, "app_setup_required"),
    ],
)
def test_recovery_keeps_a_safe_reason_without_upstream_text(code, reason) -> None:
    result = microsoft_recovery(description=f"AADSTS{code}: secret-token")
    assert result.reason == reason
    assert "secret-token" not in result.model_dump_json()


def test_same_email_never_merges_personal_and_work_identities() -> None:
    personal, work = _connection(PERSONAL_TENANT), _connection("company")
    personal.metadata["microsoft_identity"]["account_id"] = "personal-id"
    personal.username = work.username = "same@company.example"

    assert not same_microsoft_account(personal, work)
    assert same_microsoft_account(work, work.model_copy())
