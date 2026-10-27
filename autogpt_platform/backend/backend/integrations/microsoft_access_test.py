from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.microsoft_access import check_microsoft_access
from backend.integrations.oauth.microsoft_recovery import (
    MicrosoftAuthError,
    microsoft_recovery,
)
from backend.integrations.oauth.microsoft_services import (
    COPILOT_SCOPES,
    MICROSOFT_SERVICE_SCOPES,
)

PERSONAL_TENANT = "9188040d-6c67-4c5b-b112-36a304b66dad"
BASE = ["User.Read", "Mail.ReadWrite", "Mail.Send", "Calendars.ReadWrite"]


def _connection(tenant: str, scopes: list[str]) -> OAuth2Credentials:
    return OAuth2Credentials(
        provider="microsoft",
        access_token=SecretStr("graph"),
        refresh_token=SecretStr("refresh"),
        scopes=scopes,
        metadata={
            "microsoft_client_id": "deployment-app",
            "microsoft_identity": {
                "tenant_id": tenant,
                "account_id": "account",
                "account_type": "personal" if tenant == PERSONAL_TENANT else "work",
            },
        },
    )


@pytest.fixture(autouse=True)
def graph(mocker) -> AsyncMock:
    client = AsyncMock()
    client.get = AsyncMock(return_value=MagicMock(is_success=True))
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    mocker.patch("httpx.AsyncClient", return_value=context)
    return client


def _work_iq(mocker, *, error: int | None = None, tools: list | None = None):
    token = mocker.patch(
        "backend.integrations.microsoft_access.microsoft_resource_token",
        AsyncMock(return_value="work-iq-token"),
    )
    if error is not None:
        token.side_effect = MicrosoftAuthError(
            microsoft_recovery(description=f"AADSTS{error}: detail")
        )
    mcp = mocker.patch("backend.integrations.microsoft_access.MCPClient").return_value
    mcp.list_tools = AsyncMock(return_value=tools or [])
    mcp.close = AsyncMock()
    return token


def _approval_app(mocker) -> None:
    mocker.patch(
        "backend.integrations.microsoft_access.browser_sign_in_available",
        return_value=True,
    )
    mocker.patch(
        "backend.integrations.microsoft_access.microsoft_app"
    ).return_value.client_id = "deployment-app"
    config = mocker.patch("backend.integrations.microsoft_access.Settings")
    config.return_value.config.frontend_base_url = "https://app.example"


def _statuses(result) -> dict[str, str]:
    return {item.service: item.status for item in result.capabilities}


@pytest.mark.asyncio
async def test_personal_accounts_get_mail_calendar_and_files_only(mocker) -> None:
    token = _work_iq(mocker)

    result = await check_microsoft_access(
        _connection(PERSONAL_TENANT, [*BASE, "Files.ReadWrite"])
    )

    assert result.account_type == "personal"
    assert _statuses(result) == {
        "mail": "available",
        "calendar": "available",
        "files": "available",
        "teams": "work_account_only",
        "copilot": "work_account_only",
        "work_iq": "work_account_only",
    }
    token.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_permissions_name_what_to_ask_for(mocker) -> None:
    _work_iq(mocker, error=65001)

    result = await check_microsoft_access(_connection("company", BASE))
    by_service = {item.service: item for item in result.capabilities}

    assert by_service["files"].status == "permission_needed"
    assert by_service["files"].scopes == ["Files.ReadWrite"]
    assert by_service["teams"].scopes == ["Chat.ReadWrite"]
    assert by_service["copilot"].status == "permission_needed"
    assert by_service["work_iq"].status == "permission_needed"
    assert result.verified


@pytest.mark.asyncio
async def test_work_accounts_with_every_permission_are_ready(mocker) -> None:
    _work_iq(mocker, tools=[MagicMock()])
    scopes = [*BASE, "Files.ReadWrite", "Chat.ReadWrite", *COPILOT_SCOPES]

    result = await check_microsoft_access(_connection("company", scopes))

    assert set(_statuses(result).values()) == {"available"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,status",
    [(650052, "setup_required"), (90094, "admin_required"), (50000, "unavailable")],
)
async def test_work_iq_reports_what_its_organization_still_needs(
    mocker, error, status
) -> None:
    _work_iq(mocker, error=error)

    result = await check_microsoft_access(_connection("company", BASE))

    assert _statuses(result)["work_iq"] == status
    assert _statuses(result)["mail"] == "available"


@pytest.mark.asyncio
async def test_failed_probe_does_not_hide_other_services(mocker, graph) -> None:
    _work_iq(mocker, tools=[MagicMock()])
    graph.get.side_effect = [
        MagicMock(is_success=False),
        MagicMock(is_success=True),
        MagicMock(is_success=True),
    ]

    result = await check_microsoft_access(_connection(PERSONAL_TENANT, BASE))

    assert _statuses(result)["mail"] == "unavailable"
    assert _statuses(result)["calendar"] == "available"


@pytest.mark.asyncio
async def test_admins_can_approve_every_work_service_at_once(mocker) -> None:
    _work_iq(mocker, error=650052)
    _approval_app(mocker)

    result = await check_microsoft_access(_connection("company", BASE))
    personal = await check_microsoft_access(_connection(PERSONAL_TENANT, BASE))

    url = result.admin_consent_url or ""
    assert url.startswith(
        "https://login.microsoftonline.com/company/v2.0/adminconsent?"
    )
    assert "ChannelMessage.Read.All" in url and "Chat.ReadWrite" in url
    # Work IQ still needs Microsoft's own setup, which would fail the approval.
    assert "workiq" not in url
    assert personal.admin_consent_url is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [65001, 90094])
async def test_organization_approval_leaves_work_iq_to_its_own_approval(
    mocker, error
) -> None:
    # Asking for Work IQ in a tenant without its service fails the whole
    # approval (AADSTS650052, live on 10-27), and "needs your permission" or
    # "needs your admin" can't tell that tenant apart.
    _work_iq(mocker, error=error)
    _approval_app(mocker)

    result = await check_microsoft_access(_connection("company", BASE))

    url = result.admin_consent_url or ""
    assert "Chat.ReadWrite" in url
    assert "workiq" not in url


@pytest.mark.asyncio
async def test_no_organization_approval_when_only_work_iq_is_pending(mocker) -> None:
    _work_iq(mocker, error=65001)
    _approval_app(mocker)
    every_graph_scope = [
        "User.Read",
        *(
            scope
            for service, scopes in MICROSOFT_SERVICE_SCOPES.items()
            if service != "work_iq"
            for scope in scopes
        ),
    ]

    result = await check_microsoft_access(_connection("company", every_graph_scope))

    assert _statuses(result)["work_iq"] == "permission_needed"
    assert result.admin_consent_url is None


@pytest.mark.asyncio
async def test_organization_approval_only_for_this_deployments_app(mocker) -> None:
    _work_iq(mocker, error=650052)
    _approval_app(mocker)
    public = _connection("company", BASE)
    public.metadata["microsoft_client_id"] = "public-sign-in-code-app"

    result = await check_microsoft_access(public)

    # Approving the deployment's app would grant nothing to a connection
    # another app issued.
    assert result.admin_consent_url is None
