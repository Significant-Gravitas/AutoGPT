import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.mcp.client import MCPCallResult
from backend.copilot.tools.logistics_partner import (
    LogisticsPartnerTool,
    _create_access_token,
)
from backend.copilot.tools.models import ResponseType

_MODULE = "backend.copilot.tools.logistics_partner"


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


@pytest.fixture
def session():
    return SimpleNamespace(
        session_id="session-1",
        organization_id="autogpt-org-77",
        metadata=SimpleNamespace(
            source_platform="logistics-partner",
            external_account_id="fd-account-77",
            external_capabilities=["jobs.read", "reports.read"],
        ),
    )


def test_token_contains_server_derived_tenant_claims():
    token = _create_access_token(
        secret="test-secret",
        user_id="user-1",
        organization_id="org-1",
        external_account_id="fd-account-77",
        capabilities=["reports.read", "jobs.read"],
        now=1_000,
    )
    payload = json.loads(
        base64.urlsafe_b64decode(token.split(".", 1)[0] + "==").decode()
    )

    assert payload == {
        "version": 1,
        "partner_id": "logistics-partner",
        "user_id": "user-1",
        "organization_id": "org-1",
        "external_account_id": "fd-account-77",
        "capabilities": ["jobs.read", "reports.read"],
        "exp": 1_060,
    }


def test_schema_has_no_tenant_input():
    schema = LogisticsPartnerTool().parameters

    assert set(schema["properties"]) == {"report"}
    assert schema["additionalProperties"] is False


def test_tool_availability_requires_url_and_secret(monkeypatch):
    tool = LogisticsPartnerTool()
    monkeypatch.delenv("LOGISTICS_PARTNER_MCP_URL", raising=False)
    monkeypatch.delenv("LOGISTICS_PARTNER_MCP_SHARED_SECRET", raising=False)
    assert tool.is_available is False

    monkeypatch.setenv("LOGISTICS_PARTNER_MCP_URL", "http://partner-mcp/mcp")
    monkeypatch.setenv("LOGISTICS_PARTNER_MCP_SHARED_SECRET", "secret")
    assert tool.is_available is True


@pytest.mark.asyncio
async def test_queries_mcp_with_session_tenant(session, monkeypatch):
    monkeypatch.setenv("LOGISTICS_PARTNER_MCP_URL", "http://partner-mcp/mcp")
    monkeypatch.setenv("LOGISTICS_PARTNER_MCP_SHARED_SECRET", "test-shared-secret")
    client = SimpleNamespace(
        initialize=AsyncMock(),
        call_tool=AsyncMock(
            return_value=MCPCallResult(
                content=[
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "account": {
                                    "id": "fd-account-77",
                                    "name": "Northstar Freight",
                                },
                                "active_jobs": 148,
                            }
                        ),
                    }
                ]
            )
        ),
        close=AsyncMock(),
    )

    with patch(_MODULE + ".MCPClient", return_value=client) as client_factory:
        result = await LogisticsPartnerTool()._execute(
            "user-1", session, report="operations_summary"
        )

    assert result.type == ResponseType.MCP_TOOL_OUTPUT
    assert result.result["account"]["name"] == "Northstar Freight"
    client.call_tool.assert_awaited_once_with("get_operations_summary", {})
    assert client_factory.call_args.args == ("http://partner-mcp/mcp",)
    assert client_factory.call_args.kwargs["trusted_origins"] == [
        "http://partner-mcp/mcp"
    ]
    client.close.assert_awaited_once()
    auth_token = client_factory.call_args.kwargs["auth_token"]
    payload = json.loads(
        base64.urlsafe_b64decode(auth_token.split(".", 1)[0] + "==").decode()
    )
    assert payload["external_account_id"] == "fd-account-77"
    assert payload["organization_id"] == "autogpt-org-77"
    assert payload["capabilities"] == ["jobs.read", "reports.read"]


@pytest.mark.asyncio
async def test_rejects_non_partner_session_without_network(session):
    session.metadata.source_platform = "discord"

    with patch(_MODULE + ".MCPClient") as client_factory:
        result = await LogisticsPartnerTool()._execute(
            "user-1", session, report="operations_summary"
        )

    assert result.type == ResponseType.ERROR
    assert result.error == "partner_session_required"
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_rejects_missing_external_account_without_network(session):
    session.metadata.external_account_id = None

    with patch(_MODULE + ".MCPClient") as client_factory:
        result = await LogisticsPartnerTool()._execute(
            "user-1", session, report="operations_summary"
        )

    assert result.type == ResponseType.ERROR
    assert result.error == "partner_session_required"
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_closes_mcp_session_when_call_fails(session, monkeypatch):
    monkeypatch.setenv("LOGISTICS_PARTNER_MCP_URL", "http://partner-mcp/mcp")
    monkeypatch.setenv("LOGISTICS_PARTNER_MCP_SHARED_SECRET", "test-shared-secret")
    client = SimpleNamespace(
        initialize=AsyncMock(),
        call_tool=AsyncMock(side_effect=RuntimeError("unavailable")),
        close=AsyncMock(),
    )

    with patch(_MODULE + ".MCPClient", return_value=client):
        with pytest.raises(RuntimeError, match="unavailable"):
            await LogisticsPartnerTool()._execute(
                "user-1", session, report="operations_summary"
            )

    client.close.assert_awaited_once()
