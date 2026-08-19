"""Focused tests for MCP manual authentication schemes."""

from unittest.mock import AsyncMock, patch

import fastapi
import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_user_id

from backend.api.features.mcp.routes import router
from backend.blocks.mcp.client import MCPClient, normalize_mcp_authorization


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("secret-token", "Bearer secret-token"),
        ("Bearer secret-token", "Bearer secret-token"),
        ("bearer secret-token", "Bearer secret-token"),
        ("Basic cGstbGYtYWJjZA==", "Basic cGstbGYtYWJjZA=="),
        (
            "Authorization: Basic cGstbGYtYWJjZA==",
            "Basic cGstbGYtYWJjZA==",
        ),
    ],
)
def test_normalize_mcp_authorization(value: str, expected: str) -> None:
    assert normalize_mcp_authorization(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "Basic",
        "Authorization:",
        "Authorization: Digest abc",
        "Bearer token\nX-Evil: injected",
    ],
)
def test_normalize_mcp_authorization_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_mcp_authorization(value)


def test_mcp_client_preserves_explicit_basic_auth() -> None:
    client = MCPClient(
        "https://mcp.example.com/mcp",
        auth_token="Basic cGstbGYtYWJjZA==",
    )
    assert client._build_headers()["Authorization"] == "Basic cGstbGYtYWJjZA=="


def test_mcp_client_keeps_bare_tokens_as_bearer() -> None:
    client = MCPClient("https://mcp.example.com/mcp", auth_token="legacy-token")
    assert client._build_headers()["Authorization"] == "Bearer legacy-token"


app = fastapi.FastAPI()
app.include_router(router)
app.dependency_overrides[get_user_id] = lambda: "test-user-id"


@pytest_asyncio.fixture(scope="module")
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio(loop_scope="session")
async def test_store_basic_credential(client) -> None:
    with (
        patch(
            "backend.api.features.mcp.routes.validate_url_host",
            new_callable=AsyncMock,
        ),
        patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
    ):
        mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])
        mock_cm.create = AsyncMock()

        response = await client.post(
            "/token",
            json={
                "server_url": "https://mcp.example.com/mcp",
                "token": "Authorization: Basic cGstbGYtYWJjZA==",
            },
        )

    assert response.status_code == 200
    created = mock_cm.create.await_args.args[1]
    assert created.access_token.get_secret_value() == "Basic cGstbGYtYWJjZA=="
    assert created.metadata["mcp_auth_scheme"] == "basic"


@pytest.mark.asyncio(loop_scope="session")
async def test_store_rejects_header_injection(client) -> None:
    with patch(
        "backend.api.features.mcp.routes.validate_url_host",
        new_callable=AsyncMock,
    ):
        response = await client.post(
            "/token",
            json={
                "server_url": "https://mcp.example.com/mcp",
                "token": "Bearer token\r\nX-Evil: injected",
            },
        )

    assert response.status_code == 422
    assert "single line" in response.json()["detail"].lower()
