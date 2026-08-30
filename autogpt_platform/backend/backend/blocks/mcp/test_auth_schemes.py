"""Focused tests for MCP manual authentication schemes."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_user_id

from backend.api.features.mcp.routes import router
from backend.blocks.mcp.block import TEST_CREDENTIALS_INPUT, MCPToolBlock
from backend.blocks.mcp.client import MCPClient, normalize_mcp_authorization
from backend.executor.utils import _validate_node_input_credentials


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("secret-token", "Bearer secret-token"),
        ("  secret-token  ", "Bearer secret-token"),
        ("token:with:colons", "Bearer token:with:colons"),
        ("Bearer secret-token", "Bearer secret-token"),
        ("bearer secret-token", "Bearer secret-token"),
        ("Basic cGstbGYtYWJjZA==", "Basic cGstbGYtYWJjZA=="),
        (
            "Authorization: Basic cGstbGYtYWJjZA==",
            "Basic cGstbGYtYWJjZA==",
        ),
        (
            "Authorization : Basic cGstbGYtYWJjZA==",
            "Basic cGstbGYtYWJjZA==",
        ),
        ("AUTHORIZATION:bearer secret-token", "Bearer secret-token"),
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
        "Bearer",
        "Authorization:",
        "Authorization :  ",
        "Authorization: Basic",
        "Authorization: Bearer",
        "Authorization: Digest abc",
    ],
)
def test_normalize_mcp_authorization_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_mcp_authorization(value)


def test_normalize_mcp_authorization_preserves_bare_credential_with_spaces() -> None:
    assert normalize_mcp_authorization("orgid api-key") == "Bearer orgid api-key"


@pytest.mark.parametrize(
    "control_character", ["\x00", "\t", "\n", "\r", "\x1f", "\x7f"]
)
def test_normalize_mcp_authorization_rejects_control_characters(
    control_character: str,
) -> None:
    with pytest.raises(ValueError, match="single line"):
        normalize_mcp_authorization(f"Bearer token{control_character}X-Evil: injected")


def test_mcp_client_preserves_explicit_basic_auth() -> None:
    client = MCPClient(
        "https://mcp.example.com/mcp",
        auth_token="Basic cGstbGYtYWJjZA==",
    )
    assert client._build_headers()["Authorization"] == "Basic cGstbGYtYWJjZA=="


def test_mcp_client_keeps_bare_tokens_as_bearer() -> None:
    client = MCPClient("https://mcp.example.com/mcp", auth_token="legacy-token")
    assert client._build_headers()["Authorization"] == "Bearer legacy-token"


def test_mcp_client_omits_auth_when_not_provided() -> None:
    client = MCPClient("https://mcp.example.com/mcp")
    assert "Authorization" not in client._build_headers()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_default",
    [{}, {"credentials": TEST_CREDENTIALS_INPUT}],
    ids=["anonymous", "bound-credential"],
)
async def test_schema_optional_mcp_credentials_do_not_skip_execution(
    input_default: dict,
) -> None:
    """MCP's schema-level optional credential must not skip the whole node."""
    node = MagicMock()
    node.id = "mcp-node"
    node.block = MCPToolBlock()
    node.credentials_optional = False
    node.input_default = input_default
    graph = MagicMock(nodes=[node])

    errors, nodes_to_skip = await _validate_node_input_credentials(
        graph=graph,
        user_id="test-user-id",
        nodes_input_masks=None,
    )

    assert errors == {}
    assert nodes_to_skip == set()


@pytest.mark.parametrize("auth_token", ["", " ", "   "])
def test_mcp_client_rejects_non_none_blank_auth(auth_token: str) -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        MCPClient("https://mcp.example.com/mcp", auth_token=auth_token)


app = fastapi.FastAPI()
app.include_router(router)
app.dependency_overrides[get_user_id] = lambda: "test-user-id"


@pytest_asyncio.fixture(scope="module")
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio(loop_scope="session")
async def test_discover_rejects_empty_explicit_auth_token(
    client: httpx.AsyncClient,
) -> None:
    with patch(
        "backend.api.features.mcp.routes.auto_lookup_mcp_credential",
        new_callable=AsyncMock,
    ) as lookup_credential:
        response = await client.post(
            "/discover-tools",
            json={
                "server_url": "https://mcp.example.com/mcp",
                "auth_token": "",
            },
        )

    assert response.status_code == 422
    lookup_credential.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("token", "expected", "scheme"),
    [
        (
            "Authorization: Basic cGstbGYtYWJjZA==",
            "Basic cGstbGYtYWJjZA==",
            "basic",
        ),
        ("orgid api-key", "Bearer orgid api-key", "bearer"),
    ],
)
async def test_store_manual_credential(
    client: httpx.AsyncClient,
    token: str,
    expected: str,
    scheme: str,
) -> None:
    with (
        patch(
            "backend.api.features.mcp.routes.validate_url_host",
            new_callable=AsyncMock,
        ),
        patch("backend.api.features.mcp.routes.creds_manager") as mock_cm,
    ):
        create_credential = AsyncMock()
        mock_cm.store.get_creds_by_provider = AsyncMock(return_value=[])
        mock_cm.create = create_credential

        response = await client.post(
            "/token",
            json={
                "server_url": "https://mcp.example.com/mcp",
                "token": token,
            },
        )

    assert response.status_code == 200
    create_credential.assert_awaited_once()
    create_call = create_credential.await_args
    assert create_call is not None
    created = create_call.args[1]
    assert created.access_token.get_secret_value() == expected
    assert created.metadata["mcp_auth_scheme"] == scheme
    assert response.json()["mcp_auth_scheme"] == scheme


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "control_character", ["\x00", "\t", "\n", "\r", "\x1f", "\x7f"]
)
async def test_store_rejects_header_injection(
    client: httpx.AsyncClient,
    control_character: str,
) -> None:
    with patch(
        "backend.api.features.mcp.routes.validate_url_host",
        new_callable=AsyncMock,
    ) as validate_url_host:
        response = await client.post(
            "/token",
            json={
                "server_url": "https://mcp.example.com/mcp",
                "token": f"Bearer token{control_character}X-Evil: injected",
            },
        )

    assert response.status_code == 422
    assert "single line" in response.json()["detail"].lower()
    validate_url_host.assert_not_awaited()
