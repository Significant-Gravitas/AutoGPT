"""Focused tests for MCP manual authentication schemes."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_user_id
from pydantic import SecretStr

from backend.api.features.mcp.routes import router
from backend.blocks.mcp.block import TEST_CREDENTIALS_INPUT, MCPToolBlock
from backend.blocks.mcp.client import MCPClient, normalize_mcp_authorization
from backend.data.model import OAuth2Credentials
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
    "value",
    [
        "basic auth key",
        "bearer token pair",
        "Basic org id key",
    ],
)
def test_normalize_mcp_authorization_keeps_multi_word_bare_credentials_intact(
    value: str,
) -> None:
    """A bare multi-word credential is not a scheme-prefixed one.

    RFC 7235 credentials carry no internal whitespace, so a first word of
    "basic"/"bearer" followed by more than one word is part of the secret.
    Reading it as a scheme would both flip the scheme and silently drop that
    first word from a credential stored before this feature existed.
    """
    assert normalize_mcp_authorization(value) == f"Bearer {value}"


def test_normalize_mcp_authorization_rejects_multi_word_explicit_header() -> None:
    """An explicit Authorization header fails closed rather than falling back."""
    with pytest.raises(ValueError):
        normalize_mcp_authorization("Authorization: Basic user pass")


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


@pytest.mark.asyncio
async def test_block_uses_the_credential_it_was_given():
    """An injected credential is used as-is; auto-lookup is the fallback only.

    Pins the block half of the "is the bound credential actually used?"
    question: given one, ``run`` must not go looking for another.
    """
    block = MCPToolBlock()
    credentials = OAuth2Credentials(
        provider="mcp",
        title="MCP: mcp.example.com",
        access_token=SecretStr("Basic cWE6YmFzaWM="),
        scopes=[],
        metadata={"mcp_server_url": "https://mcp.example.com/mcp"},
    )

    with (
        patch.object(block, "_call_mcp_tool", new_callable=AsyncMock) as call_tool,
        patch(
            "backend.blocks.mcp.block.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
        ) as auto_lookup,
    ):
        call_tool.return_value = "ok"
        outputs = [
            output
            async for output in block.run(
                MCPToolBlock.Input(
                    server_url="https://mcp.example.com/mcp",
                    selected_tool="a_tool",
                    tool_arguments={},
                ),
                user_id="test-user-id",
                credentials=credentials,
            )
        ]

    auto_lookup.assert_not_awaited()
    assert call_tool.await_args is not None
    assert call_tool.await_args.kwargs["auth_token"] == "Basic cWE6YmFzaWM="
    assert ("result", "ok") in outputs


@pytest.mark.asyncio
async def test_block_drops_a_credential_it_cannot_send():
    """A malformed stored credential is invalidated with a reconnect message.

    The route and the copilot both do this; without it here the same broken row
    surfaced as a bare "MCP tool call failed: …" and stayed stored, so every
    later run failed identically with no way out.
    """
    block = MCPToolBlock()
    credentials = OAuth2Credentials(
        provider="mcp",
        title="MCP: mcp.example.com",
        access_token=SecretStr("Authorization: Digest nope"),
        scopes=[],
        metadata={"mcp_server_url": "https://mcp.example.com/mcp"},
    )

    with (
        patch.object(block, "_call_mcp_tool", new_callable=AsyncMock) as call_tool,
        patch(
            "backend.blocks.mcp.block.invalidate_mcp_credential",
            new_callable=AsyncMock,
        ) as invalidate,
    ):
        outputs = [
            output
            async for output in block.run(
                MCPToolBlock.Input(
                    server_url="https://mcp.example.com/mcp",
                    selected_tool="a_tool",
                    tool_arguments={},
                ),
                user_id="test-user-id",
                credentials=credentials,
            )
        ]

    call_tool.assert_not_awaited()
    invalidate.assert_awaited_once_with("test-user-id", credentials.id)
    assert outputs == [
        (
            "error",
            "The stored credential for this MCP server is no longer usable "
            "and has been removed. Please reconnect the server.",
        )
    ]
