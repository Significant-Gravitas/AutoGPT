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
from backend.blocks.mcp.helpers import mcp_authorization_header
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
    "control_character", ["\x00", "\t", "\n", "\r", "\x1f", "\x7f"]
)
def test_normalize_mcp_authorization_rejects_control_characters(
    control_character: str,
) -> None:
    with pytest.raises(ValueError, match="single line"):
        normalize_mcp_authorization(f"Bearer token{control_character}X-Evil: injected")


def test_mcp_client_sends_the_authorization_it_was_given() -> None:
    """The transport sends its value verbatim and parses nothing.

    Normalizing here as well as at the entry point made the header depend on
    how many layers a value had crossed: a stored ``"Bearer orgid api-key"``
    came back out as ``"Bearer Bearer orgid api-key"``.
    """
    for authorization in (
        "Basic cGstbGYtYWJjZA==",
        "Bearer legacy-token",
        "Bearer orgid api-key",
    ):
        client = MCPClient("https://mcp.example.com/mcp", authorization=authorization)
        assert client._build_headers()["Authorization"] == authorization


def test_stored_credential_round_trips_without_double_prefixing() -> None:
    """Store → read → send must be stable for a multi-word bare credential.

    The store path canonicalizes ``"orgid api-key"`` to ``"Bearer orgid
    api-key"``; the read path must hand that to the transport untouched.
    """
    stored = OAuth2Credentials(
        provider="mcp",
        title="MCP: mcp.example.com",
        access_token=SecretStr(normalize_mcp_authorization("orgid api-key")),
        scopes=[],
        metadata={
            "mcp_server_url": "https://mcp.example.com/mcp",
            "mcp_auth_scheme": "bearer",
        },
    )
    client = MCPClient(
        "https://mcp.example.com/mcp",
        authorization=mcp_authorization_header(stored),
    )
    assert client._build_headers()["Authorization"] == "Bearer orgid api-key"


def test_legacy_row_without_scheme_metadata_is_sent_as_bearer() -> None:
    """A pre-Basic row holds a raw token and must never be parsed.

    Parsing it is what let ``"basic auth key"`` become ``"Basic auth key"`` —
    scheme flipped and the first word silently dropped from the secret.
    """
    legacy = OAuth2Credentials(
        provider="mcp",
        title="MCP: mcp.example.com",
        access_token=SecretStr("basic auth key"),
        scopes=[],
        metadata={"mcp_server_url": "https://mcp.example.com/mcp"},
    )
    assert mcp_authorization_header(legacy) == "Bearer basic auth key"


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


@pytest.mark.parametrize("value", ["", " ", "   "])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        normalize_mcp_authorization(value)


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
        metadata={
            "mcp_server_url": "https://mcp.example.com/mcp",
            "mcp_auth_scheme": "basic",
        },
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
    assert call_tool.await_args.kwargs["authorization"] == "Basic cWE6YmFzaWM="
    assert ("result", "ok") in outputs
