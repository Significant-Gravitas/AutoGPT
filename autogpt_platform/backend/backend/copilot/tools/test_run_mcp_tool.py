"""Unit tests for the run_mcp_tool copilot tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.mcp.helpers import server_host

from ._test_data import make_session
from .models import (
    ErrorResponse,
    MCPToolOutputResponse,
    MCPToolsDiscoveredResponse,
    SetupRequirementsResponse,
)
from .run_mcp_tool import RunMCPToolTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_ID = "test-user-run-mcp-tool"
_SERVER_URL = "https://remote.mcpservers.org/fetch/mcp"


def _make_tool_list(*names: str):
    """Build a list of mock MCPClientTool objects."""
    tools = []
    for name in names:
        t = MagicMock()
        t.name = name
        t.description = f"Description for {name}"
        t.input_schema = {"type": "object", "properties": {}, "required": []}
        tools.append(t)
    return tools


def _make_call_result(content: list[dict], is_error: bool = False) -> MagicMock:
    result = MagicMock()
    result.is_error = is_error
    result.content = content
    return result


# ---------------------------------------------------------------------------
# server_host helper
# ---------------------------------------------------------------------------


def test_server_host_plain_url():
    assert server_host("https://mcp.example.com/mcp") == "mcp.example.com"


def test_server_host_strips_credentials():
    """netloc would expose user:pass — hostname must not."""
    assert server_host("https://user:secret@mcp.example.com/mcp") == "mcp.example.com"


def test_server_host_with_port():
    """Port should not appear in the returned hostname (hostname strips it)."""
    assert server_host("https://mcp.example.com:8080/mcp") == "mcp.example.com"


def test_server_host_invalid_url():
    """Falls back to the raw string for un-parseable URLs."""
    result = server_host("not-a-url")
    assert result == "not-a-url"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_missing_server_url_returns_error():
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)
    response = await tool._execute(user_id=_USER_ID, session=session)
    assert isinstance(response, ErrorResponse)
    assert "server_url" in response.message.lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_missing_user_id_returns_error():
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)
    response = await tool._execute(
        user_id=None, session=session, server_url=_SERVER_URL
    )
    assert isinstance(response, ErrorResponse)
    assert "authentication" in response.message.lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_ssrf_blocked_url_returns_error():
    """Private/loopback URLs must be rejected before any network call."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url",
        new_callable=AsyncMock,
        side_effect=ValueError("blocked loopback"),
    ):
        response = await tool._execute(
            user_id=_USER_ID, session=session, server_url="http://localhost/mcp"
        )

    assert isinstance(response, ErrorResponse)
    assert (
        "blocked" in response.message.lower() or "invalid" in response.message.lower()
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_credential_bearing_url_returns_error():
    """URLs with embedded user:pass@ must be rejected before any network call."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)
    response = await tool._execute(
        user_id=_USER_ID,
        session=session,
        server_url="https://user:secret@mcp.example.com/mcp",
    )
    assert isinstance(response, ErrorResponse)
    assert (
        "credential" in response.message.lower()
        or "do not include" in response.message.lower()
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_non_dict_tool_arguments_returns_error():
    """tool_arguments must be a JSON object — strings/arrays are rejected early."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url",
        new_callable=AsyncMock,
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await tool._execute(
                user_id=_USER_ID,
                session=session,
                server_url=_SERVER_URL,
                tool_name="fetch",
                tool_arguments=["this", "is", "a", "list"],  # wrong type
            )

    assert isinstance(response, ErrorResponse)
    assert "json object" in response.message.lower()


# ---------------------------------------------------------------------------
# Stage 1 — Discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_discover_tools_returns_discovered_response():
    """Calling with only server_url triggers discovery and returns tool list."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)
    mock_tools = _make_tool_list("fetch", "search")

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tools)
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                )

    assert isinstance(response, MCPToolsDiscoveredResponse)
    assert len(response.tools) == 2
    assert response.tools[0].name == "fetch"
    assert response.tools[1].name == "search"
    assert response.server_url == _SERVER_URL


@pytest.mark.asyncio(loop_scope="session")
async def test_discover_tools_with_credentials():
    """Stored credentials are passed as Bearer token to MCPClient."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    mock_creds = MagicMock()
    mock_creds.access_token = SecretStr("test-token-abc")
    mock_tools = _make_tool_list("push_notification")

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=mock_creds,
        ):
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tools)

            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
            ) as MockMCPClient:
                MockMCPClient.return_value = mock_client
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                )
                # Verify MCPClient was created with the resolved auth token
                MockMCPClient.assert_called_once_with(
                    _SERVER_URL, auth_token="test-token-abc"
                )

    assert isinstance(response, MCPToolsDiscoveredResponse)
    assert len(response.tools) == 1


# ---------------------------------------------------------------------------
# Stage 2 — Execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_tool_returns_output_response():
    """Calling with tool_name executes the tool and returns MCPToolOutputResponse."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)
    text_result = "# Example Domain\nThis domain is for examples."

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_result = _make_call_result([{"type": "text", "text": text_result}])
            mock_client = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value=mock_result)
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                    tool_name="fetch",
                    tool_arguments={"url": "https://example.com"},
                )

    assert isinstance(response, MCPToolOutputResponse)
    assert response.tool_name == "fetch"
    assert response.server_url == _SERVER_URL
    assert response.success is True
    assert text_result in response.result


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_tool_parses_json_result():
    """JSON text content items are parsed into Python objects."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_result = _make_call_result(
                [{"type": "text", "text": '{"status": "ok", "count": 42}'}]
            )
            mock_client = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value=mock_result)
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                    tool_name="status",
                    tool_arguments={},
                )

    assert isinstance(response, MCPToolOutputResponse)
    assert response.result == {"status": "ok", "count": 42}


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_tool_image_content():
    """Image content items are returned as {type, data, mimeType} dicts."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_result = _make_call_result(
                [{"type": "image", "data": "abc123==", "mimeType": "image/png"}]
            )
            mock_client = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value=mock_result)
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                    tool_name="screenshot",
                    tool_arguments={},
                )

    assert isinstance(response, MCPToolOutputResponse)
    assert response.result == {
        "type": "image",
        "data": "abc123==",
        "mimeType": "image/png",
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_tool_resource_content():
    """Resource content items are unwrapped to their resource payload."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_result = _make_call_result(
                [
                    {
                        "type": "resource",
                        "resource": {"uri": "file:///tmp/out.txt", "text": "hello"},
                    }
                ]
            )
            mock_client = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value=mock_result)
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                    tool_name="read_file",
                    tool_arguments={},
                )

    assert isinstance(response, MCPToolOutputResponse)
    assert response.result == {"uri": "file:///tmp/out.txt", "text": "hello"}


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_tool_multi_item_content():
    """Multiple content items are returned as a list."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_result = _make_call_result(
                [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ]
            )
            mock_client = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value=mock_result)
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                    tool_name="multi",
                    tool_arguments={},
                )

    assert isinstance(response, MCPToolOutputResponse)
    assert response.result == ["part one", "part two"]


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_tool_empty_content_returns_none():
    """Empty content list results in result=None."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_result = _make_call_result([])
            mock_client = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value=mock_result)
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                    tool_name="ping",
                    tool_arguments={},
                )

    assert isinstance(response, MCPToolOutputResponse)
    assert response.result is None


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_tool_returns_error_on_tool_failure():
    """When the MCP tool returns is_error=True, an ErrorResponse is returned."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_result = _make_call_result(
                [{"type": "text", "text": "Tool not found"}], is_error=True
            )
            mock_client = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value=mock_result)
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                    tool_name="nonexistent",
                    tool_arguments={},
                )

    assert isinstance(response, ErrorResponse)
    assert "nonexistent" in response.message


# ---------------------------------------------------------------------------
# Auth / credential flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_auth_required_without_creds_returns_setup_requirements():
    """HTTP 401 from MCP with no stored creds → SetupRequirementsResponse."""
    from backend.util.request import HTTPClientError

    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,  # No stored credentials
        ):
            mock_client = AsyncMock()
            mock_client.initialize = AsyncMock(
                side_effect=HTTPClientError("Unauthorized", status_code=401)
            )
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                with patch.object(
                    RunMCPToolTool,
                    "_build_setup_requirements",
                    return_value=MagicMock(spec=SetupRequirementsResponse),
                ) as mock_build:
                    response = await tool._execute(
                        user_id=_USER_ID,
                        session=session,
                        server_url=_SERVER_URL,
                    )
                    mock_build.assert_called_once()

    # Should have returned what _build_setup_requirements returned
    assert response is mock_build.return_value


@pytest.mark.asyncio(loop_scope="session")
async def test_auth_error_with_existing_creds_returns_error():
    """HTTP 403 when creds ARE present → generic ErrorResponse (not setup card)."""
    from backend.util.request import HTTPClientError

    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    mock_creds = MagicMock()
    mock_creds.access_token = SecretStr("stale-token")

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=mock_creds,
        ):
            mock_client = AsyncMock()
            mock_client.initialize = AsyncMock(
                side_effect=HTTPClientError("Forbidden", status_code=403)
            )
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                )

    assert isinstance(response, ErrorResponse)
    assert "403" in response.message


@pytest.mark.asyncio(loop_scope="session")
async def test_mcp_client_error_returns_error_response():
    """MCPClientError (protocol-level) maps to a clean ErrorResponse."""
    from backend.blocks.mcp.client import MCPClientError

    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_client = AsyncMock()
            mock_client.initialize = AsyncMock(
                side_effect=MCPClientError("JSON-RPC protocol error")
            )
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                )

    assert isinstance(response, ErrorResponse)
    assert "JSON-RPC" in response.message or "protocol" in response.message.lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_unexpected_exception_returns_generic_error():
    """Unhandled exceptions inside the MCP call don't leak traceback text to the user."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_client = AsyncMock()
            # An unexpected error inside initialize (inside the try block)
            mock_client.initialize = AsyncMock(
                side_effect=ValueError("Unexpected internal error")
            )
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                )

    assert isinstance(response, ErrorResponse)
    # Must not leak the raw exception message
    assert "Unexpected internal error" not in response.message
    assert (
        "unexpected" in response.message.lower() or "error" in response.message.lower()
    )


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


def test_tool_name():
    assert RunMCPToolTool().name == "run_mcp_tool"


def test_tool_requires_auth():
    assert RunMCPToolTool().requires_auth is True


def test_tool_parameters_schema():
    params = RunMCPToolTool().parameters
    assert params["type"] == "object"
    assert "server_url" in params["properties"]
    assert "tool_name" in params["properties"]
    assert "tool_arguments" in params["properties"]
    assert params["required"] == ["server_url"]


# ---------------------------------------------------------------------------
# Query/fragment rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_query_in_url_returns_error():
    """server_url with query parameters must be rejected."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)
    response = await tool._execute(
        user_id=_USER_ID,
        session=session,
        server_url="https://mcp.example.com/mcp?key=val",
    )
    assert isinstance(response, ErrorResponse)
    assert "query" in response.message.lower() or "fragment" in response.message.lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_fragment_in_url_returns_error():
    """server_url with a fragment must be rejected."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)
    response = await tool._execute(
        user_id=_USER_ID,
        session=session,
        server_url="https://mcp.example.com/mcp#section",
    )
    assert isinstance(response, ErrorResponse)
    assert "query" in response.message.lower() or "fragment" in response.message.lower()


# ---------------------------------------------------------------------------
# Credential lookup normalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_credential_lookup_normalizes_trailing_slash():
    """Credential lookup must normalize the URL (strip trailing slash)."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)
    url_with_slash = "https://mcp.example.com/mcp/"

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_lookup:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[])
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=url_with_slash,
                )
            # Credential lookup should use the normalized URL (no trailing slash)
            mock_lookup.assert_called_once_with(_USER_ID, "https://mcp.example.com/mcp")


# ---------------------------------------------------------------------------
# _build_setup_requirements
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_build_setup_requirements_returns_setup_response():
    """_build_setup_requirements should return a SetupRequirementsResponse."""
    tool = RunMCPToolTool()
    result = tool._build_setup_requirements(
        server_url=_SERVER_URL,
        session_id="test-session",
    )
    assert isinstance(result, SetupRequirementsResponse)
    assert result.setup_info.agent_id == _SERVER_URL
    assert "authentication" in result.message.lower()
