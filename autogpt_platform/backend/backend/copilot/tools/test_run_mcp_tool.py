"""Unit tests for the run_mcp_tool copilot tool."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.mcp.helpers import server_host
from backend.copilot.sdk.file_ref import FileRefExpansionError

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
        "backend.copilot.tools.run_mcp_tool.validate_url_host",
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host",
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
                tool_arguments=["this", "is", "a", "list"],  # type: ignore[arg-type]  # intentionally wrong type to test validation
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
async def test_surface_connect_card_connected_when_creds_exist_and_probe_ok():
    """surface_connect_card + valid creds + probe initialize() OK → connected.

    The probe is a one-round-trip ``MCPClient.initialize`` — no tool
    listing.  When it succeeds the cred is fresh, ``has_all_credentials``
    reports True, and the UI renders Connected/Reconnect.
    """
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    mock_creds = MagicMock()
    mock_creds.access_token = SecretStr("fresh-token")
    mock_creds.id = "fresh-cred-id"

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=mock_creds,
        ):
            mock_client = AsyncMock()
            mock_client.initialize = AsyncMock(return_value=None)
            mock_client.close = AsyncMock(return_value=None)
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ) as mock_client_cls:
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                    surface_connect_card=True,
                )
                # Probe ran exactly once — no follow-up tool listing.
                mock_client_cls.assert_called_once()
                mock_client.initialize.assert_awaited_once()
                # Session terminated via DELETE so we don't leak a row
                # server-side on every "just connect" intent.
                mock_client.close.assert_awaited_once()

    assert isinstance(response, SetupRequirementsResponse)
    assert response.setup_info.user_readiness.has_all_credentials is True
    assert response.setup_info.user_readiness.ready_to_run is True
    assert response.setup_info.user_readiness.missing_credentials == {}


@pytest.mark.asyncio(loop_scope="session")
async def test_surface_connect_card_stale_creds_invalidated_returns_not_connected():
    """surface_connect_card + stale creds + probe 401 → cred deleted, not-connected card.

    Prevents the John bug from re-surfacing through the ``surface_connect_card``
    fast-path: when the cred is revoked server-side, the probe surfaces it
    immediately and we invalidate the row + return a not-connected card so
    the user re-auths in one step instead of seeing a misleading "Connected"
    pill that 401s on the next tool call.
    """
    from backend.util.request import HTTPClientError

    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    mock_creds = MagicMock()
    mock_creds.access_token = SecretStr("stale-token")
    mock_creds.id = "stale-cred-id"

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=mock_creds,
        ):
            mock_client = AsyncMock()
            mock_client.initialize = AsyncMock(
                side_effect=HTTPClientError("Unauthorized", status_code=401)
            )
            mock_client.close = AsyncMock(return_value=None)
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                with patch(
                    "backend.copilot.tools.run_mcp_tool.invalidate_mcp_credential",
                    new_callable=AsyncMock,
                ) as mock_invalidate:
                    response = await tool._execute(
                        user_id=_USER_ID,
                        session=session,
                        server_url=_SERVER_URL,
                        surface_connect_card=True,
                    )
                    mock_invalidate.assert_awaited_once_with(_USER_ID, "stale-cred-id")
                    # close() runs in ``finally`` even when initialize raised.
                    mock_client.close.assert_awaited_once()

    assert isinstance(response, SetupRequirementsResponse)
    assert response.setup_info.user_readiness.has_all_credentials is False
    assert response.setup_info.user_readiness.ready_to_run is False


@pytest.mark.asyncio(loop_scope="session")
async def test_surface_connect_card_disconnected_when_no_creds():
    """surface_connect_card=True + no creds → SetupReq has_all_credentials=False.

    Renders as the standard "Connect <service>" card; no network call made.
    """
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient"
            ) as mock_client_cls:
                response = await tool._execute(
                    user_id=_USER_ID,
                    session=session,
                    server_url=_SERVER_URL,
                    surface_connect_card=True,
                )
                mock_client_cls.assert_not_called()

    assert isinstance(response, SetupRequirementsResponse)
    assert response.setup_info.user_readiness.has_all_credentials is False
    assert response.setup_info.user_readiness.ready_to_run is False


@pytest.mark.asyncio(loop_scope="session")
async def test_surface_connect_card_probe_5xx_reports_optimistically_connected():
    """surface_connect_card + valid creds + probe 500 → connected, cred kept.

    The probe is best-effort: transient server errors (5xx, redirects)
    must NOT delete an otherwise-valid cred.  We report
    ``has_all_credentials=True`` and let the next real tool call surface
    the actual error if it persists.
    """
    from backend.util.request import HTTPClientError

    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    mock_creds = MagicMock()
    mock_creds.access_token = SecretStr("valid-token")
    mock_creds.id = "valid-cred-id"

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=mock_creds,
        ):
            mock_client = AsyncMock()
            mock_client.initialize = AsyncMock(
                side_effect=HTTPClientError("Internal Server Error", status_code=500)
            )
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                with patch(
                    "backend.copilot.tools.run_mcp_tool.invalidate_mcp_credential",
                    new_callable=AsyncMock,
                ) as mock_invalidate:
                    response = await tool._execute(
                        user_id=_USER_ID,
                        session=session,
                        server_url=_SERVER_URL,
                        surface_connect_card=True,
                    )
                    mock_invalidate.assert_not_awaited()

    assert isinstance(response, SetupRequirementsResponse)
    assert response.setup_info.user_readiness.has_all_credentials is True
    assert response.setup_info.user_readiness.ready_to_run is True


@pytest.mark.asyncio(loop_scope="session")
async def test_surface_connect_card_probe_timeout_reports_optimistically_connected():
    """surface_connect_card + valid creds + probe TimeoutError → connected, cred kept.

    Non-HTTP failures (asyncio.TimeoutError, OSError, MCPClientError) hit
    the broad ``except Exception`` branch — same "optimistically connected"
    semantics as the 5xx path.
    """
    import asyncio

    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    mock_creds = MagicMock()
    mock_creds.access_token = SecretStr("valid-token")
    mock_creds.id = "valid-cred-id"

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=mock_creds,
        ):
            mock_client = AsyncMock()
            mock_client.initialize = AsyncMock(
                side_effect=asyncio.TimeoutError("probe took too long")
            )
            with patch(
                "backend.copilot.tools.run_mcp_tool.MCPClient",
                return_value=mock_client,
            ):
                with patch(
                    "backend.copilot.tools.run_mcp_tool.invalidate_mcp_credential",
                    new_callable=AsyncMock,
                ) as mock_invalidate:
                    response = await tool._execute(
                        user_id=_USER_ID,
                        session=session,
                        server_url=_SERVER_URL,
                        surface_connect_card=True,
                    )
                    mock_invalidate.assert_not_awaited()

    assert isinstance(response, SetupRequirementsResponse)
    assert response.setup_info.user_readiness.has_all_credentials is True
    assert response.setup_info.user_readiness.ready_to_run is True


@pytest.mark.asyncio(loop_scope="session")
async def test_auth_error_with_stale_creds_fires_setup_and_invalidates():
    """HTTP 403 when creds ARE present → still fire setup card, drop the stale row.

    Stored creds whose ``access_token_expires_at`` is in the future locally
    but which the server has revoked/expired don't get refreshed by
    ``auto_lookup_mcp_credential`` — they come back live, the request 401s,
    and the user is stuck.  The fix: on any 401/403, surface the setup card
    so the user can re-auth, and delete the stale row so the next attempt
    doesn't loop on the same dead token.
    """
    from backend.util.request import HTTPClientError

    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    mock_creds = MagicMock()
    mock_creds.access_token = SecretStr("stale-token")
    mock_creds.id = "stale-cred-id"

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
                with patch(
                    "backend.copilot.tools.run_mcp_tool.invalidate_mcp_credential",
                    new_callable=AsyncMock,
                ) as mock_invalidate:
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
                        mock_invalidate.assert_awaited_once_with(
                            _USER_ID, "stale-cred-id"
                        )

    assert response is mock_build.return_value


@pytest.mark.asyncio(loop_scope="session")
async def test_http_error_returns_clean_message_with_collapsible_detail():
    """Non-auth HTTP errors return a clean message with raw detail in the `error` field."""
    from backend.util.request import HTTPClientError

    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
    ):
        with patch(
            "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_client = AsyncMock()
            mock_client.initialize = AsyncMock(
                side_effect=HTTPClientError(
                    "<!doctype html><html><body>Not Found</body></html>",
                    status_code=404,
                )
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
    assert "404" in response.message
    # Raw HTML body must NOT leak into the user-facing message
    assert "<!doctype" not in response.message
    # Raw detail (including original body) goes in the collapsible `error` field
    assert response.error is not None
    assert "404" in response.error
    assert "<!doctype" in response.error.lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_mcp_client_error_returns_error_response():
    """MCPClientError (protocol-level) maps to a clean ErrorResponse."""
    from backend.blocks.mcp.client import MCPClientError

    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
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
    assert "sign in" in result.message.lower()


# ---------------------------------------------------------------------------
# @@agptfile: reference expansion (OPEN-3159)
# ---------------------------------------------------------------------------


def _make_tool_schema(name: str, input_schema: dict[str, Any]):
    t = MagicMock()
    t.name = name
    t.input_schema = input_schema
    return t


@pytest.mark.asyncio(loop_scope="session")
async def test_agptfile_ref_expanded_before_mcp_call():
    """@@agptfile tokens in tool_arguments expand before reaching the server."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)
    expanded = {"content": "FILE BODY"}
    schema = {"type": "object", "properties": {"content": {"type": "string"}}}
    tool_schema = _make_tool_schema("notion-update-page", schema)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
    ), patch(
        "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
        new_callable=AsyncMock,
        return_value=None,
    ), patch(
        "backend.copilot.tools.run_mcp_tool.expand_file_refs_in_args",
        new_callable=AsyncMock,
        return_value=expanded,
    ) as mock_expand:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[tool_schema])
        mock_client.call_tool = AsyncMock(
            return_value=_make_call_result([{"type": "text", "text": "ok"}])
        )
        with patch(
            "backend.copilot.tools.run_mcp_tool.MCPClient", return_value=mock_client
        ):
            await tool._execute(
                user_id=_USER_ID,
                session=session,
                server_url=_SERVER_URL,
                tool_name="notion-update-page",
                tool_arguments={"content": "@@agptfile:/home/user/report.md"},
            )

    # The tool's real schema was looked up and threaded into expansion.
    mock_expand.assert_awaited_once()
    await_args = mock_expand.await_args
    assert await_args is not None
    assert await_args.kwargs["input_schema"] == schema
    # user_id + session (positional) are forwarded — they gate file access
    # permissions and are the reason _execute_tool's signature changed.
    assert await_args.args[1] == _USER_ID
    assert await_args.args[2] is session
    # Expanded content (not the literal token) reached the server.
    assert mock_client.call_tool.call_args.args[1] == expanded


@pytest.mark.asyncio(loop_scope="session")
async def test_agptfile_expansion_failure_returns_error():
    """A failed ref resolution blocks the call and returns ErrorResponse."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
    ), patch(
        "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
        new_callable=AsyncMock,
        return_value=None,
    ), patch(
        "backend.copilot.tools.run_mcp_tool.expand_file_refs_in_args",
        new_callable=AsyncMock,
        side_effect=FileRefExpansionError("missing.md not found"),
    ):
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.call_tool = AsyncMock()
        with patch(
            "backend.copilot.tools.run_mcp_tool.MCPClient", return_value=mock_client
        ):
            response = await tool._execute(
                user_id=_USER_ID,
                session=session,
                server_url=_SERVER_URL,
                tool_name="notion-update-page",
                tool_arguments={"content": "@@agptfile:/home/user/missing.md"},
            )

    assert isinstance(response, ErrorResponse)
    assert "file reference" in response.message.lower()
    mock_client.call_tool.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_no_agptfile_ref_skips_schema_lookup():
    """Args without a ref pass through verbatim — no extra list_tools call."""
    tool = RunMCPToolTool()
    session = make_session(_USER_ID)
    raw_args = {"url": "https://example.com"}

    with patch(
        "backend.copilot.tools.run_mcp_tool.validate_url_host", new_callable=AsyncMock
    ), patch(
        "backend.copilot.tools.run_mcp_tool.auto_lookup_mcp_credential",
        new_callable=AsyncMock,
        return_value=None,
    ):
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.call_tool = AsyncMock(
            return_value=_make_call_result([{"type": "text", "text": "ok"}])
        )
        with patch(
            "backend.copilot.tools.run_mcp_tool.MCPClient", return_value=mock_client
        ):
            await tool._execute(
                user_id=_USER_ID,
                session=session,
                server_url=_SERVER_URL,
                tool_name="fetch",
                tool_arguments=raw_args,
            )

    mock_client.list_tools.assert_not_called()
    assert mock_client.call_tool.call_args.args[1] == raw_args


@pytest.mark.asyncio(loop_scope="session")
async def test_lookup_tool_schema_returns_none_on_any_failure():
    """Schema lookup degrades gracefully on any list_tools failure (not just
    HTTP/MCP errors) so expansion proceeds schema-less instead of crashing."""
    tool = RunMCPToolTool()
    mock_client = AsyncMock()
    mock_client.server_url = _SERVER_URL
    mock_client.list_tools = AsyncMock(side_effect=TimeoutError("network timeout"))

    schema = await tool._lookup_tool_schema(mock_client, "notion-update-page")

    assert schema is None
