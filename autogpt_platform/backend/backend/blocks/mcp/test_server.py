"""
Minimal MCP server for integration testing.

Implements the MCP Streamable HTTP transport (JSON-RPC 2.0 over HTTP POST)
with a few sample tools. Runs on localhost with a random available port.

The server can speak either protocol era so the client's dual-era behaviour
is exercised against real HTTP:

* ``era="legacy"`` — ``initialize`` handshake (``2025-03-26``), optional
  ``Mcp-Session-Id`` sessions, JSON-RPC ``-32601`` for unknown methods.
* ``era="modern"`` — stateless ``2026-07-28``: validates ``_meta`` and the
  mirrored ``Mcp-*`` headers, implements ``server/discover``, returns
  ``resultType`` / ``ttlMs`` / ``cacheScope``, and rejects ``initialize``.
* ``era="dual"`` — serves whichever era the client opens with.
"""

import json
import logging
import uuid
from typing import Literal

from aiohttp import web

from backend.blocks.mcp.protocol import (
    ERROR_HEADER_MISMATCH,
    ERROR_INVALID_PARAMS,
    ERROR_METHOD_NOT_FOUND,
    ERROR_UNSUPPORTED_PROTOCOL_VERSION,
    HEADER_METHOD,
    HEADER_NAME,
    HEADER_PARAM_PREFIX,
    HEADER_PROTOCOL_VERSION,
    HEADER_SESSION_ID,
    LEGACY_PROTOCOL_VERSION,
    META_CLIENT_CAPABILITIES,
    META_PROTOCOL_VERSION,
    META_SERVER_INFO,
    MODERN_PROTOCOL_VERSION,
    decode_header_value,
)

logger = logging.getLogger(__name__)

SERVER_INFO = {"name": "test-mcp-server", "version": "1.0.0"}

# Sample tools this test server exposes
TEST_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "add_numbers",
        "description": "Add two numbers together",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "echo",
        "description": "Echo back the input message",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to echo"},
            },
            "required": ["message"],
        },
    },
]

# Extra tools only a modern (2026-07-28) server exposes.
MODERN_EXTRA_TOOLS = [
    {
        # ``region`` must be mirrored into ``Mcp-Param-Region``.
        "name": "execute_sql",
        "description": "Run SQL in a region",
        "inputSchema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "x-mcp-header": "Region"},
                "query": {"type": "string"},
            },
            "required": ["region", "query"],
        },
    },
    {
        # First call answers ``input_required`` with only ``requestState``.
        "name": "needs_confirmation",
        "description": "Completes on the second round-trip",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        # Asks for elicitation, which this client does not support.
        "name": "needs_elicitation",
        "description": "Requests interactive input",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        # ``x-mcp-header`` on a ``number`` parameter is invalid: clients must
        # drop this tool from ``tools/list``.
        "name": "broken_header_tool",
        "description": "Has an invalid x-mcp-header annotation",
        "inputSchema": {
            "type": "object",
            "properties": {"ratio": {"type": "number", "x-mcp-header": "Ratio"}},
        },
    },
]
MODERN_TEST_TOOLS = TEST_TOOLS + MODERN_EXTRA_TOOLS

_CONFIRMATION_STATE = "confirm-me"


def _tool_result(params: dict) -> dict:
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    if tool_name == "get_weather":
        city = arguments.get("city", "Unknown")
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {"city": city, "temperature": 22, "condition": "sunny"}
                    ),
                }
            ],
        }

    elif tool_name == "add_numbers":
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        return {
            "content": [{"type": "text", "text": json.dumps({"result": a + b})}],
        }

    elif tool_name == "echo":
        message = arguments.get("message", "")
        return {
            "content": [{"type": "text", "text": message}],
        }

    elif tool_name == "execute_sql":
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"region": arguments.get("region"), "rows": 1}),
                }
            ],
        }

    elif tool_name == "needs_confirmation":
        if params.get("requestState") == _CONFIRMATION_STATE:
            return {"content": [{"type": "text", "text": "confirmed"}]}
        return {"resultType": "input_required", "requestState": _CONFIRMATION_STATE}

    elif tool_name == "needs_elicitation":
        return {
            "resultType": "input_required",
            "inputRequests": {
                "login": {
                    "method": "elicitation/create",
                    "params": {"mode": "form", "message": "Who are you?"},
                }
            },
        }

    else:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            "isError": True,
        }


# ─────────────────────────── legacy era ───────────────────────────


def _handle_initialize(params: dict) -> dict:
    return {
        "protocolVersion": LEGACY_PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": dict(SERVER_INFO),
    }


def _handle_tools_list(params: dict) -> dict:
    return {"tools": TEST_TOOLS}


def _handle_tools_call(params: dict) -> dict:
    return _tool_result(params)


HANDLERS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
}


def _jsonrpc_error(request_id, code: int, message: str, data=None, *, status=200):
    error: dict = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return web.json_response(
        {"jsonrpc": "2.0", "error": error, "id": request_id}, status=status
    )


def _legacy_response(app: web.Application, request: web.Request, body: dict):
    method = body.get("method", "")
    params = body.get("params", {})
    request_id = body.get("id")

    if app["sessions_enabled"] and method != "initialize":
        session_id = request.headers.get(HEADER_SESSION_ID)
        if not session_id or session_id not in app["sessions"]:
            # What the reference SDKs do for a request outside a session:
            # a plain-text 400 with no JSON-RPC body.
            return web.Response(status=400, text="Bad Request: Missing session ID")

    handler = HANDLERS.get(method)
    if not handler:
        return _jsonrpc_error(
            request_id, ERROR_METHOD_NOT_FOUND, f"Method not found: {method}"
        )

    result = handler(params)
    response = web.json_response({"jsonrpc": "2.0", "result": result, "id": request_id})
    if method == "initialize" and app["sessions_enabled"]:
        session_id = uuid.uuid4().hex
        app["sessions"].add(session_id)
        response.headers[HEADER_SESSION_ID] = session_id
    return response


# ─────────────────────────── modern era ───────────────────────────


def _modern_result(result: dict) -> dict:
    result = dict(result)
    result.setdefault("resultType", "complete")
    result.setdefault("_meta", {})[META_SERVER_INFO] = dict(SERVER_INFO)
    return result


def _modern_response(app: web.Application, request: web.Request, body: dict):
    method = body.get("method", "")
    params = body.get("params") or {}
    request_id = body.get("id")
    meta = params.get("_meta") or {}
    version = meta.get(META_PROTOCOL_VERSION)

    if not version or META_CLIENT_CAPABILITIES not in meta:
        return _jsonrpc_error(
            request_id,
            ERROR_INVALID_PARAMS,
            "Missing required _meta fields",
            status=400,
        )
    if version not in app["supported_versions"]:
        return _jsonrpc_error(
            request_id,
            ERROR_UNSUPPORTED_PROTOCOL_VERSION,
            "Unsupported protocol version",
            {"supported": list(app["supported_versions"]), "requested": version},
            status=400,
        )

    def mismatch(detail: str):
        return _jsonrpc_error(
            request_id,
            ERROR_HEADER_MISMATCH,
            f"Header mismatch: {detail}",
            status=400,
        )

    if request.headers.get(HEADER_PROTOCOL_VERSION) != version:
        return mismatch(f"{HEADER_PROTOCOL_VERSION} missing or wrong")
    if request.headers.get(HEADER_METHOD) != method:
        return mismatch(f"{HEADER_METHOD} missing or wrong")
    if method == "tools/call":
        name_header = request.headers.get(HEADER_NAME)
        if name_header is None or decode_header_value(name_header) != params.get(
            "name"
        ):
            return mismatch(f"{HEADER_NAME} missing or wrong")
        if params.get("name") == "execute_sql":
            region = (params.get("arguments") or {}).get("region")
            header = request.headers.get(f"{HEADER_PARAM_PREFIX}Region")
            if region is not None and (
                header is None or decode_header_value(header) != str(region)
            ):
                return mismatch(f"{HEADER_PARAM_PREFIX}Region missing or wrong")

    if method == "server/discover":
        result = {
            "supportedVersions": list(app["supported_versions"]),
            "capabilities": {"tools": {"listChanged": False}},
            "instructions": "Test server",
            "ttlMs": 3_600_000,
            "cacheScope": "public",
        }
    elif method == "tools/list":
        result = {"tools": MODERN_TEST_TOOLS, "ttlMs": 300_000, "cacheScope": "public"}
    elif method == "tools/call":
        result = _tool_result(params)
    else:
        return _jsonrpc_error(
            request_id,
            ERROR_METHOD_NOT_FOUND,
            f"Method not found: {method} "
            f"(supported protocol versions: {', '.join(app['supported_versions'])})",
            status=404,
        )

    return web.json_response(
        {"jsonrpc": "2.0", "result": _modern_result(result), "id": request_id}
    )


# ─────────────────────────── app wiring ───────────────────────────


def _as_sse(response: web.Response) -> web.Response:
    """Re-wrap a JSON reply as a single-event SSE stream."""
    payload = response.text
    if response.content_type != "application/json" or payload is None:
        return response
    return web.Response(
        status=response.status,
        headers={
            k: v for k, v in response.headers.items() if k.lower() != "content-type"
        },
        text=f"event: message\ndata: {payload}\n\n",
        content_type="text/event-stream",
    )


async def handle_mcp_request(request: web.Request) -> web.Response:
    """Handle incoming MCP JSON-RPC 2.0 requests."""
    app = request.app
    # Check auth if configured
    expected_token = app.get("auth_token")
    if expected_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {expected_token}":
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32001, "message": "Unauthorized"},
                    "id": None,
                },
                status=401,
            )

    body = await request.json()
    app["requests"].append(
        {
            "method": body.get("method"),
            "params": body.get("params") or {},
            "headers": dict(request.headers),
        }
    )

    # Handle notifications (no id field) — just acknowledge
    if "id" not in body:
        return web.Response(status=202)

    era = app["era"]
    params = body.get("params") or {}
    is_modern_request = META_PROTOCOL_VERSION in (params.get("_meta") or {})
    if era == "modern" or (era == "dual" and is_modern_request):
        response = _modern_response(app, request, body)
    else:
        response = _legacy_response(app, request, body)

    if app["sse"]:
        response = _as_sse(response)
    return response


async def handle_mcp_delete(request: web.Request) -> web.Response:
    """Legacy session termination; modern servers answer 405."""
    app = request.app
    if app["era"] == "modern":
        return web.Response(status=405)
    session_id = request.headers.get(HEADER_SESSION_ID)
    app["deleted_sessions"].append(session_id)
    app["sessions"].discard(session_id)
    return web.Response(status=200)


def create_test_mcp_app(
    auth_token: str | None = None,
    era: Literal["legacy", "modern", "dual"] = "legacy",
    *,
    sse: bool = False,
    sessions: bool = False,
    supported_versions: tuple[str, ...] = (MODERN_PROTOCOL_VERSION,),
) -> web.Application:
    """Create an aiohttp app that acts as an MCP server.

    Args:
        auth_token: Require this Bearer token on every request.
        era: ``"legacy"``, ``"modern"`` or ``"dual"``.
        sse: Return every reply as ``text/event-stream``.
        sessions: (legacy) mint ``Mcp-Session-Id`` on initialize and require
            it afterwards, answering session-less requests with a plain-text
            ``400`` like the reference SDKs.
        supported_versions: (modern) protocol versions advertised/accepted.
    """
    app = web.Application()
    app.router.add_post("/mcp", handle_mcp_request)
    app.router.add_delete("/mcp", handle_mcp_delete)
    app["era"] = era
    app["sse"] = sse
    app["sessions_enabled"] = sessions
    app["supported_versions"] = tuple(supported_versions)
    app["sessions"] = set()
    app["deleted_sessions"] = []
    app["requests"] = []
    if auth_token:
        app["auth_token"] = auth_token
    return app
