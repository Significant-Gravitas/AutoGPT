"""
Minimal MCP server for integration testing.

Implements the MCP Streamable HTTP transport (JSON-RPC 2.0 over HTTP POST)
with a few sample tools. Runs on localhost with a random available port.
"""

import json
import logging

from aiohttp import web

logger = logging.getLogger(__name__)

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


def _handle_initialize(params: dict) -> dict:
    return {
        "protocolVersion": "2025-03-26",
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "test-mcp-server", "version": "1.0.0"},
    }


def _handle_tools_list(params: dict) -> dict:
    return {"tools": TEST_TOOLS}


def _handle_tools_call(params: dict) -> dict:
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

    else:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            "isError": True,
        }


HANDLERS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
}


async def handle_mcp_request(request: web.Request) -> web.Response:
    """Handle incoming MCP JSON-RPC 2.0 requests."""
    # Check auth if configured
    expected_token = request.app.get("auth_token")
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

    # Handle notifications (no id field) — just acknowledge
    if "id" not in body:
        return web.Response(status=202)

    method = body.get("method", "")
    params = body.get("params", {})
    request_id = body.get("id")

    handler = HANDLERS.get(method)
    if not handler:
        return web.json_response(
            {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
                "id": request_id,
            }
        )

    result = handler(params)
    return web.json_response({"jsonrpc": "2.0", "result": result, "id": request_id})


def create_test_mcp_app(auth_token: str | None = None) -> web.Application:
    """Create an aiohttp app that acts as an MCP server."""
    app = web.Application()
    app.router.add_post("/mcp", handle_mcp_request)
    if auth_token:
        app["auth_token"] = auth_token
    return app
