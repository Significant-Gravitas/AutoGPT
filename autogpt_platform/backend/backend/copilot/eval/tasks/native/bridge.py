"""Stdio MCP server that ``claude -p`` launches during a task eval run.

It lists Otto's tool schemas (OTTO_EVAL_TOOLS, a JSON file) and forwards
every call to the eval process over a local socket (OTTO_EVAL_PORT), which
runs it against the simulated world and applies the turn rules. Kept free of
backend imports so it starts fast.
"""

import asyncio
import json
import os

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("otto")


async def ask_eval(message: dict) -> dict:
    reader, writer = await asyncio.open_connection(
        "127.0.0.1", int(os.environ["OTTO_EVAL_PORT"])
    )
    writer.write(json.dumps(message).encode() + b"\n")
    await writer.drain()
    line = await reader.readline()
    writer.close()
    return json.loads(line)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    with open(os.environ["OTTO_EVAL_TOOLS"]) as f:
        specs = json.load(f)
    return [
        types.Tool(
            name=s["name"],
            description=s.get("description", ""),
            inputSchema=s.get("parameters") or {"type": "object", "properties": {}},
        )
        for s in specs
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    reply = await ask_eval({"kind": "call", "name": name, "arguments": arguments})
    return [types.TextContent(type="text", text=reply["result"])]


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
