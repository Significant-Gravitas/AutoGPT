"""Claude Agent SDK integration for CoPilot.

This module provides the integration layer between the Claude Agent SDK
and the existing CoPilot tool system, enabling drop-in replacement of
the current LLM orchestration with the battle-tested Claude Agent SDK.

Submodule imports are deferred (PEP 562) to avoid circular imports:
tools → sdk → service → prompting → tools.
"""

from typing import Any

__all__ = [
    "stream_chat_completion_sdk",
    "create_copilot_mcp_server",
]


def __getattr__(name: str) -> Any:
    if name == "stream_chat_completion_sdk":
        from .service import stream_chat_completion_sdk

        globals()["stream_chat_completion_sdk"] = stream_chat_completion_sdk
        return stream_chat_completion_sdk
    if name == "create_copilot_mcp_server":
        from .tool_adapter import create_copilot_mcp_server

        globals()["create_copilot_mcp_server"] = create_copilot_mcp_server
        return create_copilot_mcp_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
