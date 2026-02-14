"""Claude Agent SDK integration for CoPilot.

This module provides the integration layer between the Claude Agent SDK
and the existing CoPilot tool system, enabling drop-in replacement of
the current LLM orchestration with the battle-tested Claude Agent SDK.
"""

from .service import stream_chat_completion_sdk
from .tool_adapter import create_copilot_mcp_server

__all__ = [
    "stream_chat_completion_sdk",
    "create_copilot_mcp_server",
]
