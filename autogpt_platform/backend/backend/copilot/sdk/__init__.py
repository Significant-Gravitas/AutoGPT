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

# Dispatch table for PEP 562 lazy imports.  Each entry is a (module, attr)
# pair so new exports can be added without touching __getattr__ itself.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "stream_chat_completion_sdk": (".service", "stream_chat_completion_sdk"),
    "create_copilot_mcp_server": (".tool_adapter", "create_copilot_mcp_server"),
}


def __getattr__(name: str) -> Any:
    entry = _LAZY_IMPORTS.get(name)
    if entry is not None:
        module_path, attr = entry
        import importlib

        module = importlib.import_module(module_path, package=__name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
