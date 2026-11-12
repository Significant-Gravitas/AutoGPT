"""Claude Agent SDK integration for CoPilot.

This module provides the integration layer between the Claude Agent SDK
and the existing CoPilot tool system, enabling drop-in replacement of
the current LLM orchestration with the battle-tested Claude Agent SDK.

Submodule imports are deferred via PEP 562 ``__getattr__`` to break a
circular import cycle::

    sdk/__init__ → tool_adapter → copilot.tools (TOOL_REGISTRY)
    copilot.tools → run_block → sdk.file_ref  (no cycle here, but…)
    sdk/__init__ → service → copilot.prompting → copilot.tools  (cycle!)

``tool_adapter`` uses ``TOOL_REGISTRY`` at **module level** to build the
static ``COPILOT_TOOL_NAMES`` list, so the import cannot be deferred to
function scope without a larger refactor (moving tool-name registration
to a separate lightweight module).  The lazy-import pattern here is the
least invasive way to break the cycle while keeping module-level constants
intact.
"""

from typing import TYPE_CHECKING, Any

# Static imports for type checkers so they can resolve __all__ entries
# without executing the lazy-import machinery at runtime.
if TYPE_CHECKING:
    from .service import stream_chat_completion_sdk as stream_chat_completion_sdk
    from .tool_adapter import create_copilot_mcp_server as create_copilot_mcp_server

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
