"""Computer use through cua-driver.

The shim runs `cua-driver mcp` and relays a reviewed subset of its tools
(see `catalog`); `driver` owns the process and `mcp_client` speaks MCP to
it. The clipboard writeback registry and the private-content check add
the shim's clipboard rules on top. See docs/CUA_DRIVER.md.
"""

from .catalog import (
    APPROVAL_ARGUMENTS,
    REVIEWED_DRIVER_VERSION,
    ApprovalRequired,
    ArgumentRefused,
    check_arguments,
    offered_tools,
)
from .clipboard import WRITEBACK_WINDOW_SECONDS, ClipboardWritebackRegistry
from .driver import (
    CuaDriverBridge,
    DriverSnapshot,
    DriverUnavailable,
    find_driver_binary,
)
from .errors import ClipboardConcealedError, ComputerUseError

__all__ = [
    "APPROVAL_ARGUMENTS",
    "REVIEWED_DRIVER_VERSION",
    "WRITEBACK_WINDOW_SECONDS",
    "ApprovalRequired",
    "ArgumentRefused",
    "ClipboardConcealedError",
    "ClipboardWritebackRegistry",
    "ComputerUseError",
    "CuaDriverBridge",
    "DriverSnapshot",
    "DriverUnavailable",
    "check_arguments",
    "find_driver_binary",
    "offered_tools",
]
