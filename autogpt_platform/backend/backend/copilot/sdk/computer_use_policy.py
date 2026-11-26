"""What a connected Local PC executor offers for computer use, and what
approving it grants.

The shim relays cua-driver's tools (autogpt-local-executor
``docs/CUA_DRIVER.md``). Otto reaches them through two MCP tools:
``local_pc_computer_tools`` reads the driver's guidance and the tools'
schemas, and ``local_pc_computer`` calls one. Consent covers the machine,
the driver release and the exact tool names its HELLO listed, so a shim
that comes back offering more asks again.
"""

from pydantic import BaseModel, ConfigDict

from backend.copilot.tools.local_pc_shim import ShimConnectionGuard, ShimHello

LOCAL_PC_COMPUTER_TOOLS_TOOL = "local_pc_computer_tools"
LOCAL_PC_COMPUTER_TOOL = "local_pc_computer"
LOCAL_PC_COMPUTER_TOOL_NAMES: frozenset[str] = frozenset(
    {LOCAL_PC_COMPUTER_TOOLS_TOOL, LOCAL_PC_COMPUTER_TOOL}
)


class ComputerUseScope(BaseModel):
    """What approving computer use on one connection covers."""

    model_config = ConfigDict(frozen=True)

    machine_id: str
    driver: str
    tools: tuple[str, ...]


def computer_use_scope(hello: ShimHello) -> ComputerUseScope | None:
    """The scope a HELLO offers, or None when it offers no computer use."""
    driver = hello.computer_use_driver
    if (
        "computer_use" not in hello.capabilities
        or driver is None
        or not hello.computer_use_tools
    ):
        return None
    return ComputerUseScope(
        machine_id=hello.machine_id,
        driver=driver.label,
        tools=tuple(hello.computer_use_tools),
    )


def guard_scope(
    guard: ShimConnectionGuard, capabilities: frozenset[str]
) -> ComputerUseScope | None:
    """The same scope, read from a live connection's guard."""
    if (
        "computer_use" not in capabilities
        or not guard.computer_use_driver
        or not guard.computer_use_tools
    ):
        return None
    return ComputerUseScope(
        machine_id=guard.machine_id,
        driver=guard.computer_use_driver,
        tools=guard.computer_use_tools,
    )


def local_pc_tool_names(scope: ComputerUseScope | None) -> frozenset[str]:
    """The MCP tools to register for a scope: both, or none."""
    return LOCAL_PC_COMPUTER_TOOL_NAMES if scope is not None else frozenset()
