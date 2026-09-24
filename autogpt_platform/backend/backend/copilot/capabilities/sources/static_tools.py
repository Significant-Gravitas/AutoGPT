"""Platform tools (``TOOL_REGISTRY``) as capability entries.

The tool mapping is passed in rather than imported: ``backend.copilot.tools``
imports the registry tools, which import this package, so the tools package
hands its registry over at call time instead of at import time.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from backend.copilot.capabilities.models import (
    CapabilityEntry,
    Implementation,
    clip_purpose,
    normalize_text,
)
from backend.copilot.capabilities.text import tokenize

if TYPE_CHECKING:
    from backend.copilot.tools.base import BaseTool

# Tools that stay in the model's tool list.  Everything else in the registry
# is reached through find/describe/run_capability.  The registry tools are
# listed so the eager set is complete once they exist.
EAGER_CORE: frozenset[str] = frozenset(
    {
        "find_capability",
        "describe_capability",
        "run_capability",
        "resume_capability",
        "bash_exec",
        "run_agent",
        "find_library_agent",
        "ask_question",
        "web_search",
        "web_fetch",
        "read_workspace_file",
        "write_workspace_file",
        "run_sub_session",
        "delegate_to_expert",
        "handoff_to_expert",
        "TodoWrite",
        # The expert prompt tells the model to use ``start_desktop`` by name,
        # and a deferred tool named directly is refused. Eager, the screen
        # goes on in one call instead of a find/run round trip every turn.
        "start_desktop",
        # The refusal the building gate prints tells the model to call
        # ``enter_agent_building_mode``, which a deferred tool named directly
        # refuses — the same bind ``start_desktop`` was in.
        "enter_agent_building_mode",
        # The memory supplement orders a search before answering anything a
        # past conversation could hold, and a deferred tool named directly is
        # refused, so that order can only be followed eager.
        "memory_search",
        # ``kickoff_turn_disabled_tools`` narrows a hire's first turn to this
        # one tool. Deferred, that gate leaves the turn with no tools at all:
        # the card it exists to open is unreachable, and so is the
        # ``run_capability`` that would reach it.
        "expert_onboarding",
    }
)

# Legacy discovery/execution tools the registry replaces.  They get no entry:
# a search for "run a block" should land on the block, not on run_block.
RETIRED_TOOLS: frozenset[str] = frozenset(
    {"find_block", "run_block", "run_mcp_tool", "get_mcp_guide"}
)


def tool_entries(
    tools: "Mapping[str, BaseTool]", groups: Mapping[str, str]
) -> list[CapabilityEntry]:
    """One entry per available tool, skipping the retired discovery tools."""
    return [
        _tool_entry(name, tool, groups.get(name))
        for name, tool in tools.items()
        if name not in RETIRED_TOOLS and tool.is_available
    ]


def _tool_entry(name: str, tool: "BaseTool", group: str | None) -> CapabilityEntry:
    properties = (tool.parameters or {}).get("properties") or {}
    tags = sorted(set(tokenize(name)))
    if group:
        tags.append(group)
    return CapabilityEntry(
        id=f"tool:{name}",
        kind="tool",
        klass="service",
        name=name,
        purpose=clip_purpose(tool.description),
        description=normalize_text(tool.description),
        tags=tags,
        context="direct",
        implementations=[
            Implementation(kind="tool", ref=name, name=name, context="direct")
        ],
        argument_names=list(properties),
        schema_ref=f"tool:{name}",
        eager=name in EAGER_CORE,
    )
