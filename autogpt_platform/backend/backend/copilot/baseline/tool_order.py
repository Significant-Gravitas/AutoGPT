"""Which baseline tool calls must not overlap the rest of their message."""

import orjson

from backend.copilot.capabilities.dispatch import resolve_tool_dispatch
from backend.copilot.tools import TOOL_REGISTRY
from backend.util.tool_call_loop import LLMToolCall


def writes_files(tool_call: LLMToolCall) -> bool:
    """Whether *tool_call* runs a tool that ``writes_files``, named directly
    or dispatched through ``run_capability``.

    Arguments that do not parse answer False: the executor refuses that call
    without running anything.
    """
    try:
        args = orjson.loads(tool_call.arguments)
    except orjson.JSONDecodeError:
        return False
    dispatch = resolve_tool_dispatch(tool_call.name, args)
    tool = dispatch.tool if dispatch else TOOL_REGISTRY.get(tool_call.name)
    return tool is not None and tool.writes_files
