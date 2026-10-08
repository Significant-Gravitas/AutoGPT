"""The copilot tool registry as a Pydantic AI toolset.

Exposure is the baseline's, unchanged: the caller hands in the exact OpenAI
tool list the baseline would send (``get_available_tools`` with the turn's
disabled groups and tools, which leaves the deferred registry tools to
``run_capability``, then ``_filter_tools_by_permissions``), and every schema
goes to the model as-is.

Execution is the baseline's too: every call runs through
``tools.execute_tool``, so the disabled-tool refusal, the ``run_capability``
dispatch, the envelope check and the auto-mode gate (``BaseTool._gate`` ->
``gate.check_action``, which parks the call and opens its review card) all
apply. What changes is what a parked call does to the run: instead of handing
the model a refusal and carrying on, it raises ``ApprovalRequired``, so the run
ends on ``DeferredToolRequests`` and the next turn resumes it with the result.
"""

import uuid
from typing import Any, Protocol

from openai.types.chat import ChatCompletionToolParam
from pydantic import ValidationError
from pydantic_ai import ApprovalRequired, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_core import SchemaValidator, core_schema

from backend.copilot.capabilities.dispatch import resolve_tool_dispatch
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.tool_display import tool_display_context
from backend.copilot.tools import ToolGroup, execute_tool
from backend.copilot.tools.models import ApprovalRequiredResponse, ResponseType
from backend.util.tool_call_loop import ToolCallResult

# Arguments are validated by each tool; the schema is the model's contract.
_ANY_ARGS = SchemaValidator(schema=core_schema.any_schema())
# A malformed call goes back to the model as a retry this many times before
# the run gives up, where the baseline would hand back an error result.
_MAX_RETRIES = 3


class ToolCallSink(Protocol):
    """Where a tool call reports what the user and the chat rows see."""

    def emit_output(self, output: StreamToolOutputAvailable) -> None: ...

    def set_display_name(self, tool_call_id: str, name: str) -> None: ...

    def record_result(self, result: ToolCallResult) -> None: ...

    def current_session(self) -> ChatSession: ...


def tool_definitions(tools: list[ChatCompletionToolParam]) -> list[ToolDefinition]:
    """One definition per OpenAI tool, carrying its schema unchanged."""
    return [
        ToolDefinition(
            name=tool["function"]["name"],
            description=tool["function"].get("description"),
            parameters_json_schema=dict(tool["function"].get("parameters") or {}),
        )
        for tool in tools
    ]


class RegistryToolset(AbstractToolset[None]):
    """The turn's tool surface, executed through the copilot registry."""

    def __init__(
        self,
        tools: list[ChatCompletionToolParam],
        *,
        sink: ToolCallSink,
        user_id: str | None,
        disabled_groups: list[ToolGroup],
        disabled_tools: frozenset[str],
    ) -> None:
        self._definitions = tool_definitions(tools)
        self._sink = sink
        self._user_id = user_id
        self._disabled_groups = disabled_groups
        self._disabled_tools = disabled_tools
        # Cleared for the turn's last round so the model has to answer in text.
        self.enabled = True

    @property
    def id(self) -> str | None:
        return "copilot-registry"

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        if not self.enabled:
            return {}
        return {
            definition.name: ToolsetTool(
                toolset=self,
                tool_def=definition,
                max_retries=_MAX_RETRIES,
                args_validator=_ANY_ARGS,
            )
            for definition in self._definitions
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[None],
        tool: ToolsetTool[None],
    ) -> str:
        tool_call_id = ctx.tool_call_id or str(uuid.uuid4())
        called_name = called_tool_name(name, tool_args)
        result = await self._execute(name, called_name, tool_args, tool_call_id)
        self._sink.emit_output(result)
        output = output_text(result)
        self._sink.record_result(
            ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name=called_name,
                content=output,
                is_error=not result.success,
            )
        )
        review_id = held_review_id(result)
        if review_id is not None:
            raise ApprovalRequired(
                metadata=held_metadata(review_id, called_name, output)
            )
        return output

    async def _execute(
        self, name: str, called_name: str, args: dict[str, Any], tool_call_id: str
    ) -> StreamToolOutputAvailable:
        def on_display_name(display: str) -> None:
            self._sink.set_display_name(tool_call_id, display)

        try:
            with tool_display_context(on_display_name):
                return await execute_tool(
                    tool_name=name,
                    parameters=args,
                    user_id=self._user_id,
                    session=self._sink.current_session(),
                    tool_call_id=tool_call_id,
                    disabled_groups=self._disabled_groups,
                    disabled_tools=self._disabled_tools,
                )
        except Exception as e:
            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=called_name,
                output=f"Tool execution error: {e}",
                success=False,
            )


def called_tool_name(name: str, args: dict[str, Any]) -> str:
    """The tool that actually runs: a ``run_capability`` dispatch names it."""
    dispatch = resolve_tool_dispatch(name, args)
    return dispatch.name if dispatch else name


def output_text(result: StreamToolOutputAvailable) -> str:
    """The result as the model reads it (the baseline's ``str`` of it)."""
    output = result.output
    return output if isinstance(output, str) else str(output)


def held_review_id(result: StreamToolOutputAvailable) -> str | None:
    """The review id when the gate parked this call for the user, else None.

    A refusal without a review id (rejected, already used, unrecordable) is a
    final answer, not a held call, so it goes back to the model as a result.
    """
    if result.success or not isinstance(result.output, str):
        return None
    try:
        parsed = ApprovalRequiredResponse.model_validate_json(result.output)
    except ValidationError:
        return None
    if parsed.type != ResponseType.APPROVAL_REQUIRED:
        return None
    return parsed.review_id


def held_metadata(review_id: str, tool_name: str, output: str) -> dict[str, Any]:
    """What a held call carries on ``DeferredToolRequests.metadata``."""
    return {"review_id": review_id, "tool_name": tool_name, "output": output}
