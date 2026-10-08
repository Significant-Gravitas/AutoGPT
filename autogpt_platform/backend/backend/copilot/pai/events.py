"""Pydantic AI run events -> the copilot stream wire format.

One mapper per turn, driven by the runner with the events of each node:

* model request node: ``StreamStartStep``, then text deltas as
  ``StreamText*`` (through the baseline's ``ThinkingStripper``) and thinking
  deltas as ``StreamReasoning*`` (through the baseline's
  ``BaselineReasoningEmitter``, which also writes the ``reasoning`` chat rows
  and coalesces the wire deltas), closed by ``StreamFinishStep``;
* call-tools node: ``FunctionToolCallEvent`` -> ``StreamToolInputStart`` +
  ``StreamToolInputAvailable`` named after the tool that runs (a
  ``run_capability`` dispatch names the platform tool, as the baseline does).
  Registry tools publish their own ``StreamToolOutputAvailable`` from the
  toolset; a call Pydantic AI answers itself (bad arguments, unknown tool) is
  closed here so its card never hangs.

The event sequence matches the baseline's for the same model output, so the
frontend renders a pai turn exactly like a baseline one.
"""

import uuid
from collections.abc import Callable
from typing import Any

from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponsePart,
    ModelResponsePartDelta,
)

from backend.copilot.baseline.reasoning import BaselineReasoningEmitter
from backend.copilot.capabilities.dispatch import resolve_tool_dispatch
from backend.copilot.model import ChatMessage
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamFinishStep,
    StreamStartStep,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)
from backend.copilot.thinking_stripper import ThinkingStripper
from backend.util.tool_call_loop import ToolCallResult

Emit = Callable[[StreamBaseResponse], None]


class PaiEventMapper:
    """Owns the open text/reasoning blocks of the current step."""

    def __init__(
        self,
        emit: Emit,
        session_messages: list[ChatMessage] | None = None,
        *,
        reasoning_emitter: BaselineReasoningEmitter | None = None,
    ) -> None:
        self._emit = emit
        self._reasoning = reasoning_emitter or BaselineReasoningEmitter(
            session_messages
        )
        self._stripper = ThinkingStripper()
        self._text_id = str(uuid.uuid4())
        self._text_open = False
        self._round_text = ""
        # Calls Pydantic AI answered without running a registry tool.
        self.unanswered_results: list[ToolCallResult] = []
        # Held calls this turn resumes: their card and result already exist.
        self.resumed_call_ids: set[str] = set()

    # -- model request node -------------------------------------------------

    def start_step(self) -> None:
        self._emit(StreamStartStep())
        self._stripper = ThinkingStripper()
        self._round_text = ""

    def on_model_event(self, event: AgentStreamEvent) -> None:
        if event.event_kind == "part_start":
            self._on_part(event.part)
        elif event.event_kind == "part_delta":
            self._on_delta(event.delta)

    def finish_step(self) -> str:
        """Close every open block, end the step, and return its visible text."""
        self._close_reasoning()
        tail = self._stripper.flush()
        if tail:
            self._text(tail)
        if self._text_open:
            self._emit(StreamTextEnd(id=self._text_id))
            self._text_open = False
            self._text_id = str(uuid.uuid4())
        self._emit(StreamFinishStep())
        return self._round_text

    def _on_part(self, part: ModelResponsePart) -> None:
        if part.part_kind == "text":
            self._on_text(part.content)
        elif part.part_kind == "thinking":
            self._on_thinking(part.content)
        elif part.part_kind == "tool-call":
            self._close_reasoning()

    def _on_delta(self, delta: ModelResponsePartDelta) -> None:
        if delta.part_delta_kind == "text":
            self._on_text(delta.content_delta)
        elif delta.part_delta_kind == "thinking":
            self._on_thinking(delta.content_delta or "")
        elif delta.part_delta_kind == "tool_call":
            self._close_reasoning()

    def _on_thinking(self, text: str) -> None:
        if text:
            for event in self._reasoning.on_delta(
                ChoiceDelta.model_validate({"reasoning": text})
            ):
                self._emit(event)

    def _on_text(self, text: str) -> None:
        if not text:
            return
        # Text and reasoning never interleave on the wire.
        self._close_reasoning()
        visible = self._stripper.process(text)
        if visible:
            self._text(visible)

    def _text(self, text: str) -> None:
        if not self._text_open:
            self._emit(StreamTextStart(id=self._text_id))
            self._text_open = True
        self._round_text += text
        self._emit(StreamTextDelta(id=self._text_id, delta=text))

    def _close_reasoning(self) -> None:
        for event in self._reasoning.close():
            self._emit(event)

    # -- call-tools node ----------------------------------------------------

    def on_tool_event(self, event: AgentStreamEvent) -> None:
        if event.event_kind == "function_tool_call":
            self._on_tool_call(event)
        elif event.event_kind == "function_tool_result":
            self._on_tool_result(event)

    def _on_tool_call(self, event: FunctionToolCallEvent) -> None:
        part = event.part
        if part.tool_call_id in self.resumed_call_ids:
            return
        args = safe_args(part.args_as_dict)
        dispatch = resolve_tool_dispatch(part.tool_name, args)
        name, input_args = (
            (dispatch.name, dispatch.args) if dispatch else (part.tool_name, args)
        )
        self._emit(StreamToolInputStart(toolCallId=part.tool_call_id, toolName=name))
        self._emit(
            StreamToolInputAvailable(
                toolCallId=part.tool_call_id, toolName=name, input=input_args
            )
        )

    def _on_tool_result(self, event: FunctionToolResultEvent) -> None:
        result = event.result
        if (
            result.part_kind != "retry-prompt"
            or result.tool_call_id in self.resumed_call_ids
        ):
            return
        text = result.model_response()
        self._emit(
            StreamToolOutputAvailable(
                toolCallId=result.tool_call_id,
                toolName=result.tool_name,
                output=text,
                success=False,
            )
        )
        self.unanswered_results.append(
            ToolCallResult(
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name or "unknown",
                content=text,
                is_error=True,
            )
        )


def safe_args(args_as_dict: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """A call's arguments, or ``{}`` when the model sent malformed JSON."""
    try:
        return args_as_dict()
    except ValueError:
        return {}
