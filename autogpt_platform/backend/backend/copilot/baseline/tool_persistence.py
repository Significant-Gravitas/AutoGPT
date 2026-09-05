"""Retain a baseline tool round while its concurrent executions are pending."""

from pydantic import BaseModel, Field

from backend.copilot.model import ChatMessage
from backend.util.tool_call_loop import ToolCallResult


class BaselineToolPersistence(BaseModel):
    pending_message: ChatMessage | None = None
    display_names: dict[str, str] = Field(default_factory=dict)
    results: dict[str, ToolCallResult] = Field(default_factory=dict)

    def begin(self, message: ChatMessage, messages: list[ChatMessage]) -> None:
        if self.pending_message is not None:
            return
        self.pending_message = message
        for call in message.tool_calls or []:
            name = self.display_names.get(call["id"])
            if name:
                call["display_name"] = name
        messages.append(message)

    def set_display_name(self, tool_call_id: str, name: str) -> None:
        self.display_names[tool_call_id] = name
        message = self.pending_message
        if message is None:
            return
        for call in message.tool_calls or []:
            if call["id"] == tool_call_id:
                call["display_name"] = name
                message.mark_tool_calls_pending_save()
                return

    def record_result(self, result: ToolCallResult) -> ToolCallResult:
        self.results[result.tool_call_id] = result
        return result

    def finish(self, messages: list[ChatMessage]) -> None:
        """Flush actual results once, leaving interrupted calls visibly pending.

        Orphan cleanup excludes unfinished calls from subsequent provider
        context; append-only history still retains their resolved names.
        """
        if self.pending_message is None:
            return
        messages.extend(
            ChatMessage(role="tool", content=result.content, tool_call_id=call["id"])
            for call in self.pending_message.tool_calls or []
            if (result := self.results.get(call["id"])) is not None
        )
        self.pending_message = None
        self.results.clear()
        self.display_names.clear()
