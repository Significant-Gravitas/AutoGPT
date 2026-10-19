"""Keep conversation output available before LangSmith closes the SDK span."""

import asyncio
from collections.abc import AsyncIterator

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    Message,
    ResultMessage,
    StreamEvent,
    TextBlock,
)
from langsmith import get_current_run_tree
from langsmith.run_trees import RunTree
from pydantic import BaseModel, ConfigDict


class ConversationOutputClient(ClaudeSDKClient):
    async def receive_messages(self) -> AsyncIterator[Message]:
        # receive_response's LangSmith span is active here, including before
        # the first message and when the underlying transport raises.
        run = get_current_run_tree()
        output = _ConversationOutput(
            run=run if run and run.name == "claude.conversation" else None
        )
        output.publish()
        try:
            async for message in super().receive_messages():
                output.record(message)
                yield message
        except asyncio.CancelledError:
            output.stop_reason = "cancelled"
            output.publish()
            raise
        except Exception as exc:
            output.stop_reason = "error"
            output.publish()
            if output.run is not None:
                output.run.end(error=type(exc).__name__)
            raise
        else:
            if not output.finished:
                output.stop_reason = "error"
                output.publish()


class _ConversationOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run: RunTree | None
    text: str = ""
    stop_reason: str = "cancelled"
    assistant_stop_reason: str | None = None
    message_id: str | None = None
    streamed_text: bool = False
    finished: bool = False

    def record(self, message: Message) -> None:
        if isinstance(message, ResultMessage):
            self.finish(message)
        elif isinstance(message, AssistantMessage) and not message.parent_tool_use_id:
            if not message.message_id or message.message_id != self.message_id:
                self.reset(message.message_id)
            if not self.streamed_text:
                self.text += "".join(
                    block.text
                    for block in message.content
                    if isinstance(block, TextBlock)
                )
            self.assistant_stop_reason = (
                message.stop_reason or self.assistant_stop_reason
            )
        elif isinstance(message, StreamEvent) and not message.parent_tool_use_id:
            self.record_stream_event(message)
        else:
            return
        self.publish()

    def reset(self, message_id: str | None) -> None:
        self.text = ""
        self.message_id = message_id
        self.streamed_text = False
        self.assistant_stop_reason = None

    def record_stream_event(self, message: StreamEvent) -> None:
        event = message.event
        if event.get("type") == "message_start":
            self.reset(event.get("message", {}).get("id"))
        elif event.get("type") == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "text":
                self.text += block.get("text", "")
                self.streamed_text = True
        elif event.get("type") == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                self.text += delta.get("text", "")
                self.streamed_text = True
        elif event.get("type") == "message_delta":
            self.assistant_stop_reason = event.get("delta", {}).get("stop_reason")

    def finish(self, message: ResultMessage) -> None:
        self.finished = True
        if message.terminal_reason in {"aborted_streaming", "aborted_tools"}:
            self.stop_reason = "cancelled"
        elif message.is_error:
            self.stop_reason = (
                message.subtype if message.subtype != "success" else "error"
            )
        else:
            if message.result is not None:
                self.text = message.result
            self.stop_reason = (
                message.stop_reason or self.assistant_stop_reason or "end_turn"
            )

    def publish(self) -> None:
        if self.run is not None:
            self.run.set(outputs={"output": self.text, "stop_reason": self.stop_reason})
