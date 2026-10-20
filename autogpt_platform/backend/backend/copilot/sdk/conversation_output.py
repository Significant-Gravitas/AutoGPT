"""Keep conversation output available before LangSmith closes the SDK span."""

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager

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

logger = logging.getLogger(__name__)


class ConversationOutputClient(ClaudeSDKClient):
    async def receive_messages(self) -> AsyncIterator[Message]:
        # receive_response's LangSmith span is active here, including before
        # the first message and when the underlying transport raises.
        output = _current_conversation_output()
        try:
            async for message in super().receive_messages():
                if output is not None:
                    with _ignore_trace_errors():
                        output.record(message)
                yield message
        except (asyncio.CancelledError, GeneratorExit):
            if output is not None and not output.finished:
                output.end("cancelled")
            raise
        except Exception as exc:
            if output is not None:
                output.end("error", exc)
            raise
        else:
            if output is not None and not output.finished:
                output.end("error")


def _current_conversation_output() -> "_ConversationOutput | None":
    with _ignore_trace_errors():
        run = get_current_run_tree()
        if run is not None and run.name == "claude.conversation":
            output = _ConversationOutput(run=run)
            output.publish()
            return output
    return None


@contextmanager
def _ignore_trace_errors() -> Iterator[None]:
    try:
        yield
    except Exception:
        logger.warning("Could not record conversation trace output", exc_info=True)


class _ConversationOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run: RunTree
    text: str = ""
    # A receiver closed between messages has no terminal ResultMessage.
    stop_reason: str = "cancelled"
    assistant_stop_reason: str | None = None
    message_id: str | None = None
    partial_text: str = ""
    finished: bool = False

    def record(self, message: Message) -> None:
        if isinstance(message, ResultMessage):
            self.finish(message)
        elif isinstance(message, AssistantMessage) and not message.parent_tool_use_id:
            if not message.message_id or message.message_id != self.message_id:
                self.reset(message.message_id)
            for block in message.content:
                if isinstance(block, TextBlock):
                    self.record_summary(block.text)
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
        self.partial_text = ""
        self.assistant_stop_reason = None

    def record_summary(self, text: str) -> None:
        if not text:
            return
        if text.startswith(self.partial_text):
            self.text += text[len(self.partial_text) :]
            self.partial_text = ""
        elif self.partial_text.startswith(text):
            self.partial_text = self.partial_text[len(text) :]
        else:
            self.text = self.text[: -len(self.partial_text)] + text
            self.partial_text = ""

    def record_stream_event(self, message: StreamEvent) -> None:
        event = message.event
        if event.get("type") == "message_start":
            self.reset(event.get("message", {}).get("id"))
        elif event.get("type") == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "text":
                text = block.get("text", "")
                self.text += text
                self.partial_text += text
        elif event.get("type") == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                self.text += text
                self.partial_text += text
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

    def end(self, stop_reason: str, error: Exception | None = None) -> None:
        with _ignore_trace_errors():
            self.stop_reason = stop_reason
            self.publish()
            if error is not None:
                self.run.end(error=type(error).__name__)

    def publish(self) -> None:
        self.run.set(outputs={"output": self.text, "stop_reason": self.stop_reason})
