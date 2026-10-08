"""Mutable state of one pai turn, shared by the runner, the toolset and the
event consumer.

Events go through one single-producer/single-consumer queue, exactly like the
baseline's ``_BaselineStreamState.pending_events``: the agent loop runs in its
own task and the outer generator yields from the queue, so text, reasoning
and tool output reach the wire while the model is still streaming. ``None``
closes the queue.
"""

import asyncio

from pydantic_ai.messages import ModelMessage

from backend.copilot.baseline.tool_persistence import BaselineToolPersistence
from backend.copilot.config import CopilotLlmAuthProvider
from backend.copilot.model import ChatMessage, ChatSession, RoutingSource
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamToolDisplayAvailable,
    StreamToolOutputAvailable,
    ToolDisplayData,
)
from backend.util.tool_call_loop import ToolCallResult

from .history import HeldToolCall


class TurnUsage:
    """Token buckets and provider cost summed over the turn's requests.

    ``prompt_tokens`` includes the cache buckets (as providers report it);
    :attr:`uncached_prompt_tokens` is the disjoint figure that gets billed.
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.cost_usd: float | None = None

    @property
    def uncached_prompt_tokens(self) -> int:
        return max(
            0,
            self.prompt_tokens - self.cache_read_tokens - self.cache_creation_tokens,
        )

    @property
    def reported(self) -> bool:
        return self.prompt_tokens > 0 or self.completion_tokens > 0


class PaiTurnState:
    """Everything the turn accumulates; also the toolset's ``ToolCallSink``."""

    def __init__(
        self,
        session: ChatSession,
        *,
        model: str,
        routing_source: RoutingSource,
    ) -> None:
        self.session = session
        self.model = model
        self.routing_source: RoutingSource = routing_source
        self.llm_auth_provider: CopilotLlmAuthProvider | None = (
            session.metadata.llm_auth_provider
        )
        self.llm_credential_id: str | None = session.metadata.llm_credential_id
        self.queue: asyncio.Queue[StreamBaseResponse | None] = asyncio.Queue()
        # Mirror of every queued event, for tests; production never reads it.
        self.emitted: list[StreamBaseResponse] = []
        # Rows this turn adds, flushed onto ``session.messages`` in order.
        # Mutate in place only: the reasoning emitter holds this list.
        self.session_messages: list[ChatMessage] = []
        self.tool_persistence = BaselineToolPersistence()
        self.usage = TurnUsage()
        self.assistant_text = ""
        # How much of ``assistant_text`` a mid-turn flush already persisted.
        self.flushed_text_len = 0
        # ``assistant_text`` length before the last round, for the empty-finish
        # and budget fallbacks (they look at the terminal round only).
        self.text_len_before_final_round = 0
        self.rounds = 0
        self.budget_reached = False
        self.engine_switched = False
        self.held: list[HeldToolCall] = []
        self.messages: list[ModelMessage] = []

    # -- event queue ----------------------------------------------------------

    def emit(self, event: StreamBaseResponse) -> None:
        self.queue.put_nowait(event)
        self.emitted.append(event)

    def close(self) -> None:
        self.queue.put_nowait(None)

    # -- ToolCallSink ---------------------------------------------------------

    def emit_output(self, output: StreamToolOutputAvailable) -> None:
        self.emit(output)

    def set_display_name(self, tool_call_id: str, name: str) -> None:
        self.tool_persistence.set_display_name(tool_call_id, name)
        self.emit(
            StreamToolDisplayAvailable(
                id=tool_call_id,
                data=ToolDisplayData(toolCallId=tool_call_id, displayName=name),
            )
        )

    def record_result(self, result: ToolCallResult) -> None:
        self.tool_persistence.record_result(result)

    def current_session(self) -> ChatSession:
        return self.session

    # -- rows -----------------------------------------------------------------

    def assistant_row(
        self, content: str, tool_calls: list[dict] | None = None
    ) -> ChatMessage:
        """An assistant row stamped with this turn's model and connection."""
        return ChatMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            model=self.model,
            routing_source=self.routing_source,
            llm_auth_provider=self.llm_auth_provider,
            llm_credential_id=self.llm_credential_id,
        )
