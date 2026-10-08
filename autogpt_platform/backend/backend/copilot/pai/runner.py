"""The agent loop of a pai turn: ``agent.iter()`` node by node.

Between nodes the loop does what the baseline's ``tool_call_loop`` driver does
between rounds:

* stop at a round boundary when ``enter_agent_building_mode`` registered an
  engine switch (``engine_switch.is_pending``);
* drain messages the user queued mid-turn (``pending_messages``), persist them
  as user rows after the rows so far, and hand them to the next request as a
  ``<user_follow_up>`` block;
* on the last allowed round, hide the tools and add the baseline's wrap-up
  hint, so the model ends in text instead of being cut off.

A call the gate parks ends the run on ``DeferredToolRequests``; the held calls
are kept on the state so the history stored for the next turn can resume them.
"""

import logging
from dataclasses import dataclass

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRequestNode,
)
from pydantic_ai.agent import AgentRun
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, UserContent, UserPromptPart
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from backend.copilot import engine_switch
from backend.copilot.baseline.service import _LAST_ITERATION_HINT
from backend.copilot.config import ChatConfig
from backend.copilot.pending_message_helpers import (
    drain_pending_safe,
    drained_rows_entry,
    persist_pending_as_user_rows,
)
from backend.copilot.pending_messages import (
    PendingMessage,
    format_pending_as_followup,
    format_pending_as_user_message,
)
from backend.copilot.stream_checkpoint import turn_checkpoint

from .events import PaiEventMapper
from .history import HeldToolCall
from .model import PaiRoute
from .persistence import (
    add_response_usage,
    begin_tool_round,
    finish_tool_round,
    flush_rows,
)
from .state import PaiTurnState
from .toolset import RegistryToolset

logger = logging.getLogger(__name__)

PaiAgent = Agent[None, str | DeferredToolRequests]
PaiRun = AgentRun[None, str | DeferredToolRequests]


@dataclass(kw_only=True)
class RunInputs:
    """What one agent run needs besides the state it writes to.

    A dataclass, not a model: it wires live objects (the agent, the toolset,
    the mapper), which pydantic cannot build a schema for.
    """

    agent: PaiAgent
    user_prompt: str | list[UserContent] | None
    history: list[ModelMessage]
    deferred_results: DeferredToolResults | None
    model_settings: ModelSettings
    toolset: RegistryToolset
    mapper: PaiEventMapper
    route: PaiRoute
    config: ChatConfig
    session_id: str
    turn_start: int
    max_rounds: int


async def run_agent_loop(inputs: RunInputs, state: PaiTurnState) -> None:
    """Drive the run to its end, then close the state's event queue."""
    run: PaiRun | None = None
    try:
        async with inputs.agent.iter(
            inputs.user_prompt,
            message_history=inputs.history or None,
            deferred_tool_results=inputs.deferred_results,
            model_settings=inputs.model_settings,
            usage_limits=UsageLimits(request_limit=inputs.max_rounds + 2),
            toolsets=[inputs.toolset],
        ) as run:
            await _drive(run, inputs, state)
    except UsageLimitExceeded:
        logger.warning("[PAI] Request limit reached; ending the turn")
        state.budget_reached = True
    finally:
        if run is not None:
            _collect(run, state)
        state.close()


async def _drive(run: PaiRun, inputs: RunInputs, state: PaiTurnState) -> None:
    async for node in run:
        if Agent.is_model_request_node(node):
            if not await _before_request(node.request, inputs, state):
                state.messages = [*run.all_messages(), node.request]
                return
            await _stream_request(node, run, inputs, state)
        elif Agent.is_call_tools_node(node):
            async with node.stream(run.ctx) as events:
                async for event in events:
                    inputs.mapper.on_tool_event(event)
            for result in inputs.mapper.unanswered_results:
                state.record_result(result)
            inputs.mapper.unanswered_results.clear()
            finish_tool_round(state)


async def _stream_request(
    node: ModelRequestNode[None, str | DeferredToolRequests],
    run: PaiRun,
    inputs: RunInputs,
    state: PaiTurnState,
) -> None:
    mapper = inputs.mapper
    mapper.start_step()
    try:
        async with node.stream(run.ctx) as stream:
            async for event in stream:
                mapper.on_model_event(event)
    finally:
        text = mapper.finish_step()
        state.assistant_text += text
    response = run.ctx.state.message_history[-1]
    if response.kind != "response":
        return
    add_response_usage(state, response, inputs.route, inputs.config)
    begin_tool_round(state, response, text)


async def _before_request(
    request: ModelRequest, inputs: RunInputs, state: PaiTurnState
) -> bool:
    """Round-boundary work; False ends the run before this request."""
    if state.rounds > 0:
        if engine_switch.is_pending(inputs.session_id):
            logger.info(f"[PAI] Engine switch pending for {inputs.session_id[:12]}")
            state.engine_switched = True
            return False
        state.text_len_before_final_round = len(state.assistant_text)
        await inject_follow_ups(request, inputs, state)
    state.rounds += 1
    if state.rounds >= inputs.max_rounds:
        inputs.toolset.enabled = False
        request.parts = [*request.parts, UserPromptPart(content=_LAST_ITERATION_HINT)]
        state.budget_reached = True
    return True


async def inject_follow_ups(
    request: ModelRequest, inputs: RunInputs, state: PaiTurnState
) -> None:
    """Fold queued user messages into the next request, after their rows."""
    pending = await drain_pending_safe(inputs.session_id, "[PAI]")
    if not pending:
        return
    session = await flush_rows(state)

    def row_content(message: PendingMessage) -> str:
        return format_pending_as_user_message(message)["content"]

    persisted = await persist_pending_as_user_rows(
        session, None, pending, log_prefix="[PAI]", content_of=row_content
    )
    if not persisted:
        # Re-queued for the next turn by the helper; the model must not see
        # them twice.
        return
    state.emit(drained_rows_entry(pending, row_content))
    request.parts = [
        *request.parts,
        UserPromptPart(content=format_pending_as_followup(pending)),
    ]
    checkpoint = turn_checkpoint(session.messages, inputs.turn_start)
    if checkpoint is not None:
        state.emit(checkpoint)


def _collect(run: PaiRun, state: PaiTurnState) -> None:
    """The run's history, and the held calls it ended on."""
    if not state.engine_switched:
        state.messages = run.all_messages()
    result = run.result
    if result is None or not isinstance(result.output, DeferredToolRequests):
        return
    state.held = [
        HeldToolCall(
            tool_call_id=call.tool_call_id,
            tool_name=str(meta.get("tool_name") or call.tool_name),
            review_id=str(meta.get("review_id") or ""),
            output=str(meta.get("output") or ""),
        )
        for call in result.output.approvals
        for meta in [result.output.metadata.get(call.tool_call_id, {})]
    ]
