"""``stream_chat_completion_pai``: a copilot turn on a Pydantic AI agent.

Same contract as ``stream_chat_completion_baseline`` (same arguments, same
``StreamBaseResponse`` sequence, same rows, same usage records), so the
processor wraps it in the same heartbeat and ``stream_registry.stream_and_publish``
and finishes it the same way: the processor, not this generator, calls
``mark_session_completed``.

Flow: set up (:mod:`.turn_setup`) -> ``StreamStart`` -> answered held calls
(:mod:`.resume`) -> agent loop in its own task (:mod:`.runner`) while this
generator yields its events -> fallbacks or the error envelope -> rows,
usage, history upload -> checkpoint, ``StreamUsage``, ``StreamFinish``.
"""

import asyncio
import logging
import shutil
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from langfuse import propagate_attributes
from pydantic_ai import Agent, DeferredToolRequests

from backend.copilot.baseline.service import (
    _background_tasks,
    _build_budget_exhausted_fallback_events,
    _build_natural_finish_empty_fallback_events,
    _engine_switch_finish_events,
    _enqueue_graphiti_turn,
    _humanize_baseline_error,
    _is_platform_out_of_credits,
    _pause_uncounted_box,
)
from backend.copilot.config import CopilotLLMModel
from backend.copilot.markers import append_error_marker
from backend.copilot.model import ChatSession, upsert_chat_session
from backend.copilot.provider_failure import classify as classify_provider_failure
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamCheckpoint,
    StreamError,
    StreamFinish,
    StreamProviderFailure,
    StreamStart,
    StreamUsage,
)
from backend.copilot.service import config
from backend.copilot.stream_checkpoint import turn_checkpoint
from backend.copilot.tools.e2b_sandbox import count_expert_turn, pause_sandbox_direct
from backend.copilot.transcript import next_uncovered_sequence
from backend.util.llm.provider_billing import (
    PROVIDER_UNAVAILABLE_CODE,
    PROVIDER_UNAVAILABLE_MESSAGE,
    report_provider_out_of_credits,
)

from .compaction import compaction_processor
from .events import PaiEventMapper
from .history_store import upload_history
from .model import build_model
from .persistence import finalize_rows, record_usage
from .resume import resume_held_calls
from .runner import PaiAgent, RunInputs, run_agent_loop
from .state import PaiTurnState
from .toolset import RegistryToolset
from .turn_setup import PreparedTurn, TurnRequest, late_results_prompt, prepare_turn

if TYPE_CHECKING:
    from backend.copilot.permissions import CopilotPermissions
    from backend.copilot.tree import TurnEnvelope

logger = logging.getLogger(__name__)


async def stream_chat_completion_pai(
    session_id: str,
    message: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    session: ChatSession | None = None,
    file_ids: list[str] | None = None,
    permissions: "CopilotPermissions | None" = None,
    envelope: "TurnEnvelope | None" = None,
    context: dict[str, str] | None = None,
    model: CopilotLLMModel | None = None,
    request_arrival_at: float = 0.0,
    organization_id: str | None = None,
    team_id: str | None = None,
    message_metadata: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> AsyncGenerator[StreamBaseResponse, None]:
    request = TurnRequest(
        session_id=session_id,
        message=message,
        is_user_message=is_user_message,
        user_id=user_id,
        session=session,
        file_ids=file_ids,
        permissions=permissions,
        envelope=envelope,
        context=context,
        model=model,
        request_arrival_at=request_arrival_at,
        organization_id=organization_id,
        team_id=team_id,
        message_metadata=message_metadata,
    )
    turn = await prepare_turn(request, config)
    try:
        yield StreamStart(messageId=turn.message_id, sessionId=session_id)
    except BaseException:
        if turn.sandbox is not None:
            _pause_uncounted_box(turn.sandbox, session_id, turn.session.expert_id)
        raise
    if turn.sandbox is not None:
        await count_expert_turn(session_id, turn.session.expert_id)

    state = PaiTurnState(
        turn.session, model=turn.route.model, routing_source=turn.route.source
    )
    for entry in turn.opening_entries:
        state.emit(entry)
    inputs = await _run_inputs(turn, request, state)
    trace = _open_trace(user_id, session_id)
    loop_task = asyncio.create_task(run_agent_loop(inputs, state))
    stream_error = False
    checkpoint: StreamCheckpoint | None = None
    try:
        while (event := await state.queue.get()) is not None:
            yield event
        await loop_task
        for event in _fallback_events(state):
            yield event
    except Exception as e:
        stream_error = True
        for event in await _error_events(e, state, user_id, session_id, turn):
            yield event
    finally:
        await _stop(loop_task)
        _close_trace(trace)
        checkpoint = await _finish_turn(turn, state, request, stream_error)

    if checkpoint is not None:
        yield checkpoint
    usage = state.usage
    if usage.reported:
        yield StreamUsage(
            prompt_tokens=usage.uncached_prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.uncached_prompt_tokens + usage.completion_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
        )
    for event in _engine_switch_finish_events(session_id):
        yield event
    yield StreamFinish()


def build_agent(turn: PreparedTurn) -> tuple[PaiAgent, Any]:
    """The turn's agent: static instructions first, ``<turn_context>`` last."""
    model, settings = build_model(turn.route, config, turn.static_instructions)
    turn_context = turn.turn_context

    def dynamic_instructions() -> str:
        return turn_context

    agent: PaiAgent = Agent(
        model,
        instructions=[turn.static_instructions, dynamic_instructions],
        output_type=[str, DeferredToolRequests],
        history_processors=[
            compaction_processor(
                turn.route.model, always_check=turn.route.provider == "local"
            )
        ],
    )
    return agent, settings


async def _run_inputs(
    turn: PreparedTurn, request: TurnRequest, state: PaiTurnState
) -> RunInputs:
    resumption = await resume_held_calls(
        request.user_id, state, turn.history, turn.held
    )
    agent, settings = build_agent(turn)
    mapper = PaiEventMapper(state.emit, state.session_messages)
    mapper.resumed_call_ids = resumption.resumed_call_ids
    return RunInputs(
        agent=agent,
        user_prompt=late_results_prompt(turn.user_prompt, resumption.late_results),
        history=turn.history,
        deferred_results=resumption.deferred,
        model_settings=settings,
        toolset=RegistryToolset(
            turn.tools,
            sink=state,
            user_id=request.user_id,
            disabled_groups=turn.disabled_groups,
            disabled_tools=turn.disabled_tools,
        ),
        mapper=mapper,
        route=turn.route,
        config=config,
        session_id=request.session_id,
        turn_start=turn.turn_start,
        max_rounds=config.agent_max_turns,
    )


def _fallback_events(state: PaiTurnState) -> list[StreamBaseResponse]:
    """The baseline's notices for a turn that ended with nothing to read.

    A turn that ended on a held call shows its approval card instead, and an
    engine switch narrates itself at the finish.
    """
    if state.held or state.engine_switched:
        return []
    terminal = state.assistant_text[state.text_len_before_final_round :]
    build = (
        _build_budget_exhausted_fallback_events
        if state.budget_reached
        else _build_natural_finish_empty_fallback_events
    )
    events, text = build(terminal)
    state.assistant_text += text
    return events


async def _error_events(
    error: Exception,
    state: PaiTurnState,
    user_id: str | None,
    session_id: str,
    turn: PreparedTurn,
) -> list[StreamBaseResponse]:
    """The baseline's failure envelope: tail events, error marker, error."""
    session = state.session
    auth_provider = session.metadata.llm_auth_provider
    out_of_credits = _is_platform_out_of_credits(error, auth_provider)
    message = (
        PROVIDER_UNAVAILABLE_MESSAGE
        if out_of_credits
        else _humanize_baseline_error(error)
    )
    logger.error(f"[PAI] Streaming error: {message}", exc_info=True)
    if out_of_credits:
        report_provider_out_of_credits(
            provider=config.effective_transport,
            model=turn.route.model,
            surface="copilot_pai",
            error=error,
            session_id=session_id,
            user_id=user_id,
        )
    events: list[StreamBaseResponse] = []
    while not state.queue.empty():
        queued = state.queue.get_nowait()
        if queued is not None:
            events.append(queued)
    failure = (
        None
        if out_of_credits
        else classify_provider_failure(
            error,
            auth_provider=auth_provider,
            credential_id=session.metadata.llm_credential_id,
            message=message,
        )
    )
    # Before the error is yielded: the consumer closes this generator on it.
    if append_error_marker(
        session,
        message,
        retryable=failure.retryable if failure is not None else True,
        failure=failure.as_part() if failure is not None else None,
    ):
        try:
            await upsert_chat_session(session)
        except Exception as marker_err:
            logger.error(f"[PAI] Failed to persist the error marker: {marker_err}")
    if failure is not None:
        events.append(StreamProviderFailure(failure=failure.as_part()))
        events.append(StreamError(errorText=message, code=failure.kind.value))
    elif out_of_credits:
        events.append(StreamError(errorText=message, code=PROVIDER_UNAVAILABLE_CODE))
    else:
        events.append(StreamError(errorText=message, code="baseline_error"))
    return events


async def _stop(task: asyncio.Task[None]) -> None:
    if task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def _finish_turn(
    turn: PreparedTurn, state: PaiTurnState, request: TurnRequest, stream_error: bool
) -> StreamCheckpoint | None:
    """Rows, usage, memory and history, as the baseline's ``finally`` does."""
    session = state.session
    session.clear_inflight_tool_calls()
    if turn.sandbox is not None:
        _background(
            pause_sandbox_direct(
                turn.sandbox, request.session_id, expert_id=session.expert_id
            )
        )
    await record_usage(
        state,
        turn.route,
        config,
        user_id=request.user_id,
        failed_silently=stream_error and not state.assistant_text,
    )
    final_text = finalize_rows(state)
    checkpoint: StreamCheckpoint | None = None
    try:
        state.session = await upsert_chat_session(state.session)
        checkpoint = turn_checkpoint(state.session.messages, turn.turn_start)
    except Exception as persist_err:
        logger.error(f"[PAI] Failed to persist session: {persist_err}")
    if (
        turn.graphiti_enabled
        and request.user_id
        and turn.message
        and request.is_user_message
    ):
        _background(
            _enqueue_graphiti_turn(
                request.user_id,
                state.session,
                request.session_id,
                turn.message,
                final_text,
            )
        )
    if request.user_id and turn.upload_safe and state.messages:
        await upload_history(
            request.user_id,
            request.session_id,
            state.messages,
            state.held,
            next_uncovered_sequence(state.session.messages),
        )
    if turn.working_dir is not None:
        shutil.rmtree(turn.working_dir, ignore_errors=True)
    return checkpoint


def _background(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _open_trace(user_id: str | None, session_id: str) -> Any:
    try:
        trace = propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            trace_name="copilot-pai",
            tags=["pai", "tool_surface:registry"],
        )
        trace.__enter__()
        return trace
    except Exception:
        logger.warning("[PAI] Langfuse trace context setup failed")
        return None


def _close_trace(trace: Any) -> None:
    if trace is None:
        return
    try:
        trace.__exit__(None, None, None)
    except Exception:
        logger.warning("[PAI] Langfuse trace context teardown failed")
