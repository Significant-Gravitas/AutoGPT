"""Chat rows and usage for a pai turn, written the way the baseline writes them.

Rows: an assistant row per tool round (``BaselineToolPersistence``, display
names included), its tool rows once the round finishes, ``reasoning`` rows
from the reasoning emitter, and the final text as one assistant row — the
same rows, in the same order, that ``stream_chat_completion_baseline``
produces, so ``convertChatSessionToUiMessages`` renders both engines alike.

Usage: summed per model response and recorded once through
``token_tracking.persist_and_record_usage``, with OpenRouter's reported cost
or the Anthropic rate card, as the baseline does.
"""

import logging

from pydantic_ai.messages import ModelResponse

from backend.copilot.anthropic_rate_card import compute_anthropic_cost_usd
from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatSession
from backend.copilot.pending_message_helpers import persist_session_safe
from backend.copilot.token_tracking import persist_and_record_usage
from backend.util.prompt import estimate_token_count_str

from .history import messages_to_chat_rows
from .model import PaiRoute, response_cost_usd
from .state import PaiTurnState

logger = logging.getLogger(__name__)


def begin_tool_round(state: PaiTurnState, response: ModelResponse, text: str) -> None:
    """Open the assistant row for a round that called tools."""
    calls = [
        {
            "id": call.tool_call_id,
            "type": "function",
            "function": {"name": call.tool_name, "arguments": call.args_as_json_str()},
        }
        for call in response.tool_calls
    ]
    if calls:
        state.tool_persistence.begin(
            state.assistant_row(text, calls), state.session_messages
        )


def finish_tool_round(state: PaiTurnState) -> None:
    state.tool_persistence.finish(state.session_messages)


def add_response_usage(
    state: PaiTurnState, response: ModelResponse, route: PaiRoute, config: ChatConfig
) -> None:
    usage = response.usage
    totals = state.usage
    totals.prompt_tokens += usage.input_tokens
    totals.completion_tokens += usage.output_tokens
    totals.cache_read_tokens += usage.cache_read_tokens
    totals.cache_creation_tokens += usage.cache_write_tokens
    cost = response_cost_usd(response)
    if cost is None and route.provider == "anthropic":
        cost = compute_anthropic_cost_usd(
            model=route.model,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_write_tokens,
            cache_ttl=config.baseline_prompt_cache_ttl,
        )
    if cost is not None:
        totals.cost_usd = (totals.cost_usd or 0.0) + cost


async def flush_rows(state: PaiTurnState) -> ChatSession:
    """Persist the rows so far before a mid-turn user row lands after them.

    Text from text-only rounds is not in any tool-round row, so it is written
    as its own assistant row first (the baseline's in-order flush).
    """
    recorded = "".join(
        m.content or "" for m in state.session_messages if m.role == "assistant"
    )
    unflushed = state.assistant_text[state.flushed_text_len :]
    text_only = (
        unflushed[len(recorded) :] if unflushed.startswith(recorded) else unflushed
    )
    session = state.session
    if text_only.strip():
        session.messages.append(state.assistant_row(text_only))
    session.messages.extend(state.session_messages)
    state.session_messages.clear()
    state.flushed_text_len = len(state.assistant_text)
    state.session = await persist_session_safe(session, "[PAI]")
    return state.session


def finalize_rows(state: PaiTurnState) -> str:
    """Move the buffered rows and the final text onto the session; returns
    the final assistant text (for memory ingestion)."""
    state.tool_persistence.finish(state.session_messages)
    final_text = state.assistant_text[state.flushed_text_len :]
    recorded = "".join(
        m.content or "" for m in state.session_messages if m.role == "assistant"
    )
    if state.session_messages and final_text.startswith(recorded):
        final_text = final_text[len(recorded) :]
    state.session.messages.extend(state.session_messages)
    state.session_messages.clear()
    if final_text.strip():
        state.session.messages.append(state.assistant_row(final_text))
    return final_text


async def record_usage(
    state: PaiTurnState,
    route: PaiRoute,
    config: ChatConfig,
    *,
    user_id: str | None,
    failed_silently: bool,
) -> None:
    """Record the turn's usage and cost (estimated when nothing was reported,
    except for a failed turn that produced nothing)."""
    usage = state.usage
    if not usage.reported and not failed_silently:
        # The provider reported nothing (dropped usage chunk): estimate, as
        # the baseline does, so the turn is never free.
        prompt_text = "".join(
            row.content or "" for row in messages_to_chat_rows(state.messages)
        )
        usage.prompt_tokens = max(
            1, estimate_token_count_str(prompt_text, model=route.model)
        )
        usage.completion_tokens = estimate_token_count_str(
            state.assistant_text, model=route.model
        )
    await persist_and_record_usage(
        session=state.session,
        user_id=user_id,
        prompt_tokens=usage.uncached_prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
        log_prefix="[PAI]",
        cost_usd=usage.cost_usd,
        model=route.model,
        provider=(
            "open_router"
            if route.provider == "openrouter"
            else config.transport.cost_log_provider
        ),
    )
