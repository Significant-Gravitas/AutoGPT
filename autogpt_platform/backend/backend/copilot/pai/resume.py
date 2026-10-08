"""Answered held calls at the start of a pai turn.

The gate's own flow does the work: ``gate.held.resolve_answered`` runs every
call whose card was answered (or refuses it) and returns each result as a
``<held_call_result>`` user row, which the turn persists exactly as the
baseline does so the chain row renders the same. What this engine adds is
where the result goes: a call the last run *ended on* (still pending in the
stored history) resumes through ``DeferredToolResults``, so the model reads
the result as the tool's own return; any other late result (from an older
turn, or another engine) rides in the user prompt, as on the baseline.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from pydantic_ai import DeferredToolResults
from pydantic_ai.messages import ModelMessage

from backend.copilot.gate.held import _RESULT_KEY, resolve_answered
from backend.copilot.pending_message_helpers import (
    drained_rows_entry,
    persist_pending_as_user_rows,
)
from backend.copilot.pending_messages import PendingMessage

from .history import HeldToolCall, pending_tool_calls, resume_results
from .state import PaiTurnState


class Resumption(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    deferred: DeferredToolResults | None
    resumed_call_ids: set[str]
    late_results: list[PendingMessage]


async def resume_held_calls(
    user_id: str | None,
    state: PaiTurnState,
    history: Sequence[ModelMessage],
    held: Sequence[HeldToolCall],
) -> Resumption:
    """Run answered cards, persist their rows, and route their results."""
    results = await resolve_answered(user_id, state.session)
    if results and await persist_pending_as_user_rows(
        state.session, None, results, log_prefix="[PAI]"
    ):
        state.emit(drained_rows_entry(results))
    else:
        results = []
    return route_results(history, held, results)


def route_results(
    history: Sequence[ModelMessage],
    held: Sequence[HeldToolCall],
    results: Sequence[PendingMessage],
) -> Resumption:
    pending = pending_tool_calls(history)
    pending_ids = {call.tool_call_id for call in pending}
    delivered: dict[str, str] = {}
    late: list[PendingMessage] = []
    for result in results:
        call_id = held_call_id(result)
        if call_id in pending_ids:
            delivered[call_id] = result.content
        else:
            late.append(result)
    deferred = (
        DeferredToolResults(calls=dict(resume_results(pending, held, delivered)))
        if pending
        else None
    )
    return Resumption(
        deferred=deferred, resumed_call_ids=pending_ids, late_results=late
    )


def held_call_id(result: PendingMessage) -> str:
    """The tool call a ``<held_call_result>`` row answers ("" when none)."""
    meta = (result.metadata or {}).get(_RESULT_KEY) or {}
    return str(meta.get("tool_call_id") or "")
