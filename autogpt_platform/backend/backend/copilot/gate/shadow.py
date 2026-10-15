"""Shadow mode: the gate judges calls and outside reads, holds nothing, and
records what it would have done.

With ``copilot-auto-mode`` serving its "shadow" variant, the action and read
seams hand each judgement to a background task and return at once, so
the turn neither waits on a judge nor sees its answer. Each judgement becomes
one ``chat_gate_shadow_decision`` event, which carries none of what was judged.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Literal

from prometheus_client import Counter
from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.context import get_current_envelope
from backend.copilot.llm_attribution import is_turn_impersonated
from backend.copilot.model import AutopilotMode, ChatSession
from backend.copilot.tracking import SOURCE
from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

from .classifier import ACTION_RUBRIC, DecidedBy
from .content import LLM_RUBRIC, ContentVerdict
from .jev import JEV_RUBRIC

logger = logging.getLogger(__name__)
config = ChatConfig()

Kind = Literal["action", "read"]
Judge = Literal["policy", "classifier", "jev", "content"]

# Past this many judgements in flight a new one is dropped: a slow judge must
# cost samples, not the executor's memory.
MAX_IN_FLIGHT = 64
SKIPPED = Counter(
    "copilot_gate_shadow_skipped_total",
    "Shadow-gate judgements dropped because too many were in flight",
)
RUBRIC_VERSIONS: dict[Kind, str] = {
    "action": hashlib.sha256((ACTION_RUBRIC + JEV_RUBRIC).encode()).hexdigest()[:12],
    "read": hashlib.sha256(LLM_RUBRIC.encode()).hexdigest()[:12],
}
_ACTION_JUDGES: dict[DecidedBy, Judge] = {
    "llm": "classifier",
    "jev": "jev",
    "jev+llm": "jev",
}

# asyncio holds a task only weakly, so one could be collected mid-judgement.
_in_flight: set[asyncio.Task] = set()


class Finding(BaseModel):
    """What the enforcing gate would have done, in the event's terms."""

    verdict: Literal["allow", "ask", "hold", "judge_error"]
    # A policy step or a rubric category, never the judge's words.
    reason_code: str
    # None when the shadow itself failed before any stage answered.
    judge: Judge | None
    effect: str | None = None


def judge_later(
    user_id: str,
    session: ChatSession,
    *,
    kind: Kind,
    mode: AutopilotMode,
    tool_name: str,
    tool_call_id: str,
    judge: Callable[[], Awaitable[Finding | None]],
    source_kind: str | None = None,
) -> None:
    """Run ``judge`` once the caller has moved on and record its finding; a
    None finding is a call the enforcing gate would not have judged either."""
    if not sampled(session.session_id):
        return
    if len(_in_flight) >= MAX_IN_FLIGHT:
        SKIPPED.inc()
        return
    envelope = get_current_envelope()
    event = {
        "kind": kind,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "mode": mode,
        "source_kind": source_kind,
        "chat_session_id": session.session_id,
        "turn_id": envelope.tree_id if envelope else None,
        "expert_id": session.expert_id,
        "impersonated": is_turn_impersonated(),
        "sample_rate": config.gate_shadow_sample_rate,
    }
    task = asyncio.create_task(_judge_and_record(user_id, kind, event, judge))
    _in_flight.add(task)
    task.add_done_callback(_in_flight.discard)


def sampled(session_id: str) -> bool:
    """Decided per chat, so a sampled chat is recorded whole."""
    digest = hashlib.sha256(session_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < config.gate_shadow_sample_rate


def action_finding(
    *, allowed: bool, code: str, effect: str, decided_by: DecidedBy | None
) -> Finding:
    judge = _ACTION_JUDGES[decided_by] if decided_by else None
    if code == "failed":
        return Finding(
            verdict="judge_error",
            reason_code="judge_failed",
            judge=judge or "classifier",
            effect=effect,
        )
    return Finding(
        verdict="allow" if allowed else "ask",
        reason_code=code,
        judge=judge or "policy",
        effect=effect,
    )


def read_finding(verdict: ContentVerdict) -> Finding:
    judge: Judge = "jev" if verdict.decided_by == "jev" else "content"
    if not verdict.judged:
        return Finding(verdict="judge_error", reason_code="judge_failed", judge=judge)
    return Finding(
        verdict="hold" if verdict.held else "allow",
        reason_code=verdict.code,
        judge=judge,
    )


async def _judge_and_record(
    user_id: str,
    kind: Kind,
    event: dict[str, Any],
    judge: Callable[[], Awaitable[Finding | None]],
) -> None:
    started = time.monotonic()
    try:
        finding = await judge()
    except Exception:
        logger.warning(
            f"Shadow gate could not judge {event['tool_name']}", exc_info=True
        )
        finding = Finding(verdict="judge_error", reason_code="exception", judge=None)
    if finding is None:
        return
    _record(user_id, kind, event, finding, round((time.monotonic() - started) * 1000))


def _record(
    user_id: str, kind: Kind, event: dict[str, Any], finding: Finding, latency_ms: int
) -> None:
    """Send the decision alone. An opt-in store of what was judged, for tuning
    the rubrics, keys its rows by ``judgement_id``; it waits on a privacy call."""
    modelled = finding.judge not in (None, "policy")
    llm_model = config.gate_model if kind == "action" else config.gate_content_model
    posthog_client.capture(
        user_id,
        PostHogEvent.CHAT_GATE_SHADOW_DECISION,
        {
            **event,
            **finding.model_dump(),
            "llm_model": llm_model if modelled else None,
            "jev_model": config.gate_jev_model if finding.judge == "jev" else None,
            "rubric_version": RUBRIC_VERSIONS[kind] if modelled else None,
            "latency_ms": latency_ms,
            "judgement_id": str(uuid.uuid4()),
        },
        source=SOURCE,
    )
