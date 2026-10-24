"""Shadow mode for the read screen: every outside read is judged, nothing is
held, and the verdict is recorded.

With ``copilot-read-screen`` serving "shadow", the read seams hand each
judgement to a background task and return at once, so the turn neither waits
on the judge nor sees its answer. Each judgement becomes one
``chat_read_screen_shadow_decision`` event, which carries none of what was read.
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
from backend.copilot.llm_attribution import is_turn_impersonated, turn_trace_id
from backend.copilot.model import ChatSession
from backend.copilot.tracking import SOURCE
from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

from .content import LLM_RUBRIC, ContentVerdict

logger = logging.getLogger(__name__)
config = ChatConfig()

Judge = Literal["jev", "content"]

# Past this many judgements in flight a new one is dropped: a slow judge must
# cost samples, not the executor's memory.
MAX_IN_FLIGHT = 64
SKIPPED = Counter(
    "copilot_read_screen_shadow_skipped_total",
    "Read-screen shadow judgements dropped because too many were in flight",
)
RUBRIC_VERSION = hashlib.sha256(LLM_RUBRIC.encode()).hexdigest()[:12]

# asyncio holds a task only weakly, so one could be collected mid-judgement.
_in_flight: set[asyncio.Task] = set()


class Finding(BaseModel):
    """What the enforcing read screen would have done, in the event's terms."""

    verdict: Literal["allow", "hold", "judge_error"]
    # A rubric category or a failure, never the judge's words.
    reason_code: str
    # None when the shadow itself failed before the judge answered.
    judge: Judge | None


def judge_later(
    user_id: str,
    session: ChatSession,
    *,
    tool_name: str,
    tool_call_id: str,
    source_kind: str,
    judge: Callable[[], Awaitable[Finding | None]],
) -> None:
    """Run ``judge`` once the caller has moved on and record its finding; a
    None finding is a read the enforcing screen would not have judged either."""
    if not sampled(session.session_id):
        return
    if len(_in_flight) >= MAX_IN_FLIGHT:
        SKIPPED.inc()
        return
    event = {
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "source_kind": source_kind,
        "chat_session_id": session.session_id,
        # The reply's trace, which a spawned turn shares; the tool call's
        # $ai_span, matched on tool_call_id, names the turn that read.
        "trace_id": turn_trace_id(),
        "expert_id": session.expert_id,
        "impersonated": is_turn_impersonated(),
        "sample_rate": config.read_screen_shadow_sample_rate,
    }
    task = asyncio.create_task(_judge_and_record(user_id, event, judge))
    _in_flight.add(task)
    task.add_done_callback(_in_flight.discard)


def sampled(session_id: str) -> bool:
    """Decided per chat, so a sampled chat is recorded whole."""
    digest = hashlib.sha256(session_id.encode()).digest()
    return (
        int.from_bytes(digest[:8], "big") / 2**64
        < config.read_screen_shadow_sample_rate
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
    event: dict[str, Any],
    judge: Callable[[], Awaitable[Finding | None]],
) -> None:
    started = time.monotonic()
    try:
        finding = await judge()
    except Exception:
        logger.warning(
            f"Read-screen shadow could not judge {event['tool_name']}", exc_info=True
        )
        finding = Finding(verdict="judge_error", reason_code="exception", judge=None)
    if finding is None:
        return
    _record(user_id, event, finding, round((time.monotonic() - started) * 1000))


def _record(
    user_id: str, event: dict[str, Any], finding: Finding, latency_ms: int
) -> None:
    """Send the decision alone. An opt-in store of what was read, for tuning
    the rubric, keys its rows by ``judgement_id``; it waits on a privacy call."""
    judged = finding.judge is not None
    posthog_client.capture(
        user_id,
        PostHogEvent.CHAT_READ_SCREEN_SHADOW_DECISION,
        {
            **event,
            **finding.model_dump(),
            "llm_model": config.gate_content_model if judged else None,
            "jev_model": config.gate_jev_model if finding.judge == "jev" else None,
            "rubric_version": RUBRIC_VERSION if judged else None,
            "latency_ms": latency_ms,
            "judgement_id": str(uuid.uuid4()),
        },
        source=SOURCE,
    )
