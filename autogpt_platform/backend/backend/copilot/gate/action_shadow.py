"""Shadow mode for the action gate: every tool call is judged as the enforcing
gate would judge it, nothing is held, and the verdict is recorded.

With ``copilot-auto-mode`` serving "shadow", ``check_action`` lets each call
through as with the gate off and hands its decision to a background task, so
the turn neither waits on the supervisor nor sees its answer. Each decision
becomes one ``chat_gate_shadow_decision`` event, which carries codes only:
none of the call's arguments, the files it runs or the user's messages.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, Awaitable, Callable, Literal, get_args

from prometheus_client import Counter
from pydantic import BaseModel

from backend.copilot.autopilot_mode import AutopilotMode
from backend.copilot.config import ChatConfig
from backend.copilot.llm_attribution import is_turn_impersonated, turn_trace_id
from backend.copilot.model import ChatSession
from backend.copilot.tracking import SOURCE
from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

from . import jev
from .classifier import ACTION_RUBRIC, JUDGED_ACTION_RUBRIC, Judgement
from .policy import Effect
from .review import ReasonKind
from .subject import Subject

logger = logging.getLogger(__name__)
config = ChatConfig()

DecidedBy = Literal["policy", "rule", "approval", "jev", "llm"]
MODES: tuple[AutopilotMode, ...] = get_args(AutopilotMode)

# Past this many judgements in flight a new one is dropped: a slow supervisor
# must cost samples, not the executor's memory.
MAX_IN_FLIGHT = 64
SKIPPED = Counter(
    "copilot_gate_shadow_skipped_total",
    "Action-gate shadow judgements dropped because too many were in flight",
)
RUBRIC_VERSION = hashlib.sha256(
    "\0".join(
        [
            ACTION_RUBRIC,
            JUDGED_ACTION_RUBRIC,
            jev.JEV_RUBRIC,
            jev.JUDGED_JEV_RUBRIC,
            repr(jev.QUESTIONS),
            repr(jev.JUDGED_QUESTIONS),
        ]
    ).encode()
).hexdigest()[:12]

# asyncio holds a task only weakly, so one could be collected mid-judgement.
_in_flight: set[asyncio.Task] = set()


class ModeVerdict(BaseModel):
    """What one mode's gate would have done with the call."""

    verdict: Literal["run", "ask"]
    decided_by: DecidedBy
    # A policy step or the supervisor's code, never its words.
    reason_code: str


class Finding(BaseModel):
    effect: Effect
    # block, workflow, mcp or tool; None when the call has no subject of its own.
    subject_kind: str | None
    verdicts: dict[AutopilotMode, ModeVerdict]
    judgement: Judgement | None = None


def judge_later(
    user_id: str,
    session: ChatSession,
    *,
    mode: AutopilotMode,
    tool_name: str,
    tool_call_id: str,
    judge: Callable[[], Awaitable[Finding]],
) -> None:
    """Run ``judge`` once the caller has moved on, and record what ``mode``,
    the chat's own, would have done."""
    if not sampled(session.session_id):
        return
    if len(_in_flight) >= MAX_IN_FLIGHT:
        SKIPPED.inc()
        return
    event = {
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "mode": mode,
        "chat_session_id": session.session_id,
        # The reply's trace; the tool call's $ai_span, matched on tool_call_id,
        # names the turn that made the call.
        "trace_id": turn_trace_id(),
        "expert_id": session.expert_id,
        "impersonated": is_turn_impersonated(),
        "sample_rate": config.gate_shadow_sample_rate,
    }
    task = asyncio.create_task(_judge_and_record(user_id, mode, event, judge))
    _in_flight.add(task)
    task.add_done_callback(_in_flight.discard)


def sampled(session_id: str) -> bool:
    """Decided per chat, so a sampled chat is recorded whole."""
    digest = hashlib.sha256(session_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < config.gate_shadow_sample_rate


def unjudged(
    effect: Effect, subject: Subject | None, *, approved: bool = False
) -> Finding:
    """A call every mode runs: it is not gated, or its card was approved."""
    verdict = ModeVerdict(
        verdict="run",
        decided_by="approval" if approved else "policy",
        reason_code="card_approved" if approved else "not_gated",
    )
    return Finding(
        effect=effect,
        subject_kind=_subject_kind(subject),
        verdicts={mode: verdict for mode in MODES},
    )


def mode_verdict(
    *,
    run: bool,
    by: Literal["rule", "mode", "supervisor"],
    reason_kind: ReasonKind,
    judgement: Judgement | None,
) -> ModeVerdict:
    """One mode's ruling in the event's terms; ``judgement`` is the
    supervisor's, read only when it decided."""
    verdict: Literal["run", "ask"] = "run" if run else "ask"
    if by == "supervisor" and judgement is not None:
        return ModeVerdict(
            verdict=verdict,
            decided_by=_judged_by(judgement),
            reason_code=judgement.code,
        )
    if by == "rule":
        return ModeVerdict(
            verdict=verdict, decided_by="rule", reason_code=f"rule_{verdict}"
        )
    code = "subject_ask" if reason_kind == "subject" else f"mode_{verdict}"
    return ModeVerdict(verdict=verdict, decided_by="policy", reason_code=code)


def finding(
    effect: Effect,
    subject: Subject | None,
    verdicts: dict[AutopilotMode, ModeVerdict],
    judgement: Judgement | None,
) -> Finding:
    return Finding(
        effect=effect,
        subject_kind=_subject_kind(subject),
        verdicts=verdicts,
        judgement=judgement,
    )


async def _judge_and_record(
    user_id: str,
    mode: AutopilotMode,
    event: dict[str, Any],
    judge: Callable[[], Awaitable[Finding]],
) -> None:
    started = time.monotonic()
    try:
        found = await judge()
    except Exception:
        logger.warning(
            f"Action-gate shadow could not judge {event['tool_name']}", exc_info=True
        )
        found = None
    latency_ms = round((time.monotonic() - started) * 1000)
    try:
        _record(user_id, mode, event, found, latency_ms)
    except Exception:
        logger.warning("Action-gate shadow could not record a decision", exc_info=True)


def _record(
    user_id: str,
    mode: AutopilotMode,
    event: dict[str, Any],
    found: Finding | None,
    latency_ms: int,
) -> None:
    """``found`` None: the shadow itself failed, where the enforcing gate
    would have refused the call."""
    judgement = found.judgement if found else None
    jev_calls = judgement.jev_calls if judgement else 0
    llm_calls = judgement.llm_calls if judgement else 0
    ruled = found.verdicts[mode] if found else None
    posthog_client.capture(
        user_id,
        PostHogEvent.CHAT_GATE_SHADOW_DECISION,
        {
            **event,
            "effect": found.effect.value if found else None,
            "subject_kind": found.subject_kind if found else None,
            "verdict": ruled.verdict if ruled else "error",
            "decided_by": ruled.decided_by if ruled else None,
            "reason_code": ruled.reason_code if ruled else "exception",
            **{
                f"verdict_{other}": found.verdicts[other].verdict if found else None
                for other in MODES
            },
            "judge_error": judgement.judge_error if judgement else None,
            "jev_calls": jev_calls,
            "llm_calls": llm_calls,
            "jev_model": config.gate_jev_model if jev_calls else None,
            "llm_model": config.gate_model if llm_calls else None,
            "rubric_version": RUBRIC_VERSION if judgement else None,
            "latency_ms": latency_ms,
        },
        source=SOURCE,
    )


def _judged_by(judgement: Judgement) -> DecidedBy:
    # No model answered (too long, unreadable files): the policy asks.
    if judgement.decided_by is None:
        return "policy"
    return "llm" if judgement.decided_by == "llm" else "jev"


def _subject_kind(subject: Subject | None) -> str | None:
    if subject is None:
        return None
    kind, colon, _ = subject.key.partition(":")
    return kind if colon else "tool"
