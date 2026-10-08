"""One heartbeat, end to end.

Cheapest check first, and the model call last:

1. settings off                         -> skipped (unless forced)
2. checklist effectively empty          -> skipped
3. outside the active hours             -> skipped (unless forced)
4. a turn of the user's already running -> skipped
5. no new agent run and no new chat since the last beat -> skipped (unless
   forced); the first beat always runs
6. an isolated turn in a fresh hidden session, on the cheap tier, with only
   read tools and ``heartbeat_respond``
7. suppression (``suppress.decide``), then the 24-hour repeat check
8. delivery
"""

import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from backend.copilot.active_turns import ConcurrentTurnLimitError, count_running_turns
from backend.copilot.config import CopilotLlmAuthProvider, CopilotLLMModel
from backend.copilot.model import create_chat_session, update_session_title
from backend.copilot.permissions import ALL_TOOL_NAMES, CopilotPermissions
from backend.copilot.transports import resolve_default_chat_route
from backend.data.db_accessors import chat_db, execution_db

from . import state
from .config import (
    HeartbeatConfig,
    checklist_is_empty,
    in_active_hours,
    load_config,
    resolve_timezone,
)
from .delivery import DeliveryReport, deliver_alert
from .prompt import RESPOND_TOOL, build_prompt
from .suppress import ExplicitResponse, decide, explicit_from_tool_calls

logger = logging.getLogger(__name__)

# Under the scheduler's 300 s operation timeout, so a slow beat is reported
# as one rather than killed mid-delivery.
TURN_TIMEOUT_SECONDS = 240
# Enough recent chats to see one opened since the last beat behind a few that
# were only written to.
_RECENT_SESSIONS = 10
# Read tools a heartbeat may not use: nobody answers a question, a teammate
# consult and the browser cost real money, and onboarding opens a card.
_EXCLUDED_READS = frozenset(
    {
        "ask_question",
        "consult_teammate",
        "browser_navigate",
        "browser_screenshot",
        "decompose_goal",
        "expert_onboarding",
    }
)

Status = Literal["skipped", "silent", "delivered", "failed"]


class HeartbeatTurn(BaseModel):
    user_id: str
    session_id: str
    prompt: str
    model_tier: CopilotLLMModel
    permissions: CopilotPermissions
    llm_auth_provider: CopilotLlmAuthProvider = "platform"
    llm_credential_id: str | None = None


class TurnResult(BaseModel):
    outcome: str
    response_text: str = ""
    tool_calls: list[Any] = []


class HeartbeatRunResult(BaseModel):
    status: Status
    reason: str
    session_id: str | None = None
    text: str | None = None
    delivery: DeliveryReport | None = None


Engine = Callable[[HeartbeatTurn], Awaitable[TurnResult]]


async def run_heartbeat(
    user_id: str,
    *,
    force: bool = False,
    engine: Engine | None = None,
    now: datetime | None = None,
) -> HeartbeatRunResult:
    """Run one beat for the user. ``force`` (the "run now" endpoint) skips
    the switch, the window and the change check, never the empty checklist
    or a running turn."""
    now = now or datetime.now(UTC)
    config = await load_config(user_id)
    if not config.enabled and not force:
        return HeartbeatRunResult(status="skipped", reason="disabled")
    if checklist_is_empty(config.checklist):
        return HeartbeatRunResult(status="skipped", reason="empty_checklist")
    tz_name = await resolve_timezone(user_id, config)
    now_local = now.astimezone(ZoneInfo(tz_name))
    if not force and not in_active_hours(config, now_local):
        return HeartbeatRunResult(status="skipped", reason="outside_active_hours")
    if await _turn_running(user_id):
        return HeartbeatRunResult(status="skipped", reason="turn_running")
    if not force and not await changed_since_last_run(user_id):
        return HeartbeatRunResult(status="skipped", reason="no_changes")

    llm_auth_provider, llm_credential_id = await resolve_default_chat_route(user_id)
    session_id = await _open_session(user_id, llm_auth_provider, llm_credential_id)
    # Before the model call: a beat that fails must not make the next one
    # think nothing has run since an older beat and burn a call on that.
    await state.set_last_run(user_id, now)
    turn = HeartbeatTurn(
        user_id=user_id,
        session_id=session_id,
        prompt=build_prompt(config.checklist, now_local, tz_name),
        model_tier=config.model_tier,
        permissions=heartbeat_permissions(),
        llm_auth_provider=llm_auth_provider,
        llm_credential_id=llm_credential_id,
    )
    try:
        result = await (engine or queue_engine)(turn)
    except ConcurrentTurnLimitError:
        return HeartbeatRunResult(
            status="skipped", reason="turn_limit", session_id=session_id
        )
    if result.outcome != "completed":
        logger.warning(
            "Heartbeat turn for user %s ended %s", user_id[:12], result.outcome
        )
        return HeartbeatRunResult(
            status="failed", reason=result.outcome, session_id=session_id
        )
    return await _settle(user_id, session_id, config, result)


async def _settle(
    user_id: str, session_id: str, config: HeartbeatConfig, result: TurnResult
) -> HeartbeatRunResult:
    explicit = await _explicit_response(session_id, result)
    verdict = decide(result.response_text, explicit)
    if not verdict.deliver:
        return HeartbeatRunResult(
            status="silent", reason=verdict.reason, session_id=session_id
        )
    if await state.is_repeat_alert(user_id, verdict.text):
        return HeartbeatRunResult(
            status="silent", reason="duplicate", session_id=session_id
        )
    report = await deliver_alert(user_id, session_id, verdict.text, config.delivery)
    await state.remember_alert(user_id, verdict.text)
    return HeartbeatRunResult(
        status="delivered",
        reason=verdict.reason,
        session_id=session_id,
        text=verdict.text,
        delivery=report,
    )


async def changed_since_last_run(user_id: str) -> bool:
    """Whether a new agent run or a new chat appeared since the last beat.

    Fails open: a lookup that errors runs the beat, because a skipped beat
    can hide the very thing the checklist watches for.
    """
    last = await state.get_last_run(user_id)
    if last is None:
        return True
    try:
        runs = await execution_db().get_graph_executions(
            user_id=user_id, created_time_gte=last, limit=1
        )
        if runs:
            return True
        sessions = await chat_db().get_user_chat_sessions(
            user_id, limit=_RECENT_SESSIONS, pinned_first=False
        )
    except Exception:
        logger.warning(
            "Heartbeat could not check for changes for user %s; running anyway",
            user_id[:12],
            exc_info=True,
        )
        return True
    return any(_aware(s.started_at) > last for s in sessions)


def heartbeat_permissions() -> CopilotPermissions:
    """Read tools and ``heartbeat_respond``, as a whitelist: a beat runs with
    nobody watching, so it changes nothing and reaches nothing outward."""
    from backend.copilot.gate.policy import Effect, classified_tools, effect_for

    reads = {
        name
        for name in classified_tools()
        if effect_for(name) is Effect.READ and name not in _EXCLUDED_READS
    }
    allowed = (reads & ALL_TOOL_NAMES) | {RESPOND_TOOL}
    return CopilotPermissions(tools=sorted(allowed), tools_exclude=False)


async def queue_engine(turn: HeartbeatTurn) -> TurnResult:
    """Dispatch the turn on the copilot executor queue and wait for it, as
    ``run_copilot_turn_via_queue`` does, but on the configured model tier."""
    from backend.copilot.executor.utils import schedule_turn
    from backend.copilot.sdk.session_waiter import wait_for_session_result

    await schedule_turn(
        session_id=turn.session_id,
        user_id=turn.user_id,
        turn_id=str(uuid.uuid4()),
        message=turn.prompt,
        tool_call_id="heartbeat",
        tool_name="heartbeat",
        model=turn.model_tier,
        llm_auth_provider=turn.llm_auth_provider,
        llm_credential_id=turn.llm_credential_id,
        permissions=turn.permissions,
        unattended=True,
    )
    outcome, observed = await wait_for_session_result(
        session_id=turn.session_id,
        user_id=turn.user_id,
        timeout=TURN_TIMEOUT_SECONDS,
    )
    return TurnResult(
        outcome=outcome,
        response_text=observed.response_text,
        tool_calls=list(observed.tool_calls),
    )


async def _open_session(
    user_id: str,
    llm_auth_provider: CopilotLlmAuthProvider,
    llm_credential_id: str | None,
) -> str:
    """A fresh hidden session per beat, so no beat reads another's history."""
    session = await create_chat_session(
        user_id,
        dry_run=False,
        # Machine-authored prompt: the staffing tools refuse it and the gate
        # stays out of a turn nobody can answer.
        origin="automation",
        kind=state.HEARTBEAT_SESSION_KIND,
        llm_auth_provider=llm_auth_provider,
        llm_credential_id=llm_credential_id,
    )
    await update_session_title(
        session.session_id, user_id, state.HEARTBEAT_SESSION_TITLE
    )
    return session.session_id


async def _explicit_response(
    session_id: str, result: TurnResult
) -> ExplicitResponse | None:
    recorded = await state.read_response(session_id)
    if recorded is not None:
        return ExplicitResponse(
            notify=bool(recorded.get("notify")),
            notification_text=str(recorded.get("notification_text") or ""),
        )
    return explicit_from_tool_calls(result.tool_calls)


async def _turn_running(user_id: str) -> bool:
    try:
        return await count_running_turns(user_id) > 0
    except Exception:
        logger.warning("Heartbeat could not count running turns", exc_info=True)
        return False


def _aware(when: datetime) -> datetime:
    return when if when.tzinfo else when.replace(tzinfo=UTC)
