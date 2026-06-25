"""Per-user FIFO queue for AutoPilot chat turns that exceeded the soft
running cap.

Queue state lives on :class:`prisma.models.ChatSession`'s ``chatStatus``
text column:

* ``"idle"``    — DEFAULT, no turn in flight (the 99% case)
* ``"queued"``  — task waiting for a running slot to free
* ``"running"`` — turn currently being processed

The user's pending message itself is just a normal ChatMessage row (no
status of its own).  The dispatcher's submit-time payload (``file_ids``,
``mode``, ``model``, ``permissions``, ``context``, ``request_arrival_at``)
is stashed in that row's ``metadata`` JSONB so a later promotion can
replay the turn faithfully.

State transitions live in :func:`backend.copilot.db.update_chat_session_status`:

* (insert + flip)  ``"idle"`` → ``"queued"`` via :func:`enqueue_turn`
* ``"queued"``     → ``"running"`` via :func:`claim_queued_turn` (dispatcher)
* ``"queued"``     → ``"idle"``    via :func:`cancel_queued_turn`  (user)
* ``"running"``    → ``"idle"``    via :func:`backend.copilot.active_turns.release_turn_slot`
* ``"running"``    → ``"queued"``  on dispatch-failure restore

If the dispatcher finds the user paywalled / rate-limited at promote
time, the session stays ``"queued"`` and the next slot-free hook
re-validates — auto-recovers when eligibility returns, or the user
cancels manually.
"""

import logging
import uuid
from typing import Any, Mapping

from backend.copilot.active_turns import TurnSlot, count_running_turns
from backend.copilot.config import ChatConfig
from backend.copilot.model import (
    CHAT_STATUS_IDLE,
    CHAT_STATUS_QUEUED,
    CHAT_STATUS_RUNNING,
    ChatMessage,
    _get_session_lock,
    invalidate_session_cache,
)
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    check_rate_limit,
    get_global_rate_limits,
    is_user_paywalled,
)
from backend.data.db_accessors import chat_db

logger = logging.getLogger(__name__)


async def count_queued_turns(user_id: str) -> int:
    """Number of ``chatStatus='queued'`` ChatSession rows for ``user_id``."""
    return await chat_db().count_chat_sessions_by_status(
        user_id=user_id, status=CHAT_STATUS_QUEUED
    )


async def count_inflight_turns(user_id: str) -> int:
    """Running + queued. Hard cap is enforced against this.

    Counts queued first then running so a concurrent queued→running
    promotion between the two reads can be double-counted (safe — caller
    rejects one extra task) but never missed.  The cap may briefly read
    high under burst load, never low.
    """
    queued = await count_queued_turns(user_id)
    running = await count_running_turns(user_id)
    return queued + running


async def list_queued_sessions(user_id: str):
    """User's queued sessions, oldest-first (FIFO order).  UX surface
    for the 'your queued tasks' panel."""
    return await chat_db().list_chat_sessions_by_status(
        user_id=user_id, status=CHAT_STATUS_QUEUED
    )


class InflightCapExceeded(Exception):
    """User's running + queued total has reached the configured hard cap.

    Raised by :func:`try_enqueue_turn` so the route can map to HTTP 429.
    """


async def try_enqueue_turn(
    *,
    user_id: str,
    inflight_cap: int,
    session_id: str,
    message: str,
    message_id: str | None = None,
    is_user_message: bool = True,
    context: Mapping[str, str] | None = None,
    file_ids: list[str] | None = None,
    mode: str | None = None,
    model: str | None = None,
    permissions: Mapping[str, Any] | None = None,
    request_arrival_at: float = 0.0,
) -> ChatMessage:
    """Admit a queued turn against the user's hard cap.

    Non-locked count-then-insert: under burst, two concurrent submits
    can both pass the count and both insert, leaving the user briefly
    one or two over the cap.  Same trade-off the graph-execution credit
    rate-limit accepts on its INCRBY path; the cap is a safeguard, not
    a budget.
    """
    if await count_inflight_turns(user_id) >= inflight_cap:
        raise InflightCapExceeded()
    return await enqueue_turn(
        user_id=user_id,
        session_id=session_id,
        message=message,
        message_id=message_id,
        is_user_message=is_user_message,
        context=context,
        file_ids=file_ids,
        mode=mode,
        model=model,
        permissions=permissions,
        request_arrival_at=request_arrival_at,
    )


async def enqueue_turn(
    *,
    user_id: str,
    session_id: str,
    message: str,
    message_id: str | None = None,
    is_user_message: bool = True,
    context: Mapping[str, str] | None = None,
    file_ids: list[str] | None = None,
    mode: str | None = None,
    model: str | None = None,
    permissions: Mapping[str, Any] | None = None,
    request_arrival_at: float = 0.0,
) -> ChatMessage:
    """Persist the user's pending message and flip the session to
    ``"queued"``.  Caller is responsible for the in-flight cap check
    AND session-ownership check upstream — once the row is committed
    the dispatcher owns it.

    The user message is a regular ChatMessage row (no special status).
    The dispatcher's submit-time payload is stashed in the row's
    ``metadata`` JSONB so a later promotion replays the turn faithfully.
    """
    metadata: dict[str, Any] = {}
    if context is not None:
        metadata["context"] = dict(context)
    if file_ids is not None:
        metadata["file_ids"] = list(file_ids)
    if mode is not None:
        metadata["mode"] = mode
    if model is not None:
        metadata["model"] = model
    if permissions is not None:
        metadata["permissions"] = dict(permissions)
    if request_arrival_at:
        metadata["request_arrival_at"] = request_arrival_at

    # The Redis NX session lock serialises with ``append_and_save_message``
    # so two concurrent submits to the same session can't pick the same
    # ``sequence`` and PK-collide on ``(sessionId, sequence)``.
    db = chat_db()
    async with _get_session_lock(session_id):
        live_sequence = await db.get_next_sequence(session_id)
        row = await db.add_chat_message(
            message_id=message_id or str(uuid.uuid4()),
            session_id=session_id,
            role="user" if is_user_message else "assistant",
            content=message,
            sequence=live_sequence,
            metadata=metadata or None,
        )
    # Flip the session to ``"queued"``.  CAS-gated on ``"idle"`` so a
    # double-submit (session already queued/running) leaves the state
    # alone; the second pending message persists as a normal ChatMessage
    # row.  When the session eventually promotes, the dispatcher reads
    # the most-recent user row via ``get_latest_user_message_in_session``;
    # earlier pending rows aren't independently scheduled, they sit in
    # the chat history and the model sees them as context.
    await db.update_chat_session_status(
        session_id=session_id,
        expect_status=CHAT_STATUS_IDLE,
        status=CHAT_STATUS_QUEUED,
        user_id=user_id,
    )
    # Invalidate the session cache so the next /chat read picks up the
    # queued row + the session's new status (frontend renders the
    # 'Queued' badge from ``session.chat_status``).
    await invalidate_session_cache(session_id)
    return row


async def cancel_queued_turn(*, user_id: str, session_id: str) -> bool:
    """Flip the user's session from ``"queued"`` → ``"idle"``.  Returns
    True iff the CAS matched AND the session is owned by the user.
    Cancel/dispatch races resolve in a single atomic update."""
    ok = await chat_db().update_chat_session_status(
        session_id=session_id,
        expect_status=CHAT_STATUS_QUEUED,
        status=CHAT_STATUS_IDLE,
        user_id=user_id,
    )
    if not ok:
        return False
    await invalidate_session_cache(session_id)
    return True


async def claim_queued_session(session_id: str) -> bool:
    """Atomically claim a queued session by transitioning ``chatStatus``
    ``"queued"`` → ``"running"``.  Returns True iff the CAS matched
    (i.e. the session was still queued; not cancelled / claimed by a
    concurrent dispatcher)."""
    return await chat_db().update_chat_session_status(
        session_id=session_id,
        expect_status=CHAT_STATUS_QUEUED,
        status=CHAT_STATUS_RUNNING,
    )


async def dispatch_next_for_user(user_id: str) -> bool:
    """Promote at most one queued session for ``user_id`` from queued →
    running.  Called by ``mark_session_completed`` after every turn
    ends — the slot-free hook is the only dispatcher trigger.

    Returns ``True`` iff a session was actually promoted.

    Pre-start re-validation runs *before* claiming so a paywalled
    user's queue head stays queued (rather than consuming a running
    slot for a turn that would immediately 402).  Auto-recovers on
    the next completion-driven tick once eligibility returns, or the
    user cancels manually.
    """
    # ``executor.utils`` stays a local import: it pulls
    # ``turn_queue.count_inflight_turns`` lazily back through this module,
    # so top-leveling it here would deadlock the import graph.
    from backend.copilot.executor.utils import dispatch_turn

    queued = await list_queued_sessions(user_id)
    if not queued:
        return False
    head = queued[0]

    if await is_user_paywalled(user_id):
        logger.info(
            "dispatch_next_for_user: user=%s paywalled, leaving session=%s queued",
            user_id,
            head.session_id,
        )
        return False

    cfg = ChatConfig()
    try:
        daily_limit, weekly_limit, _ = await get_global_rate_limits(
            user_id,
            cfg.daily_cost_limit_microdollars,
            cfg.weekly_cost_limit_microdollars,
        )
        await check_rate_limit(
            user_id=user_id,
            daily_cost_limit=daily_limit,
            weekly_cost_limit=weekly_limit,
        )
    except RateLimitExceeded as exc:
        logger.info(
            "dispatch_next_for_user: user=%s rate-limited (%s), leaving session=%s queued",
            user_id,
            exc,
            head.session_id,
        )
        return False
    except RateLimitUnavailable:
        logger.warning(
            "dispatch_next_for_user: rate-limit service degraded for user=%s; "
            "leaving queue intact for the next tick",
            user_id,
        )
        return False

    # Claim by transitioning the session ``queued`` → ``running``.  A
    # parallel cancel between validation and claim rejects this
    # dispatch via the CAS returning False.
    if not await claim_queued_session(head.session_id):
        return False

    # Find the pending user message in this session (the most recent
    # user-role row with no following assistant rows — i.e. the one
    # that triggered the queue).  Its ``metadata`` carries the
    # dispatcher payload.
    pending = await chat_db().get_latest_user_message_in_session(head.session_id)
    if pending is None or pending.content is None:
        # Shouldn't happen — enqueue_turn always persists a row before
        # flipping the session to queued.  If it does (corrupted
        # state), roll back to idle so the next tick doesn't loop.
        await chat_db().update_chat_session_status(
            session_id=head.session_id,
            expect_status=CHAT_STATUS_RUNNING,
            status=CHAT_STATUS_IDLE,
        )
        # Drop the cache so the sidebar doesn't keep showing the
        # stale ``running`` indicator after the rollback.
        await invalidate_session_cache(head.session_id)
        return False

    metadata = pending.metadata or {}
    turn_id = str(uuid.uuid4())
    try:
        # The user's message is already persisted AND the session is
        # already ``chatStatus='running'`` from claim_queued_session.
        # Build a TurnSlot directly (no acquire) so we don't re-check
        # the cap (would over-count our own just-promoted session) and
        # don't re-flip the status (already running).  ``dispatch_turn``
        # calls ``slot.keep()`` internally; release happens via
        # ``mark_session_completed`` → ``release_turn_slot``.
        slot = TurnSlot(user_id, head.session_id)
        slot.admitted = True
        await dispatch_turn(
            slot,
            session_id=head.session_id,
            user_id=user_id,
            turn_id=turn_id,
            message=pending.content,
            is_user_message=pending.role == "user",
            context=metadata.get("context"),
            file_ids=metadata.get("file_ids"),
            mode=metadata.get("mode"),
            model=metadata.get("model"),
            permissions=metadata.get("permissions"),
            request_arrival_at=float(metadata.get("request_arrival_at") or 0.0),
        )
    except BaseException:
        # Roll the claim back so a missed-dispatch tick or the next
        # slot-free event can retry.  ``BaseException`` (not just
        # ``Exception``) so a task cancellation that lands mid-dispatch
        # still leaves the session in a recoverable ``queued`` state
        # rather than a stuck ``running``.  Redis-side cleanup of the
        # meta that ``dispatch_turn``'s ``create_session`` wrote is
        # handled inside ``dispatch_turn`` itself (try/finally on its
        # ``committed`` flag), so we only restore the DB side here.
        try:
            await chat_db().update_chat_session_status(
                session_id=head.session_id,
                expect_status=CHAT_STATUS_RUNNING,
                status=CHAT_STATUS_QUEUED,
            )
            await invalidate_session_cache(head.session_id)
        except BaseException as restore_exc:
            logger.error(
                "dispatch_next_for_user: failed to restore claim for "
                "session=%s after dispatch failure; session left in "
                "chatStatus='running' and will need manual recovery: %s",
                head.session_id,
                restore_exc,
            )
        raise

    await invalidate_session_cache(head.session_id)
    return True
