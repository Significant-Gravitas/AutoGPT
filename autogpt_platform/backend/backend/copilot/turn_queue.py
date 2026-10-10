"""Per-user FIFO queue for Otto chat turns that exceeded the soft
running cap.

Queue state lives on :class:`prisma.models.ChatSession`'s ``chatStatus``
text column:

* ``"idle"``    — DEFAULT, no turn in flight (the 99% case)
* ``"queued"``  — task waiting for a running slot to free
* ``"running"`` — turn currently being processed

The user's pending message itself is just a normal ChatMessage row (no
status of its own).  The dispatcher's submit-time payload (``file_ids``,
``model``, ``permissions``, ``context``, ``request_arrival_at``)
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
from typing import Any, Literal, Mapping

from prisma.errors import UniqueViolationError
from pydantic import BaseModel, ValidationError

from backend.copilot.active_turns import (
    TurnSlot,
    count_running_turns,
    get_delegated_turn_limit,
    get_running_turn_limit,
)
from backend.copilot.config import ChatConfig, CopilotLlmAuthProvider
from backend.copilot.db import is_duplicate_chat_message_id_error
from backend.copilot.model import (
    CHAT_STATUS_IDLE,
    CHAT_STATUS_QUEUED,
    CHAT_STATUS_RUNNING,
    ChatMessage,
    ChatSession,
    ChatSessionInfo,
    _get_session_lock,
    append_and_save_message,
    invalidate_session_cache,
)
from backend.copilot.offers import EntitlementUnavailable, advanced_tier_entitled
from backend.copilot.permissions import ALL_TOOL_NAMES, CopilotPermissions
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    check_rate_limit,
    get_global_rate_limits,
    is_user_paywalled,
)
from backend.copilot.session_permissions import resolve_session_permissions
from backend.copilot.tracking import track_user_message
from backend.copilot.tree import TreeRefusal, TurnEnvelope, release_turn
from backend.data.db_accessors import chat_db
from backend.integrations.codex.access import has_codex_access

logger = logging.getLogger(__name__)

# Pending-row metadata: the envelope a queued approval wake starts under.
_ENVELOPE_KEY = "envelope"
# The assistant row that closes a queued turn the promotion could not start.
_REFUSED_KEY = "queued_turn_refused"
WAKE_LATER = "Your answer is kept and reaches the assistant with your next message."
TURN_CANCELLED = "This task was cancelled before it started."
UNREADABLE_SPAWN = (
    "This task's limits could not be read back, so it was not started. "
    "Start it again."
)
UNRECORDED_WAKE = (
    "The approved action could not be resumed: what it was allowed to do was "
    f"not recorded. {WAKE_LATER}"
)


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


class SessionNotIdle(Exception):
    """The session already has a turn queued or running, so a spawned child
    cannot be queued into it."""


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
    message_metadata: Mapping[str, Any] | None = None,
    is_user_message: bool = True,
    context: Mapping[str, str] | None = None,
    file_ids: list[str] | None = None,
    model: str | None = None,
    llm_auth_provider: CopilotLlmAuthProvider = "platform",
    llm_credential_id: str | None = None,
    permissions: Mapping[str, Any] | None = None,
    request_arrival_at: float = 0.0,
    envelope: TurnEnvelope | None = None,
) -> ChatMessage | None:
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
        message_metadata=message_metadata,
        is_user_message=is_user_message,
        context=context,
        file_ids=file_ids,
        model=model,
        llm_auth_provider=llm_auth_provider,
        llm_credential_id=llm_credential_id,
        permissions=permissions,
        request_arrival_at=request_arrival_at,
        envelope=envelope,
    )


async def enqueue_turn(
    *,
    user_id: str,
    session_id: str,
    message: str,
    message_id: str | None = None,
    message_metadata: Mapping[str, Any] | None = None,
    is_user_message: bool = True,
    context: Mapping[str, str] | None = None,
    file_ids: list[str] | None = None,
    model: str | None = None,
    llm_auth_provider: CopilotLlmAuthProvider = "platform",
    llm_credential_id: str | None = None,
    permissions: Mapping[str, Any] | None = None,
    request_arrival_at: float = 0.0,
    envelope: TurnEnvelope | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    only_if_idle: bool = False,
) -> ChatMessage | None:
    """Persist the user's pending message and flip the session to
    ``"queued"``.  Caller is responsible for the in-flight cap check
    AND session-ownership check upstream — once the row is committed
    the dispatcher owns it.

    The user message is a regular ChatMessage row (no special status).
    The dispatcher's submit-time payload is stashed in the row's
    ``metadata`` JSONB so a later promotion replays the turn faithfully.
    """
    metadata = dict(message_metadata or {})
    if context is not None:
        metadata["context"] = dict(context)
    if file_ids is not None:
        metadata["file_ids"] = list(file_ids)
    if model is not None:
        metadata["model"] = model
    metadata["llm_auth_provider"] = llm_auth_provider
    if llm_credential_id is not None:
        metadata["llm_credential_id"] = llm_credential_id
    if permissions is not None:
        metadata["permissions"] = dict(permissions)
    if request_arrival_at:
        metadata["request_arrival_at"] = request_arrival_at
    # A wake starts under the envelope its call was held under, and a spawned
    # child under the one admitted for it, whichever turn frees the slot.
    if envelope is not None:
        metadata[_ENVELOPE_KEY] = envelope.model_dump(mode="json")
    if tool_call_id is not None:
        metadata["tool_call_id"] = tool_call_id
    if tool_name is not None:
        metadata["tool_name"] = tool_name

    # The Redis NX session lock serialises with ``append_and_save_message``
    # so two concurrent submits to the same session can't pick the same
    # ``sequence`` and PK-collide on ``(sessionId, sequence)``.
    db = chat_db()
    async with _get_session_lock(session_id):
        # A spawned child's row is the one promotion replays, so it claims the
        # idle session first and is written only if it won: a lost claim leaves
        # nothing behind. Promotion reads the row under this lock, so it never
        # sees the claim without it.
        if only_if_idle and not await _flip(db, session_id, user_id):
            raise SessionNotIdle(session_id)
        live_sequence = await db.get_next_sequence(session_id)
        try:
            row = await db.add_chat_message(
                message_id=message_id or str(uuid.uuid4()),
                session_id=session_id,
                role="user" if is_user_message else "assistant",
                content=message,
                sequence=live_sequence,
                metadata=metadata or None,
            )
        except BaseException as exc:
            if only_if_idle:
                await db.update_chat_session_status(
                    session_id=session_id,
                    expect_status=CHAT_STATUS_QUEUED,
                    status=CHAT_STATUS_IDLE,
                    user_id=user_id,
                )
            if (
                isinstance(exc, UniqueViolationError)
                and message_id
                and is_duplicate_chat_message_id_error(exc)
            ):
                return None
            raise
        # Flip the session to ``"queued"``.  CAS-gated on ``"idle"`` so a
        # double-submit (session already queued/running) leaves the state
        # alone; the second pending message persists as a normal ChatMessage
        # row.  When the session eventually promotes, the dispatcher reads
        # the most-recent user row via ``get_latest_user_message_in_session``;
        # earlier pending rows aren't independently scheduled, they sit in
        # the chat history and the model sees them as context.
        if not only_if_idle:
            await _flip(db, session_id, user_id)
    # Invalidate the session cache so the next /chat read picks up the
    # queued row + the session's new status (frontend renders the
    # 'Queued' badge from ``session.chat_status``).
    await invalidate_session_cache(session_id)
    return row


async def _flip(db: Any, session_id: str, user_id: str) -> bool:
    return await db.update_chat_session_status(
        session_id=session_id,
        expect_status=CHAT_STATUS_IDLE,
        status=CHAT_STATUS_QUEUED,
        user_id=user_id,
    )


async def cancel_queued_turn(*, user_id: str, session_id: str) -> bool:
    """Flip the user's session from ``"queued"`` → ``"idle"``.  A queued child
    gets a closing row saying it was cancelled, which its waiter reads, and its
    node goes back to its tree.  Returns True iff the CAS matched AND the
    session is owned by the user.  Cancel/dispatch races resolve in a single
    atomic update."""
    db = chat_db()
    # Under the lock a waiter reads under, so it never sees idle without the note.
    async with _get_session_lock(session_id):
        # Read before the flip: once idle, a new message can become the latest row.
        pending = await db.get_latest_user_message_in_session(session_id)
        if not await db.update_chat_session_status(
            session_id=session_id,
            expect_status=CHAT_STATUS_QUEUED,
            status=CHAT_STATUS_IDLE,
            user_id=user_id,
        ):
            return False
        spawned = pending is not None and _is_spawned(pending)
        if spawned:
            await db.add_chat_message(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content=TURN_CANCELLED,
                sequence=await db.get_next_sequence(session_id),
                metadata={_REFUSED_KEY: True},
            )
        await invalidate_session_cache(session_id)
    child = _stored_envelope(pending.metadata or {}) if spawned and pending else None
    if child is not None:
        await release_turn(child)
    return True


async def claim_queued_session(
    session: ChatSessionInfo, *, sub_work: bool
) -> Literal["admitted", "full", "busy"]:
    """Claim a queued session, ``"queued"`` → ``"running"``, if the user has a
    slot for it: sub-work below the reserve, their own message below the cap.
    ``"busy"`` when it was cancelled or claimed elsewhere since it was read."""
    return await chat_db().admit_chat_session_turn(
        session_id=session.session_id,
        user_id=session.user_id,
        expect_status=CHAT_STATUS_QUEUED,
        capacity=get_delegated_turn_limit() if sub_work else get_running_turn_limit(),
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
    # A head that may never start is closed and the next one tried, in a loop:
    # recursing once per closed row overflows on a long run of them.
    while (promoted := await _promote_head(user_id)) is None:
        pass
    return promoted


async def _promote_head(user_id: str) -> bool | None:
    """One promotion attempt: whether a session was promoted, or ``None`` when
    the head was closed as one that may never start, so the next can be tried."""
    # ``executor.utils`` stays a local import: it pulls
    # ``turn_queue.count_inflight_turns`` lazily back through this module,
    # so top-leveling it here would deadlock the import graph.
    from backend.copilot.executor.utils import dispatch_turn

    # Local for the same reason: the gate imports this module back.
    from backend.copilot.gate.held import is_answer_row

    # The user's own turns go before sub-work, oldest first in each, so sub-work
    # that does not fit means nothing behind it does; one that may not start yet
    # does not hold up the rest.
    queued = await list_queued_sessions(user_id)
    sub_work = {s.session_id: await _is_sub_work(s) for s in queued}
    candidates = sorted(queued, key=lambda s: sub_work[s.session_id])
    gates = _UserGates(user_id)
    head = None
    try:
        for candidate in candidates:
            if await _may_start(gates, candidate):
                head = candidate
                break
    except _QueueOnHold as hold:
        logger.warning(
            f"dispatch_next_for_user: user={user_id} {hold}; "
            "leaving queue intact for the next tick"
        )
        return False
    if head is None:
        return False

    claim = await claim_queued_session(head, sub_work=sub_work[head.session_id])
    if claim == "full":
        return False
    if claim == "busy":
        # Cancelled or claimed elsewhere since it was listed: try the next.
        return None

    # The admitted envelope of a queued child, whose node a refusal gives back.
    child: TurnEnvelope | None = None
    spawned = False
    try:
        # Find the pending user message in this session (the most recent
        # user-role row with no following assistant rows — i.e. the one
        # that triggered the queue).  Its ``metadata`` carries the
        # dispatcher payload.
        # Under the session lock: a spawned child claims the session before it
        # writes its row, inside the same lock.
        async with _get_session_lock(head.session_id):
            pending = await chat_db().get_latest_user_message_in_session(
                head.session_id
            )
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
        queued_envelope = _stored_envelope(metadata)
        spawned = _is_spawned(pending)
        if spawned and queued_envelope is None:
            # Nothing but that envelope bounds a child its spawner admitted.
            await _refuse_queued_turn(head, UNREADABLE_SPAWN)
            return None
        child = queued_envelope if spawned else None
        if (
            is_answer_row(pending)
            and queued_envelope is None
            and not is_users_own_chat(head)
        ):
            # Deriving one from the turn that just ended would run the approved
            # action under that turn's limits; the answer reaches the next turn.
            await _refuse_queued_turn(head, UNRECORDED_WAKE)
            return None

        turn_id = str(uuid.uuid4())
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
            # Session-anchored tenancy: promoted turns attribute to the
            # session's org/team, same as directly-dispatched turns —
            # without this, capped users' queued turns would lose their
            # org context on promotion.
            organization_id=head.organization_id,
            team_id=head.team_id,
            model=metadata.get("model"),
            llm_auth_provider=head.metadata.llm_auth_provider,
            llm_credential_id=head.metadata.llm_credential_id,
            permissions=_promotion_permissions(head, metadata, spawned=spawned),
            request_arrival_at=float(metadata.get("request_arrival_at") or 0.0),
            tool_call_id=metadata.get("tool_call_id") or "chat_stream",
            tool_name=metadata.get("tool_name") or "chat",
            # A typed message is a root, as the chat route makes it, not the
            # child of the turn this hook runs in; a wake or a child brings its own.
            envelope=queued_envelope,
            root=queued_envelope is None,
        )
    except TreeRefusal as refused:
        # Only a stored envelope is re-checked here; a root is never refused.
        await _refuse_queued_turn(
            head,
            refused.message if spawned else f"{refused.message} {WAKE_LATER}",
            release=child,
        )
        return None
    except BaseException:
        # Roll the claim back so a missed-dispatch tick or the next
        # slot-free event can retry.  ``BaseException`` (not just
        # ``Exception``) so a task cancellation that lands after the claim
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

    if pending.role == "user" and pending.content and not spawned:
        try:
            track_user_message(
                user_id=user_id,
                session_id=head.session_id,
                message_length=len(pending.content),
                expert_id=head.expert_id,
                origin=head.metadata.origin,
                source_platform=head.metadata.source_platform,
            )
        except Exception:
            logger.warning("Failed to track promoted chat turn", exc_info=True)

    await invalidate_session_cache(head.session_id)
    return True


async def _is_sub_work(session: ChatSessionInfo) -> bool:
    """A message the user typed is theirs whatever session it is in; an
    approval wake or a spawned child there is sub-work if
    :func:`wakes_sub_work` says so."""
    if not wakes_sub_work(session):
        return False
    # Local: the gate imports this module back.
    from backend.copilot.gate.held import is_answer_row

    waiting = await chat_db().get_latest_user_message_in_session(session.session_id)
    return waiting is not None and (is_answer_row(waiting) or _is_spawned(waiting))


async def _may_start(gates: "_UserGates", head: ChatSessionInfo) -> bool:
    """Whether ``head`` may start now: :func:`turn_refusal` for its route and,
    on the platform route, an Advanced turn's tier."""
    route_provider = head.metadata.llm_auth_provider
    if not await gates.route_open(route_provider):
        return False
    if route_provider != "platform":
        return True
    # A turn can sit in the queue long enough for the plan that bought it to
    # lapse. The tier was checked when the turn was accepted, but promoting it
    # is a second, later decision to spend, so it gets its own check --
    # otherwise a downgrade between the two buys a free Advanced run. The turn
    # stays queued rather than quietly re-running on Standard: nothing in this
    # feature changes what a turn runs on without being asked. It promotes
    # itself once entitlement returns, and can be cancelled meanwhile.
    pending = await chat_db().get_latest_user_message_in_session(head.session_id)
    model = (pending.metadata or {}).get("model") if pending else None
    if model == "advanced" and not await gates.advanced_tier():
        logger.info(
            f"dispatch_next_for_user: user={gates.user_id} lacks the Advanced tier, "
            f"leaving session={head.session_id} queued"
        )
        return False
    return True


class _UserGates:
    """The per-user checks, made once per pass over the queue however many
    queued sessions are tried."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._refusals: dict[CopilotLlmAuthProvider, TurnRefusal | None] = {}
        self._advanced: bool | None = None

    async def route_open(self, route: CopilotLlmAuthProvider) -> bool:
        """Raises :class:`_QueueOnHold` when the gate could not read the user's
        state, which leaves the whole queue for the next tick."""
        if route not in self._refusals:
            self._refusals[route] = await turn_refusal(self.user_id, route)
        refusal = self._refusals[route]
        if refusal is None:
            return True
        if refusal.transient:
            raise _QueueOnHold(refusal.reason)
        logger.info(f"dispatch_next_for_user: user={self.user_id} {refusal.reason}")
        return False

    async def advanced_tier(self) -> bool:
        if self._advanced is None:
            try:
                self._advanced = await advanced_tier_entitled(self.user_id)
            except EntitlementUnavailable:
                logger.warning(
                    "dispatch_next_for_user: could not resolve the Advanced "
                    f"entitlement for user={self.user_id}",
                    exc_info=True,
                )
                self._advanced = False
        return self._advanced


class _QueueOnHold(Exception):
    """Nothing is promoted this tick: the user's state could not be read."""


def is_users_own_chat(session: ChatSessionInfo) -> bool:
    """A chat the user opened themselves, not one another session or a graph
    started. Its turns are roots, so a wake there whose envelope was lost can
    start as one; elsewhere nothing on the row bounds what the call held."""
    return (
        session.metadata.origin == "interactive"
        and session.metadata.delegated_by_session_id is None
    )


def wakes_sub_work(session: ChatSessionInfo) -> bool:
    """An approval wake or a spawned child here is the sub-work of the session
    that opened this one, so it is admitted within the user's reserve."""
    return session.metadata.delegated_by_session_id is not None


async def _refuse_queued_turn(
    head: ChatSessionInfo, reason: str, *, release: TurnEnvelope | None = None
) -> None:
    """Close a promoted turn that may not start: say why in its thread, free
    its slot, which a failed post must not keep, and give back ``release``'s
    node: the node a queued child was admitted with."""
    try:
        await post_refusal(head.session_id, reason)
    except Exception:
        # Not raised: the drain goes on, and the next queued turn takes the slot.
        logger.exception(
            f"dispatch_next_for_user: could not post why session={head.session_id} "
            "was not started"
        )
    finally:
        await chat_db().update_chat_session_status(
            session_id=head.session_id,
            expect_status=CHAT_STATUS_RUNNING,
            status=CHAT_STATUS_IDLE,
        )
        await invalidate_session_cache(head.session_id)
        if release is not None:
            await release_turn(release)


async def post_refusal(
    session_id: str, reason: str, *, message_id: str | None = None
) -> None:
    """Say in the thread why a turn that was waiting will not start."""
    await append_and_save_message(
        session_id,
        ChatMessage(
            id=message_id,
            role="assistant",
            content=reason,
            metadata={_REFUSED_KEY: True},
        ),
    )


def queued_turn_refusal(session: ChatSession) -> str | None:
    """Why a waiting turn was not started, if its thread ends that way."""
    last = session.messages[-1] if session.messages else None
    if last is None or not (last.metadata or {}).get(_REFUSED_KEY):
        return None
    return last.content


def _is_spawned(pending: ChatMessage) -> bool:
    """A queued child: it carries the envelope admitted for it, where a typed
    message carries none and a wake carries its holding turn's."""
    # Local: the gate imports this module back.
    from backend.copilot.gate.held import is_answer_row

    return _ENVELOPE_KEY in (pending.metadata or {}) and not is_answer_row(pending)


def _stored_envelope(metadata: Mapping[str, Any]) -> TurnEnvelope | None:
    """The envelope a wake or a child was queued under. One that no longer
    parses (a schema change between deploys) counts as unrecorded."""
    stored = metadata.get(_ENVELOPE_KEY)
    if not stored:
        return None
    try:
        return TurnEnvelope.model_validate(stored)
    except ValidationError:
        logger.warning("dispatch_next_for_user: a stored envelope did not parse")
        return None


def _promotion_permissions(
    head: ChatSessionInfo, metadata: Mapping[str, Any], *, spawned: bool
) -> CopilotPermissions | None:
    """The session's permissions as they are now, never looser than the ones
    stored at enqueue. Those are the session's own, so a value that no longer
    parses is read afresh; a child's are its spawner's, so it refuses instead."""
    current = resolve_session_permissions(head)
    stored = metadata.get("permissions")
    if not stored:
        return current
    try:
        queued = CopilotPermissions.model_validate(stored)
    except ValidationError:
        if spawned:
            raise TreeRefusal(UNREADABLE_SPAWN)
        logger.warning(
            f"dispatch_next_for_user: stored permissions on session={head.session_id} "
            "did not parse; using the session's current ones"
        )
        return current
    return (
        queued
        if current is None
        else queued.merged_with_parent(current, ALL_TOOL_NAMES)
    )


class TurnRefusal(BaseModel):
    reason: str
    # The usage store was unreadable: an outage, not the user's state.
    transient: bool = False


async def turn_refusal(
    user_id: str, llm_auth_provider: CopilotLlmAuthProvider
) -> TurnRefusal | None:
    """Why a turn billed to this route may not start for this user, if so.

    Access first (Codex entitlement, the platform paywall), then the platform
    route's usage: its daily and weekly windows and a trial's total budget.
    Lookup errors propagate, so each caller decides what an unknown means.
    """
    if llm_auth_provider == "codex":
        if await has_codex_access(user_id):
            return None
        return TurnRefusal(reason="lacks Codex entitlement")
    if llm_auth_provider != "platform":
        return None
    if await is_user_paywalled(user_id):
        return TurnRefusal(reason="paywalled")
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
        return TurnRefusal(reason=f"rate-limited ({exc})")
    except RateLimitUnavailable:
        return TurnRefusal(reason="has unreadable usage limits", transient=True)
    return None
