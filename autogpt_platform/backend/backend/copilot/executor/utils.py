"""RabbitMQ queue configuration for CoPilot executor.

Defines two exchanges and queues following the graph executor pattern:
- 'copilot_execution' (DIRECT) for chat generation tasks
- 'copilot_cancel' (FANOUT) for cancellation requests
"""

import logging
from typing import Any

from pydantic import BaseModel

from backend.copilot.active_turns import (
    ConcurrentTurnLimitError,
    TurnSlot,
    acquire_turn_slot,
    get_inflight_turn_limit,
    inflight_turn_limit_message,
)
from backend.copilot.config import CopilotLlmAuthProvider, CopilotLLMModel
from backend.copilot.context import get_current_envelope
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.tree import (
    SpawnRequest,
    TreeRefusal,
    TurnEnvelope,
    admit_turn,
    derive_child_envelope,
    release_turn,
    root_envelope,
)
from backend.data.rabbitmq import Exchange, ExchangeType, Queue, RabbitMQConfig
from backend.util.logging import TruncatedLogger, is_structured_logging_enabled

logger = logging.getLogger(__name__)


# ============ Logging Helper ============ #


class CoPilotLogMetadata(TruncatedLogger):
    """Structured logging helper for CoPilot executor.

    In cloud environments (structured logging enabled), uses a simple prefix
    and passes metadata via json_fields. In local environments, uses a detailed
    prefix with all metadata key-value pairs for easier debugging.

    Args:
        logger: The underlying logger instance
        max_length: Maximum log message length before truncation
        **kwargs: Metadata key-value pairs (e.g., session_id="xyz", turn_id="abc")
            These are added to json_fields in cloud mode, or to the prefix in local mode.
    """

    def __init__(
        self,
        logger: logging.Logger,
        max_length: int = 1000,
        **kwargs: str | None,
    ):
        # Filter out None values
        metadata = {k: v for k, v in kwargs.items() if v is not None}
        metadata["component"] = "CoPilotExecutor"

        if is_structured_logging_enabled():
            prefix = "[CoPilotExecutor]"
        else:
            # Build prefix from metadata key-value pairs
            meta_parts = "|".join(
                f"{k}:{v}" for k, v in metadata.items() if k != "component"
            )
            prefix = (
                f"[CoPilotExecutor|{meta_parts}]" if meta_parts else "[CoPilotExecutor]"
            )

        super().__init__(
            logger,
            max_length=max_length,
            prefix=prefix,
            metadata=metadata,
        )


# ============ Exchange and Queue Configuration ============ #

COPILOT_EXECUTION_EXCHANGE = Exchange(
    name="copilot_execution",
    type=ExchangeType.DIRECT,
    durable=True,
    auto_delete=False,
)
# ``_v2`` suffix marks the classic→quorum rollover; old-image consumers
# drain the unsuffixed queue. Orphans cleaned up in a follow-up PR.
COPILOT_EXECUTION_QUEUE_NAME = "copilot_execution_queue_v2"
COPILOT_EXECUTION_ROUTING_KEY = "copilot.run"

COPILOT_CANCEL_EXCHANGE = Exchange(
    name="copilot_cancel",
    type=ExchangeType.FANOUT,
    durable=True,
    auto_delete=False,
)
COPILOT_CANCEL_QUEUE_NAME = "copilot_cancel_queue_v2"


def get_session_lock_key(session_id: str) -> str:
    """Redis key for the per-session cluster lock held by the executing pod."""
    return f"copilot:session:{session_id}:lock"


# CoPilot operations can include extended thinking and agent generation
# which may take several hours to complete. Matches the pod's
# terminationGracePeriodSeconds in the helm chart so a rolling deploy can let
# the longest legitimate turn finish. Also bounds the stale-session auto-
# complete watchdog in stream_registry (consumer_timeout + 5min buffer).
COPILOT_CONSUMER_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours

# Graceful shutdown timeout - must match COPILOT_CONSUMER_TIMEOUT_SECONDS so
# cleanup can let the longest legitimate turn complete before the pod is
# SIGKILL'd by kubelet.
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = COPILOT_CONSUMER_TIMEOUT_SECONDS


def create_copilot_queue_config() -> RabbitMQConfig:
    """Create RabbitMQ configuration for CoPilot executor.

    Defines two exchanges and queues:
    - 'copilot_execution' (DIRECT) for chat generation tasks
    - 'copilot_cancel' (FANOUT) for cancellation requests

    Returns:
        RabbitMQConfig with exchanges and queues defined
    """
    run_queue = Queue(
        name=COPILOT_EXECUTION_QUEUE_NAME,
        exchange=COPILOT_EXECUTION_EXCHANGE,
        routing_key=COPILOT_EXECUTION_ROUTING_KEY,
        durable=True,
        auto_delete=False,
        arguments={
            # Quorum (not classic mirrored) for leader election + stronger
            # replication across RabbitMQ 4.x cluster nodes.
            "x-queue-type": "quorum",
            # Consumer timeout matches the pod graceful-shutdown window so a
            # rolling deploy never forces redelivery of a turn that the pod
            # is still legitimately finishing.
            #
            # Deploy note: RabbitMQ (verified on 4.1.4) does NOT strictly
            # compare ``x-consumer-timeout`` on queue redeclaration, so this
            # value can change between deploys without triggering
            # PRECONDITION_FAILED. To update the *effective* timeout on an
            # already-running queue before the new code deploys (so pods
            # mid-shutdown don't have their consumer cancelled at the old
            # limit), apply a policy:
            #
            #     rabbitmqctl set_policy copilot-consumer-timeout \
            #       "^copilot_execution_queue_v2$" \
            #       '{"consumer-timeout": 21600000}' \
            #       --apply-to queues
            #
            # The policy takes effect immediately. Once the policy is set
            # to match the code's value the policy is redundant for new
            # pods and can be removed after a stable deploy if desired —
            # but it's harmless to leave in place.
            "x-consumer-timeout": COPILOT_CONSUMER_TIMEOUT_SECONDS * 1000,
        },
    )
    cancel_queue = Queue(
        name=COPILOT_CANCEL_QUEUE_NAME,
        exchange=COPILOT_CANCEL_EXCHANGE,
        routing_key="",  # not used for FANOUT
        durable=True,
        auto_delete=False,
        arguments={"x-queue-type": "quorum"},
    )
    return RabbitMQConfig(
        vhost="/",
        exchanges=[COPILOT_EXECUTION_EXCHANGE, COPILOT_CANCEL_EXCHANGE],
        queues=[run_queue, cancel_queue],
    )


# ============ Message Models ============ #


class CoPilotExecutionEntry(BaseModel):
    """Task payload for CoPilot AI generation.

    This model represents a chat generation task to be processed by the executor.
    """

    session_id: str
    """Chat session ID (also used for dedup/locking)"""

    turn_id: str = ""
    """Per-turn UUID for Redis stream isolation"""

    user_id: str | None
    """User ID (may be None for anonymous users)"""

    message: str
    """User's message to process"""

    is_user_message: bool = True
    """Whether the message is from the user (vs system/assistant)"""

    context: dict[str, str] | None = None
    """Optional context for the message (e.g., {url: str, content: str})"""

    file_ids: list[str] | None = None
    """Workspace file IDs attached to the user's message"""

    organization_id: str | None = None
    """Active organization for tenant-scoped execution"""

    team_id: str | None = None
    """Active workspace for tenant-scoped execution"""

    model: CopilotLLMModel | None = None
    """Per-request model tier: 'standard' or 'advanced'. None = server default."""

    llm_auth_provider: CopilotLlmAuthProvider = "platform"
    """Session-selected model authentication and execution route."""

    llm_credential_id: str | None = None
    """Opaque user-owned credential identifier for a Codex route."""

    permissions: CopilotPermissions | None = None
    """Capability filter inherited from a parent run (e.g. ``run_sub_session``
    forwards its parent's permissions so the sub can't escalate). ``None``
    means the worker applies no filter."""

    envelope: TurnEnvelope | None = None
    """The turn's tree envelope (depth, tool ceiling, taint, deadline), derived
    at dispatch from the spawning turn's. ``None`` only for entries queued
    before the field existed."""

    request_arrival_at: float = 0.0
    """Unix-epoch seconds (server clock) when the originating HTTP
    ``/stream`` request arrived.  The executor's turn-start drain uses
    this to decide whether each pending message was typed BEFORE or AFTER
    the turn's ``current`` message, and orders the combined user bubble
    chronologically.  Defaults to ``0.0`` for backward compatibility with
    queue messages written before this field existed (they sort as "all
    pending before current" — the pre-fix behaviour)."""


class CancelCoPilotEvent(BaseModel):
    """Event to cancel a CoPilot operation."""

    session_id: str
    """Session ID to cancel"""


# ============ Queue Publishing Helpers ============ #


async def enqueue_copilot_turn(
    session_id: str,
    user_id: str | None,
    message: str,
    turn_id: str,
    is_user_message: bool = True,
    context: dict[str, str] | None = None,
    file_ids: list[str] | None = None,
    organization_id: str | None = None,
    team_id: str | None = None,
    model: CopilotLLMModel | None = None,
    llm_auth_provider: CopilotLlmAuthProvider = "platform",
    llm_credential_id: str | None = None,
    permissions: CopilotPermissions | None = None,
    request_arrival_at: float = 0.0,
    *,
    envelope: TurnEnvelope,
) -> None:
    """Enqueue a CoPilot task for processing by the executor service.

    ``envelope`` is required and keyword-only on purpose. Every turn belongs to
    a tree, and a turn that arrives without one is unenforced end to end: the
    ``BaseTool.execute`` check no-ops and its first spawn is minted as a fresh
    root. Making the parameter mandatory means a new entry point has to decide
    which tree its turn belongs to rather than silently opting out.
    ``CoPilotExecutionEntry.envelope`` stays optional so a message enqueued by
    an older worker still decodes.

    Args:
        session_id: Chat session ID (also used for dedup/locking)
        user_id: User ID (may be None for anonymous users)
        message: User's message to process
        turn_id: Per-turn UUID for Redis stream isolation
        is_user_message: Whether the message is from the user (vs system/assistant)
        context: Optional context for the message (e.g., {url: str, content: str})
        file_ids: Optional workspace file IDs attached to the user's message
        mode: Autopilot mode override ('fast' or 'extended_thinking'). None = server default.
        model: Per-request model tier ('standard' or 'advanced'). None = server default.
        permissions: Capability filter inherited from a parent run (sub-AutoPilot).
            None = no filter.
    """
    from backend.util.clients import get_async_copilot_queue

    entry = CoPilotExecutionEntry(
        session_id=session_id,
        turn_id=turn_id,
        user_id=user_id,
        message=message,
        is_user_message=is_user_message,
        context=context,
        file_ids=file_ids,
        organization_id=organization_id,
        team_id=team_id,
        model=model,
        llm_auth_provider=llm_auth_provider,
        llm_credential_id=llm_credential_id,
        permissions=permissions,
        request_arrival_at=request_arrival_at,
        envelope=envelope,
    )

    queue_client = await get_async_copilot_queue()
    await queue_client.publish_message(
        routing_key=COPILOT_EXECUTION_ROUTING_KEY,
        message=entry.model_dump_json(),
        exchange=COPILOT_EXECUTION_EXCHANGE,
    )


async def schedule_turn(
    *,
    session_id: str,
    user_id: str | None,
    turn_id: str,
    message: str,
    tool_call_id: str = "chat_stream",
    tool_name: str = "chat",
    is_user_message: bool = True,
    context: dict[str, str] | None = None,
    file_ids: list[str] | None = None,
    organization_id: str | None = None,
    team_id: str | None = None,
    model: CopilotLLMModel | None = None,
    llm_auth_provider: CopilotLlmAuthProvider = "platform",
    llm_credential_id: str | None = None,
    permissions: CopilotPermissions | None = None,
    request_arrival_at: float = 0.0,
    spawn: SpawnRequest | None = None,
    spawner_envelope: TurnEnvelope | None = None,
) -> None:
    """End-to-end "start a copilot turn": reserve a per-user concurrency
    slot, register the session in the stream registry, then publish the
    work to the executor queue.

    Convenience wrapper for callers that don't need to interleave any
    work between slot acquisition and dispatch (no message persistence,
    no duplicate-detection, etc.). Used by ``run_sub_session`` and
    ``AutoPilotBlock`` via ``session_waiter.run_copilot_turn_via_queue``.

    The HTTP chat route uses :func:`acquire_turn_slot` + :func:`dispatch_turn`
    directly because it must save the user message *inside* the slot
    context (so a 429 cannot leave a ghost message in chat history) and
    skip dispatch entirely when the message is a duplicate.

    Atomicity guarantees:

    * If the user is at the configured concurrent-turn cap,
      :class:`ConcurrentTurnLimitError` is raised before any side-effects
      (no stream registry write, no queue publish). Caller maps to HTTP 429.
    * If ``create_session`` or ``publish_message`` raises, the slot is
      auto-released by the :func:`acquire_turn_slot` context manager —
      a RabbitMQ blip cannot leak a slot until the stale-cutoff sweep.
    * On success the slot is *kept*: ownership transfers to
      ``mark_session_completed``, which releases it when the turn ends.

    Capacity is the *inflight* cap (default 15) rather than the running
    cap (default 5): non-HTTP callers (``run_sub_session``,
    ``AutoPilotBlock``) have no FIFO queue fallback, so applying the
    soft running cap here would regress concurrency below the prior
    SECRT-2335 hotfix behaviour.

    The user's queued turns from the chat HTTP route DO count against
    the inflight cap — pre-check ``running + queued`` here so a user
    with 5 running + 10 queued can't slip a 16th task through this
    path; ``acquire_turn_slot`` only sees the running pool.
    """
    if user_id:
        from backend.copilot.turn_queue import count_inflight_turns

        inflight_cap = get_inflight_turn_limit()
        if await count_inflight_turns(user_id) >= inflight_cap:
            raise ConcurrentTurnLimitError(inflight_turn_limit_message(inflight_cap))

    async with acquire_turn_slot(
        user_id, session_id, capacity=get_inflight_turn_limit()
    ) as slot:
        await dispatch_turn(
            slot,
            session_id=session_id,
            user_id=user_id,
            turn_id=turn_id,
            message=message,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            is_user_message=is_user_message,
            context=context,
            file_ids=file_ids,
            organization_id=organization_id,
            team_id=team_id,
            model=model,
            llm_auth_provider=llm_auth_provider,
            llm_credential_id=llm_credential_id,
            permissions=permissions,
            request_arrival_at=request_arrival_at,
            spawn=spawn,
            spawner_envelope=spawner_envelope,
        )


async def dispatch_turn(
    slot: TurnSlot,
    *,
    session_id: str,
    user_id: str | None,
    turn_id: str,
    message: str,
    tool_call_id: str = "chat_stream",
    tool_name: str = "chat",
    is_user_message: bool = True,
    context: dict[str, str] | None = None,
    file_ids: list[str] | None = None,
    organization_id: str | None = None,
    team_id: str | None = None,
    model: CopilotLLMModel | None = None,
    llm_auth_provider: CopilotLlmAuthProvider = "platform",
    llm_credential_id: str | None = None,
    permissions: CopilotPermissions | None = None,
    request_arrival_at: float = 0.0,
    spawn: SpawnRequest | None = None,
    spawner_envelope: TurnEnvelope | None = None,
) -> None:
    """Within an already-held turn slot, register the session in the
    stream registry, publish the work to the executor queue, and
    transfer slot ownership to ``mark_session_completed``.

    This is the one chokepoint every turn passes — HTTP chat, scheduler,
    ``AutoPilotBlock``, and the three spawn tools — so it is where the
    turn's tree envelope is derived and admitted. A spawned turn that the
    tree refuses raises :class:`TreeRefusal` before any side effect.

    Caller is responsible for acquiring ``slot`` via
    :func:`acquire_turn_slot`. This function is the post-acquire dispatch
    glue — extracted so the HTTP chat route (which needs to save the
    user message between acquire and dispatch) can share the same
    create-session + enqueue + keep sequence as :func:`schedule_turn`'s
    callers without inlining it.

    Failure semantics: if ``create_session`` or the queue publish raises,
    ``slot.keep()`` is never reached, so the context manager that owns
    ``slot`` releases it on exit — no leak.
    """
    # Local import: stream_registry imports executor.utils (the
    # COPILOT_CONSUMER_TIMEOUT_SECONDS constant) → top-level circular.
    from backend.copilot import stream_registry

    envelope = await _admitted_turn_envelope(
        turn_id, user_id, permissions, spawn, spawner_envelope
    )

    # Everything after the admit above runs inside the try: the tree's node
    # counter is already incremented, so an exception from ``create_session``
    # (a Redis blip) would otherwise leak a node for the key's whole TTL, and
    # a handful of those exhaust a tree that never ran anything.
    #
    # Once ``create_session`` has written Redis meta, EVERY exit path
    # from this point on must either (a) commit the turn (``slot.keep()``
    # + RabbitMQ message enqueued) or (b) tear the Redis meta down — or
    # a future read will see ``status='running'`` with no consumer.
    # ``finally`` catches CancelledError as well (which ``except
    # Exception`` would miss); the ``committed`` flag distinguishes the
    # happy path from any failure / cancellation.
    committed = False
    try:
        # Inside the try on purpose: the admit above already incremented the
        # tree's node count, so anything that can raise between there and the
        # finally must be covered by ``release_turn``.
        permissions = _narrow_permissions(permissions, envelope)
        await stream_registry.create_session(
            session_id=session_id,
            user_id=user_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            turn_id=turn_id,
        )
        await enqueue_copilot_turn(
            session_id=session_id,
            user_id=user_id,
            message=message,
            turn_id=turn_id,
            is_user_message=is_user_message,
            context=context,
            file_ids=file_ids,
            organization_id=organization_id,
            team_id=team_id,
            model=model,
            llm_auth_provider=llm_auth_provider,
            llm_credential_id=llm_credential_id,
            permissions=permissions,
            request_arrival_at=request_arrival_at,
            envelope=envelope,
        )
        slot.keep()
        committed = True
    finally:
        if not committed:
            await release_turn(envelope)
            try:
                await stream_registry.delete_session_meta(session_id)
            except BaseException:
                # Already in a failure path — log + swallow so the
                # original exception/cancellation isn't masked.
                logger.exception(
                    "dispatch_turn: redis meta cleanup failed for session=%s",
                    session_id,
                )


async def _admitted_turn_envelope(
    turn_id: str,
    user_id: str | None,
    permissions: CopilotPermissions | None,
    spawn: SpawnRequest | None,
    spawner_envelope: TurnEnvelope | None = None,
) -> TurnEnvelope:
    """Derive this turn's envelope from the running turn's (a child) or mint
    a root, then admit it against the tree ledger.

    The spawner's envelope comes from the executor contextvar, so a caller
    outside any turn — the HTTP route, the scheduler, a graph block — is a
    root by construction rather than by declaration.
    """
    # An explicit spawner wins over the contextvar: a turn started from the
    # graph executor is in a different process, so the contextvar is empty
    # there even though the turn genuinely has a parent (AutoPilotBlock
    # rebuilds it from the execution context).
    spawner = spawner_envelope or get_current_envelope()
    if spawn is not None and spawner is None:
        # A caller that passed a SpawnRequest is by construction a spawn tool
        # running inside a turn, so a missing spawner envelope means the
        # context was lost — a pre-deploy queue entry, or a refactor that
        # dropped it. Minting a root there would hand the child FULL
        # authority, which is the opposite of what the caller asked for, so
        # refuse instead. This also makes the stream_heartbeat task boundary
        # belt-and-braces rather than load-bearing.
        raise TreeRefusal(
            "This task's context was lost, so its limits cannot be carried "
            "over. Start it again from the top."
        )
    if spawner is None:
        envelope = root_envelope(turn_id)
    else:
        envelope = derive_child_envelope(
            spawner, spawn or SpawnRequest(), spawner_permissions=permissions
        )
    await admit_turn(envelope, user_id=user_id)
    return envelope


def _narrow_permissions(
    permissions: CopilotPermissions | None, envelope: TurnEnvelope
) -> CopilotPermissions | None:
    """The envelope's tool set as the turn's whitelist, keeping any block
    filter the caller passed. Hides the tools from the model; the refusal
    itself lives in ``BaseTool.execute``."""
    narrowed = envelope.as_permissions()
    if narrowed is None:
        return permissions
    if permissions is None:
        return narrowed
    # The caller's ``_parent`` is dropped on purpose: it belongs to the
    # spawner's turn, and the envelope is already the narrower bound.
    #
    # ``tools_exclude`` is carried, never hardcoded: an empty envelope encodes
    # deny-all as a blacklist of every tool, and forcing ``False`` here would
    # reread that as a whitelist of every tool — the exact inversion the
    # encoding exists to prevent.
    return CopilotPermissions(
        tools=narrowed.tools,
        tools_exclude=narrowed.tools_exclude,
        blocks=permissions.blocks,
        blocks_exclude=permissions.blocks_exclude,
    )


async def schedule_chat_turn(
    *,
    session_id: str,
    user_id: str,
    message: str,
    message_id: str | None = None,
    message_metadata: dict[str, Any] | None = None,
    message_already_persisted: bool = False,
    is_user_message: bool = True,
    context: dict[str, str] | None = None,
    file_ids: list[str] | None = None,
    organization_id: str | None = None,
    team_id: str | None = None,
    model: CopilotLLMModel | None = None,
    llm_auth_provider: CopilotLlmAuthProvider = "platform",
    llm_credential_id: str | None = None,
    permissions: CopilotPermissions | None = None,
    request_arrival_at: float = 0.0,
) -> str | None:
    """HTTP-chat-flavoured :func:`schedule_turn`: acquires the user's
    concurrency slot, persists the user message *inside* the slot, then
    dispatches the turn — atomically, in that order.

    Returns the new ``turn_id`` on a fresh dispatch, or ``None`` if the
    inbound message was a duplicate of one already saved (caller should
    subscribe to the existing in-flight turn's stream instead of opening
    a new one). ``message_already_persisted`` re-dispatches an orphaned
    kickoff row only when this caller atomically admits the idle session.

    Raises :class:`backend.copilot.active_turns.ConcurrentTurnLimitError`
    when the user is at the configured cap. Caller maps that to HTTP 429.

    Acquire-before-persist is intentional: a 429 must not leave a ghost
    user-message in chat history with no answer ever scheduled. The
    re-acquire path of :func:`acquire_turn_slot` keeps same-session
    network retries safe — the existing slot's score is bumped, no new
    slot is admitted, and exiting without :meth:`TurnSlot.keep` does not
    release the slot held by the original turn.
    """
    # Deferred so the executor module stays a leaf for the queue dataclasses
    # (only the chat HTTP path persists user messages this way).
    from uuid import uuid4

    from backend.copilot.model import ChatMessage, append_and_save_message
    from backend.copilot.tracking import track_user_message

    async with acquire_turn_slot(user_id, session_id) as slot:
        if message_already_persisted and not slot.admitted:
            return None
        is_duplicate = False
        if message and not message_already_persisted:
            chat_message = ChatMessage(
                id=message_id,
                role="user" if is_user_message else "assistant",
                content=message,
                metadata=message_metadata,
            )
            is_duplicate = (
                await append_and_save_message(session_id, chat_message)
            ) is None
            if not is_duplicate and is_user_message:
                track_user_message(
                    user_id=user_id,
                    session_id=session_id,
                    message_length=len(message),
                )

        if is_duplicate:
            return None

        turn_id = str(uuid4())
        await dispatch_turn(
            slot,
            session_id=session_id,
            user_id=user_id,
            turn_id=turn_id,
            message=message,
            is_user_message=is_user_message,
            context=context,
            file_ids=file_ids,
            organization_id=organization_id,
            team_id=team_id,
            model=model,
            llm_auth_provider=llm_auth_provider,
            llm_credential_id=llm_credential_id,
            permissions=permissions,
            request_arrival_at=request_arrival_at,
        )
        return turn_id


async def enqueue_cancel_task(session_id: str) -> None:
    """Publish a cancel request for a running CoPilot session.

    Sends a ``CancelCoPilotEvent`` to the FANOUT exchange so all executor
    pods receive the cancellation signal.
    """
    from backend.util.clients import get_async_copilot_queue

    event = CancelCoPilotEvent(session_id=session_id)
    queue_client = await get_async_copilot_queue()
    await queue_client.publish_message(
        routing_key="",  # FANOUT ignores routing key
        message=event.model_dump_json(),
        exchange=COPILOT_CANCEL_EXCHANGE,
    )
