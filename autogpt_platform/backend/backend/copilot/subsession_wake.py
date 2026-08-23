"""Wake the delegating chat when a delegated sub-session finishes.

``delegate_to_expert`` records who asked for the work on the sub's session
metadata (``delegated_by_session_id``). Until now nothing consumed that on
the *completion* side: if the teammate finished after the delegating turn
had already closed, the result sat in the sub's transcript until the user
nudged the parent chat.

:func:`schedule_parent_wake` closes that loop. When a sub reaches a terminal
status it posts one system-framed message onto the delegating chat telling
the model to read the result and report it to the user in its own voice.

Three things keep this safe to run from the completion hot path:

* **Detached** — :func:`spawn_background_task`, so a slow or failing wake
  never delays or breaks the sub's own completion.
* **At most once** — the enqueue is claimed with a ``SET NX`` marker keyed
  on ``(parent, sub, terminal status)``. Unlike the briefing dedupe there is
  no message row to hang a PK on (we enqueue a *turn*, not a message), so
  the Redis claim is the atomic primitive. It is taken *before* the enqueue:
  a dropped wake is recoverable by the model's own polling, a double-posted
  one is not.
* **Never double-reports** — a turn that is still blocked inside
  ``wait_for_session_result`` for this sub will surface the result through
  its own tool call. That wait registers an :func:`inline_wait_key` lease
  for the duration, and a live lease suppresses the wake.
"""

import logging
from typing import Literal

from backend.copilot.model import get_chat_session_metadata
from backend.data.redis_client import get_redis_async
from backend.util.background import spawn_background_task
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

TerminalStatus = Literal["completed", "failed"]

# Claim marker for "this (parent, sub, status) wake has been enqueued".
# The TTL only has to outlive retries and pod restarts for the same
# completion, not the session — a day is generous.
_WAKE_CLAIM_PREFIX = "copilot:subsession_wake:"
_WAKE_CLAIM_TTL_SECONDS = 24 * 60 * 60

# Lease marking "some turn is currently blocked waiting on this session's
# terminal event". Written by ``wait_for_session_result``.
_INLINE_WAIT_PREFIX = "copilot:subsession_wait:"
_INLINE_WAIT_SLACK_SECONDS = 15


def inline_wait_key(session_id: str) -> str:
    return f"{_INLINE_WAIT_PREFIX}{session_id}"


def schedule_parent_wake(sub_session_id: str, status: TerminalStatus) -> None:
    """Fire-and-forget the parent-chat wake for a just-finished session."""
    spawn_background_task(
        _wake_parent(sub_session_id, status),
        name=f"subsession-wake-{sub_session_id[:12]}",
    )


async def _wake_parent(sub_session_id: str, status: TerminalStatus) -> None:
    """Post the delegated-task-finished turn onto the delegating chat.

    Every guard below is a silent no-op rather than an error: the sub has
    already finished successfully by the time we run, and nothing here is
    worth failing that.
    """
    try:
        sub = await get_chat_session_metadata(sub_session_id)
        if sub is None:
            return

        parent_session_id = sub.metadata.delegated_by_session_id
        if not parent_session_id:
            return

        # A handoff writes the delegation fields too, but transfers
        # *ownership*: the receiving expert reports to the user directly and
        # the handing-off session cannot even poll the sub
        # (``get_sub_session_result._in_caller_scope``). Waking it would
        # report work that is no longer its own.
        if sub.metadata.handed_off_from_expert_id is not None:
            return

        if not await is_feature_enabled(
            Flag.COPILOT_SUBSESSION_WAKE, sub.user_id, default=False
        ):
            return

        if await _is_awaited_inline(sub_session_id):
            logger.debug(
                "subsession wake skipped for sub=%s: a turn is still waiting "
                "on it inline",
                sub_session_id[:12],
            )
            return

        parent = await get_chat_session_metadata(parent_session_id, sub.user_id)
        if parent is None:
            return

        if not await _claim_wake(parent_session_id, sub_session_id, status):
            return

        # Local import: ``stream_registry`` imports this module for the
        # completion hook, and ``session_waiter`` imports ``stream_registry``.
        from backend.copilot.sdk.session_waiter import run_copilot_turn_via_queue

        outcome, _ = await run_copilot_turn_via_queue(
            session_id=parent_session_id,
            user_id=parent.user_id,
            message=wake_message(sub_session_id=sub_session_id, status=status),
            # 0 = don't wait: an idle parent gets a fresh turn dispatched, a
            # busy one gets the message on its pending buffer. Either way we
            # return without occupying this worker.
            timeout=0,
            tool_call_id=f"subwake:{sub_session_id}",
            tool_name="delegated_task_completed",
        )
        logger.info(
            "subsession wake enqueued on parent=%s for sub=%s (status=%s, "
            "outcome=%s)",
            parent_session_id[:12],
            sub_session_id[:12],
            status,
            outcome,
        )
    except Exception:
        logger.warning(
            "subsession wake failed for sub=%s; dropping",
            sub_session_id[:12],
            exc_info=True,
        )


def wake_message(*, sub_session_id: str, status: TerminalStatus) -> str:
    """The system-framed prompt the delegating model wakes up to.

    There is no author field on a pending message — everything buffered is
    presented to the model as the user — so the framing is the only thing
    that stops the model from answering this line as if the user typed it.
    Same convention as ``delegate_to_expert._handoff_message``.
    """
    return (
        "[System notice, not the user speaking: a task you delegated has "
        "finished. Do not reply to this notice itself.]\n\n"
        f'<delegated_task_completed sub_session_id="{sub_session_id}" '
        f'status="{status}" />\n\n'
        "Call `get_sub_session_result` with that sub_session_id to read what "
        "came back. It returns the `success_criteria` this delegation was "
        "given — check the result against every one of them before you call "
        "it done. Then report the outcome to the user in your own voice: what "
        "you had asked for, what you got, and what happens next. If it failed "
        "or left any criterion unmet, name that criterion plainly and take the "
        "next step yourself rather than handing the chase back to the user."
    )


async def _claim_wake(
    parent_session_id: str, sub_session_id: str, status: TerminalStatus
) -> bool:
    """Claim the right to enqueue this wake exactly once."""
    redis = await get_redis_async()
    key = f"{_WAKE_CLAIM_PREFIX}{parent_session_id}:{sub_session_id}:{status}"
    return bool(await redis.set(key, "1", nx=True, ex=_WAKE_CLAIM_TTL_SECONDS))


async def _is_awaited_inline(session_id: str) -> bool:
    redis = await get_redis_async()
    return bool(await redis.get(inline_wait_key(session_id)))


async def mark_awaited_inline(session_id: str, timeout: float) -> None:
    """Lease "a turn is blocked on this session" for the length of the wait."""
    try:
        redis = await get_redis_async()
        await redis.setex(
            inline_wait_key(session_id),
            int(timeout) + _INLINE_WAIT_SLACK_SECONDS,
            "1",
        )
    except Exception:
        logger.debug(
            "could not register inline-wait lease for session=%s",
            session_id[:12],
            exc_info=True,
        )


async def clear_awaited_inline(session_id: str) -> None:
    try:
        redis = await get_redis_async()
        await redis.delete(inline_wait_key(session_id))
    except Exception:
        logger.debug(
            "could not clear inline-wait lease for session=%s",
            session_id[:12],
            exc_info=True,
        )
