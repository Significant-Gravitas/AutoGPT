"""Stop a closing account's AutoPilot turns and the chat links that start them."""

import asyncio
import logging

from backend.api.features.account_deletion.best_effort import (
    attempt_step,
    log_step_failure,
)
from backend.copilot.active_turns import get_running_session_ids
from backend.copilot.executor.utils import enqueue_cancel_task
from backend.copilot.turn_queue import cancel_queued_turn, list_queued_sessions
from backend.data.db_accessors import platform_linking_db

logger = logging.getLogger(__name__)

# How long to keep cancelling until the user has no running turn: the same 5 s
# the chat's Stop button waits for its turn to end. It holds back the run
# listing that follows, so a turn still winding down can't start a run after it.
TURN_STOP_WAIT_SECONDS = 5.0
TURN_STOP_POLL_SECONDS = 0.5


async def unlink_chat_platforms(user_id: str) -> None:
    """Unlink every chat server and DM the user linked to AutoPilot.

    A message in a linked server (Discord, Slack, Telegram and the other
    chat platforms) or a linked DM starts an AutoPilot turn as the link's
    owner with no login. Closing the account deletes any this fails to
    remove, and the bot treats a closed account's links as unlinked, so none
    runs a turn after the close. Unlinked here first, so no new turn starts
    while the running ones are cancelled. Uses the same
    unlink calls as the platform-linking routes. Best effort and never raises:
    each call that fails is logged and the rest still run.
    """
    try:
        links = platform_linking_db()
    except Exception as e:
        log_step_failure("reach the chat platform links", user_id, e)
        return
    servers, dms = await asyncio.gather(
        attempt_step(
            "list linked chat servers", user_id, links.list_server_links(user_id)
        ),
        attempt_step("list linked chat DMs", user_id, links.list_user_links(user_id)),
    )
    await asyncio.gather(
        *(
            attempt_step(
                f"unlink {link.platform} server link #{link.id}",
                user_id,
                links.delete_server_link(link.id, user_id),
            )
            for link in servers or []
        ),
        *(
            attempt_step(
                f"unlink {link.platform} DM link #{link.id}",
                user_id,
                links.delete_user_link(link.id, user_id),
            )
            for link in dms or []
        ),
    )


async def cancel_autopilot_turns(user_id: str) -> None:
    """Dequeue the user's queued AutoPilot turns, then cancel the running ones.

    Queued turns go first because a running turn's end promotes the user's
    oldest queued turn. Running turns get the cancel the chat's Stop button
    sends, which reaches a turn an executor has picked up. Only turns that
    took a running slot are listed: a chat-platform turn skips the slot, so
    one already running is not cancelled here. Best effort and never raises;
    bounded by STEP_TIMEOUT_SECONDS per call for the queue and
    TURN_STOP_WAIT_SECONDS in all for the running turns.
    """
    await _dequeue_turns(user_id)
    try:
        await asyncio.wait_for(
            _cancel_until_none_run(user_id), timeout=TURN_STOP_WAIT_SECONDS
        )
    except TimeoutError:
        logger.warning(
            f"Account deletion could not confirm within "
            f"{TURN_STOP_WAIT_SECONDS:.0f}s that every AutoPilot turn of user "
            f"{user_id} had ended"
        )
    except Exception as e:
        log_step_failure("list running AutoPilot turns", user_id, e)


async def _dequeue_turns(user_id: str) -> None:
    queued = await attempt_step(
        "list queued AutoPilot turns", user_id, list_queued_sessions(user_id)
    )
    await asyncio.gather(
        *(
            attempt_step(
                f"dequeue the AutoPilot turn in chat #{chat.session_id}",
                user_id,
                cancel_queued_turn(user_id=user_id, session_id=chat.session_id),
            )
            for chat in queued or []
        )
    )


async def _cancel_until_none_run(user_id: str) -> None:
    """Cancel each running turn once, including any that starts meanwhile
    (a promoted queued turn, or one a still-running agent starts), until the
    user has none running."""
    cancelled: set[str] = set()
    while running := await get_running_session_ids(user_id):
        await asyncio.gather(
            *(
                attempt_step(
                    f"cancel the AutoPilot turn in chat #{session_id}",
                    user_id,
                    enqueue_cancel_task(session_id),
                )
                for session_id in sorted(running - cancelled)
            )
        )
        cancelled |= running
        await asyncio.sleep(TURN_STOP_POLL_SECONDS)
