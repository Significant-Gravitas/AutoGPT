"""Stop a closing account's scheduled, triggered and running work."""

import asyncio
import logging

from prisma.models import AgentNode, AgentPreset

from backend.api.features.account_deletion.autopilot_work import (
    cancel_autopilot_turns,
    unlink_chat_platforms,
)
from backend.api.features.account_deletion.best_effort import (
    attempt_step,
    log_step_failure,
)
from backend.data.execution import ExecutionStatus, get_graph_executions
from backend.data.graph import set_node_webhook
from backend.executor.scheduler import SchedulerClient
from backend.executor.utils import stop_graph_execution
from backend.util.clients import get_scheduler_client

logger = logging.getLogger(__name__)

# How long each stop waits for its run to confirm it ended. The cancel goes
# out before the wait, so this bounds the confirmation, not the stop itself:
# queued and incomplete runs are marked terminated on the first poll, and a
# running one that needs longer has already been told to stop. Every stop runs
# at once with cascade off (a child run belongs to the same user, so it is in
# the list itself), so the whole step takes about this long however many runs
# there are and however deeply they nest.
STOP_WAIT_TIMEOUT_SECONDS = 5.0

_UNFINISHED_STATUSES = [
    ExecutionStatus.INCOMPLETE,
    ExecutionStatus.QUEUED,
    ExecutionStatus.RUNNING,
]


async def stop_user_work(user_id: str) -> None:
    """Stop everything that can start or keep doing work for the user.

    In order, so nothing new starts after its kind is listed:
    1. What starts work without a login: every schedule, the webhook
       triggers, and the chat links that start AutoPilot turns.
    2. AutoPilot turns, queued then running (they can start runs).
    3. Unfinished graph runs.

    Residual gaps: a schedule fire or webhook delivery already under way can
    still add a run or turn; a turn dispatched but not yet picked up by an
    executor misses its cancel; a turn already running from a chat platform
    is not cancelled (it never takes a running slot, so it isn't listed; the
    unlink only stops new ones); a run can start an AutoPilot turn (AutoPilot
    block) between steps 2 and 3; until the account is closed, a request
    from the user's own session can start work too; and a schedule or trigger
    whose removal failed is not retried here (the erasure reruns this). Once
    the account is closed, the gates of runs, AutoPilot turns, webhook
    deliveries and memory crons (``is_user_refused``) refuse it anyway, so a
    leftover can't start work.

    Best effort: each call that fails is logged and the rest still run, and
    this never raises, so the account can still be closed. Every call is
    bounded in time: it waits up to 5 s for running AutoPilot turns to end,
    and with every service hung it returns in about a minute.
    """
    if not user_id:
        # An empty id would turn the run listing into "every user's runs".
        logger.error("Account deletion stopped no work: no user id was given")
        return
    await asyncio.gather(
        _remove_schedules(user_id),
        _stop_webhook_triggers(user_id),
        unlink_chat_platforms(user_id),
    )
    await cancel_autopilot_turns(user_id)
    await _stop_unfinished_runs(user_id)


async def _remove_schedules(user_id: str) -> None:
    try:
        scheduler = get_scheduler_client()
    except Exception as e:
        log_step_failure("reach the scheduler", user_id, e)
        return
    await asyncio.gather(
        _remove_listed_schedules(scheduler, user_id),
        attempt_step(
            "remove the morning briefing",
            user_id,
            scheduler.remove_morning_briefing_schedule(user_id=user_id),
        ),
    )
    # The memory crons (nightly batch, community rebuild, skill learning) stay
    # registered: their gate (is_user_refused) skips a closed account, and
    # they resume on their own if it is kept. The erasure removes them.


async def _remove_listed_schedules(scheduler: SchedulerClient, user_id: str) -> None:
    """Graph and copilot-turn schedules, paused ones (a fired expert's) included."""
    schedules = await attempt_step(
        "list schedules",
        user_id,
        scheduler.get_execution_schedules(user_id=user_id, include_paused=True),
    )
    # By job id, not schedule_id: a legacy job has no schedule_id in its args.
    await asyncio.gather(
        *(
            attempt_step(
                f"delete schedule #{schedule.id}",
                user_id,
                scheduler.delete_schedule(schedule_id=schedule.id, user_id=user_id),
            )
            for schedule in schedules or []
        )
    )


async def _stop_webhook_triggers(user_id: str) -> None:
    """Stop the user's webhooks from starting runs.

    A delivery to a webhook's ingress URL needs no login. Disconnecting the
    user's apps removes the webhooks registered with those credentials; this
    also covers the credential-free ones (the generic webhook block, Compass,
    AllQuiet) and any the disconnect missed. The webhooks themselves are left
    for data erasure.
    """
    await asyncio.gather(
        attempt_step("switch off presets", user_id, _deactivate_presets(user_id)),
        _detach_triggered_nodes(user_id),
    )


async def _deactivate_presets(user_id: str) -> int:
    # The switch the expert archive flips: a webhook skips an inactive preset.
    return await AgentPreset.prisma().update_many(
        where={"userId": user_id, "isActive": True},
        data={"isActive": False},
    )


async def _detach_triggered_nodes(user_id: str) -> None:
    """Legacy triggered nodes (migrated to presets at startup, but still run
    by the webhook ingress) have no on/off switch, so they are detached."""
    nodes = await attempt_step(
        "list webhook-triggered nodes",
        user_id,
        AgentNode.prisma().find_many(where={"Webhook": {"is": {"userId": user_id}}}),
    )
    await asyncio.gather(
        *(
            attempt_step(
                f"detach node #{node.id} from its webhook",
                user_id,
                set_node_webhook(node.id, None),
            )
            for node in nodes or []
        )
    )


async def _stop_unfinished_runs(user_id: str) -> None:
    runs = await attempt_step(
        "list unfinished runs",
        user_id,
        get_graph_executions(user_id=user_id, statuses=_UNFINISHED_STATUSES),
    )
    await asyncio.gather(
        *(
            attempt_step(f"stop run #{run.id}", user_id, _stop_run(user_id, run.id))
            for run in runs or []
        )
    )


async def _stop_run(user_id: str, run_id: str) -> None:
    try:
        await stop_graph_execution(
            user_id=user_id,
            graph_exec_id=run_id,
            wait_timeout=STOP_WAIT_TIMEOUT_SECONDS,
            cascade=False,
        )
    except TimeoutError:
        logger.warning(
            f"Account deletion sent a stop to run #{run_id} of user {user_id}, "
            f"which had not stopped after {STOP_WAIT_TIMEOUT_SECONDS:.0f}s"
        )
