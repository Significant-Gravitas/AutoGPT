"""The scheduler side of a routine: its jobs, and what a firing turn writes back.

Split from ``routines``, which owns the record and what its owner does to it.
Everything here is the other half — one APScheduler job per fire time, the
Jenkins-style ``H`` minute resolved before anything reaches a trigger, and the
handful of writes a running job makes to the row behind it (the thread it
minted, the one-shot it spent, the job it lost).
"""

import hashlib
import logging
from datetime import datetime, timezone

import prisma.models
import prisma.types

from backend.util.clients import get_scheduler_client

logger = logging.getLogger(__name__)

# Jenkins's spelling for "some minute inside this hour, consistently" —
# a thing plain cron has no syntax for. Resolved to a real minute at
# install time by ``spread_cron``; never persisted or handed to APScheduler.
_SPREAD_MINUTE = "H"


async def pause_routines_for_archive(user_id: str, expert_id: str) -> None:
    """Stop an archived expert's routines from waking the scheduler forever.

    The fire path already refuses an archived expert, but a recurring job that
    is merely refused keeps its next run time, so it re-arms on every tick and
    stays listed as a scheduled chat for an expert the owner has let go.

    Only routines this call actually pauses are marked, so reviving restores
    exactly them — one the owner had switched off themselves must not come
    back running.
    """
    scheduler = get_scheduler_client()
    for row in await _enabled_routines(expert_id):
        paused_any = False
        for schedule_id in row.scheduleIds:
            try:
                paused_any |= await scheduler.pause_schedule(
                    schedule_id, user_id=user_id
                )
            except Exception as e:
                logger.warning(
                    f"Failed to pause routine schedule #{schedule_id} while "
                    f"detaching expert #{expert_id}: {type(e).__name__}: {e}"
                )
        if paused_any:
            await prisma.models.ExpertRoutine.prisma().update(
                where={"id": row.id}, data={"pausedByExpertArchive": True}
            )


async def resume_routines_after_revive(user_id: str, expert_id: str) -> None:
    """Reverse of ``pause_routines_for_archive``, for re-hire.

    Scoped to rows archiving paused. APScheduler recomputes the next fire from
    the trigger, so a long-archived expert picks up at her next cadence instead
    of replaying every run she missed.
    """
    scheduler = get_scheduler_client()
    rows = await prisma.models.ExpertRoutine.prisma().find_many(
        where={"expertId": expert_id, "pausedByExpertArchive": True}
    )
    for row in rows:
        resumed_all = True
        for schedule_id in row.scheduleIds:
            try:
                await scheduler.resume_schedule(schedule_id, user_id=user_id)
            except Exception as e:
                resumed_all = False
                logger.warning(
                    f"Failed to resume routine schedule #{schedule_id} while "
                    f"reviving expert #{expert_id}: {type(e).__name__}: {e}"
                )
        if not resumed_all:
            # The marker is the only record that archiving is what stopped
            # these. Clearing it after a failed resume strands the schedule
            # paused with nothing left to say it should not be — and the next
            # re-hire, which is the natural retry, would skip the row.
            continue
        await prisma.models.ExpertRoutine.prisma().update(
            where={"id": row.id}, data={"pausedByExpertArchive": False}
        )


async def record_routine_thread(routine_id: str, session_id: str) -> None:
    """Remember the thread a THREAD routine minted on its first fire.

    Guarded so the first fire to get there wins: two ticks racing must not
    leave the routine pointing at a thread the other one is writing to.
    """
    await prisma.models.ExpertRoutine.prisma().update_many(
        where={"id": routine_id, "sessionId": None},
        data={"sessionId": session_id},
    )


async def record_routine_fired(routine_id: str) -> None:
    """Mark a one-shot spent, once its turn has been dispatched.

    APScheduler drops the job itself, so without this the row would keep
    saying it is scheduled for a time that has passed. Recurring routines are
    left alone: their next run is the point of them.
    """
    await prisma.models.ExpertRoutine.prisma().update_many(
        where={"id": routine_id, "firedAt": None, "NOT": [{"runAt": None}]},
        data={"firedAt": datetime.now(timezone.utc)},
    )


async def mark_routine_unscheduled(routine_id: str, schedule_id: str) -> None:
    """Drop a job the fire path removed, and switch the routine off if it was
    the last one.

    The scheduler self-deletes a job whose target chat is gone or whose scope
    no longer matches. Without this the row keeps saying it is on while nothing
    behind it can fire — a state the owner cannot act on, because switching it
    off is the only thing offered and it is already off in every sense that
    matters.

    Left enabled when other jobs survive: a routine that fires at 08:30 and
    13:00 and lost one of them is still running, and a reader has the remaining
    ``scheduleIds`` to say so.
    """
    row = await prisma.models.ExpertRoutine.prisma().find_unique(
        where={"id": routine_id}
    )
    if row is None or schedule_id not in row.scheduleIds:
        return
    remaining = [sid for sid in row.scheduleIds if sid != schedule_id]
    data: prisma.types.ExpertRoutineUpdateInput = {"scheduleIds": remaining}
    if not remaining:
        data["enabledAt"] = None
    await prisma.models.ExpertRoutine.prisma().update(
        where={"id": routine_id}, data=data
    )


def spread_cron(cron: str, *, seed: str) -> str:
    """Resolve a Jenkins-style ``H`` minute to a concrete one.

    Five different experts all want "Monday at 9". Someone who hires three of
    them has three routines waking in the same minute, against a cap on how many
    chat turns can run at once — and the ones that miss out do not fail loudly,
    they simply never happen, so the owner's Monday briefing is just absent with
    nothing on screen to explain it.

    Plain cron cannot express "some minute in this hour": ``*`` means all sixty.
    So a roster cron says ``H 9 * * 1`` — borrowing Jenkins's ``H`` — and this
    picks the minute from who the owner is and which routine it is. The same
    routine for the same person lands on the same minute every week, and two
    accounts' morning sweeps almost never share one.

    It spreads; it does not guarantee. A digest byte modulo 60 collides about
    once in sixty for any given pair, so two of one owner's routines can still
    land together — this turns a certainty into a small chance, which for five
    personas that all literally say "9am" is the whole of the win. Making it a
    guarantee needs allocation against what the owner already has, and the
    firing that loses a collision needs queueing rather than dropping; both are
    worth doing and neither is this function.

    Only ``H`` is resolved. A cron that names a minute means that minute, from a
    roster author who wrote 07:40 on purpose as much as from an owner who asked
    for 10am — the implicit version of this used to move both.
    """
    fields = cron.split()
    if len(fields) != 5 or fields[0] != _SPREAD_MINUTE:
        return cron
    # hashlib, not hash(): Python salts the builtin per process, which would
    # give the same routine a different minute after every deploy.
    digest = hashlib.sha256(seed.encode()).digest()
    fields[0] = str(digest[0] % 60)
    return " ".join(fields)


async def create_routine_schedules(
    *,
    user_id: str,
    expert_id: str | None,
    row: prisma.models.ExpertRoutine,
    prompt: str,
    crons: list[str],
    run_at: datetime | None,
    session_id: str | None,
    user_timezone: str,
) -> list[str]:
    scheduler = get_scheduler_client()
    created: list[str] = []
    # A one-shot is one job with no cron; a cadence is one job per cron. Both
    # carry ``routine_id``, which is what tells the fire path there is a row
    # behind this turn deciding where it lands and how far it can reach.
    plan: list[tuple[str | None, datetime | None]] = (
        [(None, run_at)] if run_at is not None else [(cron, None) for cron in crons]
    )
    try:
        for cron, at in plan:
            info = await scheduler.add_copilot_turn_schedule(
                user_id=user_id,
                session_id=session_id,
                message=prompt,
                cron=cron,
                run_at=at,
                name=row.title,
                user_timezone=user_timezone,
                expert_id=expert_id,
                routine_id=row.id,
            )
            created.append(info.id)
    except Exception:
        # All or nothing: a routine that fires at 08:30 but not at 13:00 is a
        # quieter lie than one that does not fire at all.
        for schedule_id in created:
            try:
                await scheduler.delete_schedule(schedule_id, user_id=user_id)
            except Exception as e:
                logger.warning(
                    f"Leaked routine schedule #{schedule_id} for routine "
                    f"#{row.id} after a failed enable: {type(e).__name__}: {e}"
                )
        raise
    return created


async def delete_routine_schedules(
    user_id: str, row: prisma.models.ExpertRoutine
) -> None:
    scheduler = get_scheduler_client()
    for schedule_id in row.scheduleIds:
        try:
            await scheduler.delete_schedule(schedule_id, user_id=user_id)
        except Exception as e:
            # Best effort, like the preload path. A job we cannot delete
            # fires into a row that has already moved on — switched off, or
            # switched to a new cadence — and the fire-time lookup is what
            # reconciles it.
            logger.warning(
                f"Could not delete routine schedule #{schedule_id}: "
                f"{type(e).__name__}: {e}"
            )


async def _enabled_routines(expert_id: str) -> list[prisma.models.ExpertRoutine]:
    return await prisma.models.ExpertRoutine.prisma().find_many(
        where={"expertId": expert_id, "NOT": [{"enabledAt": None}], "firedAt": None}
    )
