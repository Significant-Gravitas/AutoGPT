"""Standing work an expert does unattended: the lifecycle of ExpertRoutine.

A routine is a prompt on a cadence that fires as a copilot turn.
``schedule_followup`` is the primitive underneath; this module owns the durable
record around it — the cadence, the thread, and whether it is switched on.

Three rules run through everything here:

* **A seeded routine arrives off.** Hiring copies the template's proposals with
  no scheduler job at all, so a routine nobody asked for costs nothing and
  fires nothing. Off is the absence of a job, not a paused one.
* **A seeded routine reaches nothing.** Until its owner says otherwise the
  fire-time turn resolves an empty credential allow-list, so it can read,
  think, and write to its own thread and no further. That is the rule
  ``PreloadSeed.cron`` states for preloads, made enforceable.
* **Switching one on resolves it.** The template ships a proposal; the owner's
  answers become the prompt and the cadence that actually run, and from then on
  the row is theirs.
"""

import hashlib
import logging
from datetime import datetime, timezone

import prisma.enums
import prisma.models
import prisma.types
from apscheduler.triggers.cron import CronTrigger

from backend.api.features.experts.models import ExpertRoutine
from backend.data.user import get_user_by_id
from backend.util.clients import get_scheduler_client
from backend.util.timezone_utils import get_user_timezone_or_utc

logger = logging.getLogger(__name__)


def to_model(row: prisma.models.ExpertRoutine) -> ExpertRoutine:
    return ExpertRoutine(
        id=row.id,
        expert_id=row.expertId,
        key=row.key,
        title=row.title,
        prompt=row.prompt,
        crons=row.crons,
        asks=row.asks,
        session_mode=row.sessionMode.value,
        session_id=row.sessionId,
        enabled=row.enabledAt is not None,
        customized=row.customizedAt is not None,
        grants_credentials=row.grantsCredentials,
    )


async def get_routine(routine_id: str) -> ExpertRoutine | None:
    """Load one routine by id, for the fire path.

    Unscoped by owner on purpose: the caller is the scheduler dispatching a job
    it already persisted, and the job's own ``user_id`` is what every ownership
    check downstream runs against. Every user-facing path goes through
    ``_owned_routine`` instead.
    """
    row = await prisma.models.ExpertRoutine.prisma().find_unique(
        where={"id": routine_id}
    )
    return to_model(row) if row is not None else None


class RoutineNotFoundError(Exception):
    """No such routine on this expert, for this owner."""


class RoutineUnansweredAsksError(Exception):
    """The routine still has questions nobody answered.

    A seeded proposal names what it needs to know — which repo, which inbox,
    what hour. Scheduling it before those are answered would run it against
    guesses on somebody's account, every day, unattended.
    """


async def install_routines(
    expert_id: str, template_routines: list[prisma.models.ExpertRoutine]
) -> None:
    """Copy a template's routine proposals onto a new hire, all switched off.

    Deliberately creates no scheduler jobs: a cadence that fires unattended
    from the day of hire is a much bigger promise than a skill, and nobody has
    made it yet. The rows exist so the expert can offer them on its first turn.
    """
    for routine in template_routines:
        data: prisma.types.ExpertRoutineCreateInput = {
            "expertId": expert_id,
            "key": routine.key,
            "title": routine.title,
            "prompt": routine.prompt,
            "crons": routine.crons,
            "asks": routine.asks,
            "sessionMode": routine.sessionMode,
        }
        try:
            await prisma.models.ExpertRoutine.prisma().create(data=data)
        except Exception:
            # Honest partial hire, same as a failed preload: the expert is
            # still worth having without one of its offers.
            logger.exception(
                f"Failed to install routine {routine.key!r} on expert #{expert_id}"
            )


async def list_routines(user_id: str, expert_id: str) -> list[ExpertRoutine]:
    rows = await prisma.models.ExpertRoutine.prisma().find_many(
        where={"Expert": {"is": {"id": expert_id, "ownerUserId": user_id}}},
        order=[{"createdAt": "asc"}, {"id": "asc"}],
    )
    return [to_model(row) for row in rows]


async def enable_routine(
    user_id: str,
    expert_id: str,
    routine_id: str,
    *,
    prompt: str | None = None,
    crons: list[str] | None = None,
    session_id: str | None = None,
    grants_credentials: bool = False,
) -> ExpertRoutine:
    """Switch a routine on, resolving the proposal into what actually runs.

    *prompt* and *crons* are the owner's answers. Passing either marks the row
    customized, which takes it out of reach of every later roster edit — from
    here on what this routine does is the owner's, not the template's.

    Creates one scheduler job per cron. A partial failure leaves nothing
    behind: a routine that fires at 08:30 but not at 13:00 is a quieter lie
    than one that does not fire at all.
    """
    row = await _owned_routine(user_id, expert_id, routine_id)
    resolved_prompt = prompt or row.prompt
    customized = prompt is not None or crons is not None
    if row.asks and not customized:
        raise RoutineUnansweredAsksError(
            f"'{row.title}' needs answers before it can run: " + "; ".join(row.asks)
        )

    user = await get_user_by_id(user_id)
    user_timezone = get_user_timezone_or_utc(user.timezone if user else None)
    if crons is not None:
        resolved_crons = crons
    else:
        # Only a template's suggested hour gets nudged. A time the owner named
        # is used exactly as given.
        resolved_crons = [
            _spread_cron(cron, seed=f"{user_id}:{row.key or row.id}:{index}")
            for index, cron in enumerate(row.crons)
        ]
    for cron in resolved_crons:
        CronTrigger.from_crontab(cron, timezone=user_timezone)

    schedule_ids = await _create_routine_schedules(
        user_id=user_id,
        expert_id=expert_id,
        row=row,
        prompt=resolved_prompt,
        crons=resolved_crons,
        session_id=session_id,
        user_timezone=user_timezone,
    )
    now = datetime.now(timezone.utc)
    data: prisma.types.ExpertRoutineUpdateInput = {
        "prompt": resolved_prompt,
        "crons": resolved_crons,
        "scheduleIds": schedule_ids,
        "sessionId": session_id,
        "enabledAt": now,
        "grantsCredentials": grants_credentials,
    }
    if customized:
        data["customizedAt"] = now
    updated = await prisma.models.ExpertRoutine.prisma().update(
        where={"id": row.id}, data=data
    )
    if updated is None:
        raise RoutineNotFoundError(routine_id)
    return to_model(updated)


async def disable_routine(
    user_id: str, expert_id: str, routine_id: str
) -> ExpertRoutine:
    """Switch a routine off, deleting its jobs but keeping the row.

    The resolved prompt and cadence survive, so switching it back on later
    restores what the owner set up rather than reverting to the proposal.
    """
    row = await _owned_routine(user_id, expert_id, routine_id)
    await _delete_routine_schedules(user_id, row)
    updated = await prisma.models.ExpertRoutine.prisma().update(
        where={"id": row.id},
        data={
            "scheduleIds": [],
            "enabledAt": None,
            "pausedByExpertArchive": False,
        },
    )
    if updated is None:
        raise RoutineNotFoundError(routine_id)
    return to_model(updated)


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
        for schedule_id in row.scheduleIds:
            try:
                await scheduler.resume_schedule(schedule_id, user_id=user_id)
            except Exception as e:
                logger.warning(
                    f"Failed to resume routine schedule #{schedule_id} while "
                    f"reviving expert #{expert_id}: {type(e).__name__}: {e}"
                )
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


def _spread_cron(cron: str, *, seed: str) -> str:
    """Move a suggested fire time to its own minute inside the same hour.

    Five different experts all say "Monday at 9". Someone who hires three of
    them has three routines waking in the same minute, against a cap on how
    many chat turns can run at once — and the ones that miss out do not fail
    loudly, they simply never happen, so the owner's Monday briefing is just
    absent with nothing on screen to explain it.

    So each routine gets a minute of its own, picked from who the owner is and
    which routine it is. The same routine for the same person lands on the same
    minute every week; two people's morning sweeps land on different minutes;
    two routines on one account never collide with each other.

    Only a plain numeric minute is moved. Anything else — ``*``, a list, a
    step — is an intent we cannot rewrite without changing what it means, so it
    is left exactly as written. A time the owner named never reaches here.
    """
    fields = cron.split()
    if len(fields) != 5 or not fields[0].isdigit():
        return cron
    # hashlib, not hash(): Python salts the builtin per process, which would
    # give the same routine a different minute after every deploy.
    digest = hashlib.sha256(seed.encode()).digest()
    fields[0] = str(digest[0] % 60)
    return " ".join(fields)


async def _create_routine_schedules(
    *,
    user_id: str,
    expert_id: str,
    row: prisma.models.ExpertRoutine,
    prompt: str,
    crons: list[str],
    session_id: str | None,
    user_timezone: str,
) -> list[str]:
    scheduler = get_scheduler_client()
    created: list[str] = []
    try:
        for cron in crons:
            info = await scheduler.add_copilot_turn_schedule(
                user_id=user_id,
                session_id=session_id,
                message=prompt,
                cron=cron,
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
                    f"Leaked routine schedule #{schedule_id} for expert "
                    f"#{expert_id} after a failed enable: {type(e).__name__}: {e}"
                )
        raise
    return created


async def _delete_routine_schedules(
    user_id: str, row: prisma.models.ExpertRoutine
) -> None:
    scheduler = get_scheduler_client()
    for schedule_id in row.scheduleIds:
        try:
            await scheduler.delete_schedule(schedule_id, user_id=user_id)
        except Exception as e:
            # Best effort, like the preload path: a job we cannot delete still
            # fires into a routine row that now says it is off, and the
            # fire-time lookup is what stops it.
            logger.warning(
                f"Could not delete routine schedule #{schedule_id}: "
                f"{type(e).__name__}: {e}"
            )


async def _enabled_routines(expert_id: str) -> list[prisma.models.ExpertRoutine]:
    return await prisma.models.ExpertRoutine.prisma().find_many(
        where={"expertId": expert_id, "NOT": [{"enabledAt": None}]}
    )


async def _owned_routine(
    user_id: str, expert_id: str, routine_id: str
) -> prisma.models.ExpertRoutine:
    row = await prisma.models.ExpertRoutine.prisma().find_first(
        where={
            "id": routine_id,
            "expertId": expert_id,
            # Joined rather than trusting the caller's expert_id: this is the
            # only ownership check between a routine id and somebody else's
            # standing work.
            "Expert": {
                "is": {
                    "ownerUserId": user_id,
                    "isTemplate": False,
                    "isArchived": False,
                }
            },
        }
    )
    if row is None:
        raise RoutineNotFoundError(routine_id)
    return row
