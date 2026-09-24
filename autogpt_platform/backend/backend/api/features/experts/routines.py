"""Standing work done unattended: the lifecycle of ExpertRoutine.

A routine is a prompt on a cadence that fires as a copilot turn.
``schedule_followup`` is the primitive underneath; this module owns the durable
record around it — the cadence, the thread, how far it may reach, and whether
it is switched on. An expert owns routines, and so does the account itself:
Otto is the platform's default assistant rather than a row in ``Expert``, so
its standing work hangs off the owner directly.

Four rules run through everything here:

* **A seeded routine arrives off.** Hiring copies the template's proposals with
  no scheduler job at all, so a routine nobody asked for costs nothing and
  fires nothing. Off is the absence of a job, not a paused one.
* **A seeded routine reaches nothing.** Until its owner says otherwise, a
  ``TEMPLATE`` routine's fire-time turn is refused every tool that carries a
  credential outward, so it can read, think, and write to its own thread and no
  further. That is the rule ``PreloadSeed.cron`` states for preloads, made
  enforceable.
* **An owner's own routine is not a template.** A routine dictated in the
  owner's chat has no third party in its prompt, so it runs with the reach
  ``schedule_followup`` has always had. Muting it would buy nothing: the same
  words typed into the same chat already run unmuted.
* **Switching one on resolves it.** The template ships a proposal; the owner's
  answers become the prompt and the cadence that actually run, and from then on
  the row is theirs.

The scheduler side — the jobs themselves, the ``H`` minute, and the writes a
firing turn makes back to the row — lives in ``routine_jobs``.
"""

import logging
from datetime import datetime, timezone

import prisma.enums
import prisma.models
import prisma.types
from apscheduler.triggers.cron import CronTrigger

from backend.api.features.experts.models import ExpertRoutine
from backend.api.features.experts.routine_jobs import (
    create_routine_schedules,
    delete_routine_schedules,
    spread_cron,
)
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
        run_at=row.runAt,
        asks=row.asks,
        # ``str``, not ``.value``: the generated type says StrEnum but Prisma
        # hydrates a row with a plain string, so ``.value`` type-checks and then
        # raises. StrEnum subclasses str, so this is right for both.
        session_mode=str(row.sessionMode),
        session_id=row.sessionId,
        source=str(row.source),
        # A spent one-shot is not on. Its row outlives its job so it can still
        # be listed and counted, and without this it would keep describing
        # itself as scheduled long after the only thing it did was happen.
        enabled=row.enabledAt is not None and row.firedAt is None,
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
    """No such routine on this owner."""


class RoutineUnansweredAsksError(Exception):
    """The routine still has questions nobody answered.

    A seeded proposal names what it needs to know — which repo, which inbox,
    what hour. Scheduling it before those are answered would run it against
    guesses on somebody's account, every day, unattended.
    """


async def install_routines(
    expert_id: str, template_routines: list[prisma.models.ExpertRoutine]
) -> list[str]:
    """Copy a template's routine proposals onto a new hire, all switched off.

    Deliberately creates no scheduler jobs: a cadence that fires unattended
    from the day of hire is a much bigger promise than a skill, and nobody has
    made it yet. The rows exist so the expert can offer them on its first turn.
    Returns the titles that failed to install.
    """
    failed: list[str] = []
    for routine in template_routines:
        data: prisma.types.ExpertRoutineCreateInput = {
            "expertId": expert_id,
            "key": routine.key,
            "title": routine.title,
            "prompt": routine.prompt,
            "crons": routine.crons,
            "asks": routine.asks,
            "sessionMode": routine.sessionMode,
            # Spelled out rather than left to the column default: this is the
            # line that decides a hired routine reaches nothing, and a default
            # is a thing somebody edits without reading what it was holding up.
            "source": prisma.enums.ExpertRoutineSource.TEMPLATE,
        }
        try:
            await prisma.models.ExpertRoutine.prisma().create(data=data)
        except Exception:
            # Honest partial hire, same as a failed preload: the expert is
            # still worth having without one of its offers.
            logger.exception(
                f"Failed to install routine {routine.key!r} on expert #{expert_id}"
            )
            failed.append(routine.title)
    return failed


async def list_routines(
    user_id: str, expert_id: str | None = None
) -> list[ExpertRoutine]:
    """One owner's routines: an expert's, or the account's own when *expert_id*
    is None."""
    rows = await prisma.models.ExpertRoutine.prisma().find_many(
        where=_owner_where(user_id, expert_id),
        order=[{"createdAt": "asc"}, {"id": "asc"}],
    )
    return [to_model(row) for row in rows]


async def enable_routine(
    user_id: str,
    expert_id: str | None,
    routine_id: str,
    *,
    prompt: str | None = None,
    crons: list[str] | None = None,
    run_at: datetime | None = None,
    session_mode: str | None = None,
    pinned_session_id: str | None = None,
    grants_credentials: bool | None = None,
) -> ExpertRoutine:
    """Switch a routine on, resolving the proposal into what actually runs.

    *prompt*, *crons* and *run_at* are the owner's answers. Passing any of them
    marks the row customized, which takes it out of reach of every later roster
    edit — from here on what this routine does is the owner's, not the
    template's.

    *session_mode* rewrites the row, not just its ``sessionId``: a row whose
    mode still read THREAD while its id pointed at a real chat fired correctly
    by luck and described itself wrongly everywhere else — in the tool's
    confirmation, in the context block, and in any UI reading the row.

    Creates one scheduler job per cron, or exactly one for a ``run_at``. A
    partial failure leaves nothing behind: a routine that fires at 08:30 but
    not at 13:00 is a quieter lie than one that does not fire at all.
    """
    row = await _owned_routine(user_id, expert_id, routine_id)
    resolved_prompt = prompt or row.prompt
    customized = prompt is not None or crons is not None or run_at is not None
    # ``customizedAt``, not just this call: the asks are answered once, and the
    # answers live in the row from then on. Reading only the current call meant
    # a routine that was set up, run, and switched off could never be switched
    # back on — its questions are still listed, and the owner has no way to
    # answer them a second time.
    if row.asks and not customized and row.customizedAt is None:
        raise RoutineUnansweredAsksError(
            f"'{row.title}' needs answers before it can run: " + "; ".join(row.asks)
        )

    user = await get_user_by_id(user_id)
    user_timezone = get_user_timezone_or_utc(user.timezone if user else None)

    # Naming either one switches the routine to that shape, so an owner can
    # turn "every morning" into "once, tomorrow" without deleting anything.
    # Naming neither keeps whatever the row already holds.
    if crons is not None and run_at is not None:
        raise ValueError("Give a cadence or a single time, not both.")
    if crons is not None:
        source_crons, resolved_run_at = crons, None
    elif run_at is not None:
        source_crons, resolved_run_at = [], run_at
    else:
        source_crons, resolved_run_at = list(row.crons), row.runAt

    # Resolved on every path, so an ``H`` that reaches here from the template
    # or straight back from the model never gets as far as APScheduler, which
    # has no idea what it means.
    resolved_crons = [
        spread_cron(cron, seed=f"{user_id}:{row.key or row.id}:{index}")
        for index, cron in enumerate(source_crons)
    ]
    if bool(resolved_crons) == (resolved_run_at is not None):
        raise ValueError(
            "A routine runs either on a cadence or once at a time, not both "
            "and not neither."
        )
    for cron in resolved_crons:
        CronTrigger.from_crontab(cron, timezone=user_timezone)
    # A spent one-shot switched on again needs a new time: re-arming it at the
    # old one would either fire immediately or never, and neither is a thing
    # the owner asked for.
    if resolved_run_at is not None and resolved_run_at <= datetime.now(timezone.utc):
        raise ValueError("That time has already passed; give a future one.")

    mode = _session_mode(session_mode or str(row.sessionMode))
    # Only PINNED names a session up front. THREAD leaves it null for the first
    # fire to mint; FRESH leaves it null for good.
    pinned = (
        (pinned_session_id or row.sessionId)
        if mode == prisma.enums.ExpertRoutineSession.PINNED
        else None
    )
    if mode == prisma.enums.ExpertRoutineSession.PINNED and not pinned:
        # The scheduler reads a null session as "fire into a fresh chat", so
        # without this the row would say PINNED and behave like FRESH. The tool
        # always supplies one; this closes the RPC path, which does not.
        raise ValueError("A PINNED routine needs the chat it should fire into.")
    schedule_ids = await create_routine_schedules(
        user_id=user_id,
        expert_id=expert_id,
        row=row,
        prompt=resolved_prompt,
        crons=resolved_crons,
        run_at=resolved_run_at,
        session_id=pinned,
        user_timezone=user_timezone,
    )
    now = datetime.now(timezone.utc)
    data: prisma.types.ExpertRoutineUpdateInput = {
        "prompt": resolved_prompt,
        "crons": resolved_crons,
        "runAt": resolved_run_at,
        "scheduleIds": schedule_ids,
        "sessionMode": mode,
        "sessionId": pinned,
        "enabledAt": now,
        # Re-arming a spent one-shot makes it pending again; leaving the old
        # stamp would enable a row that still reads as already run.
        "firedAt": None,
    }
    # ``None`` means "leave the grant as it is", which is what an owner
    # adjusting a cadence expects. Only an explicit answer moves it, in either
    # direction, so a rewording never quietly widens what a routine can touch.
    if grants_credentials is not None:
        data["grantsCredentials"] = grants_credentials
    if customized:
        data["customizedAt"] = now
    try:
        updated = await prisma.models.ExpertRoutine.prisma().update(
            where={"id": row.id}, data=data
        )
    except Exception:
        # The jobs exist and nothing records them, so they would fire forever
        # with no row able to name or remove them. Undo them and leave the
        # caller in the state they started in.
        await _drop_schedules(user_id, schedule_ids)
        raise
    if updated is None:
        await _drop_schedules(user_id, schedule_ids)
        raise RoutineNotFoundError(routine_id)
    # Switching on a routine that was already on is how a cadence gets changed,
    # and its old jobs are still armed. Cleared last, and only once the row
    # names the new ones: a failure here leaves an extra run to explain, while
    # clearing first would leave a routine that says it is running and is not.
    await delete_routine_schedules(user_id, row)
    return to_model(updated)


def _session_mode(value: str) -> prisma.enums.ExpertRoutineSession:
    """Parse a mode name, defaulting to THREAD rather than raising.

    The value arrives as a model argument, and a typo should give the routine
    its own thread — the safe, memory-keeping default — rather than fail a call
    the owner has already agreed to.
    """
    try:
        return prisma.enums.ExpertRoutineSession(value.upper())
    except ValueError:
        logger.warning("Unknown routine session mode %r; using THREAD", value)
        return prisma.enums.ExpertRoutineSession.THREAD


async def create_routine(
    user_id: str,
    expert_id: str | None,
    *,
    title: str,
    prompt: str,
    crons: list[str] | None = None,
    run_at: datetime | None = None,
    session_mode: str | None = None,
    session_id: str | None = None,
    grants_credentials: bool = True,
) -> ExpertRoutine:
    """Record standing work worked out with the owner in conversation.

    ``key`` stays null: this came from a conversation rather than a roster, so
    nothing syncs it and no roster edit can ever reach it. It is created OFF for
    the same reason a seeded one is — agreeing what a routine should say is not
    agreeing that it should start running, and enabling is a second step the
    owner can still stop.

    ``customizedAt`` is stamped at birth: there was no proposal to resolve, so
    the row is the owner's from its first moment. ``source`` is OWNER for the
    same reason, and that is what lets it run with the reach an ordinary
    follow-up has: the prompt is the owner's own words, typed into their own
    chat. *grants_credentials* still defaults true rather than being assumed,
    so an owner who wants a routine that only reads and drafts can say so.
    """
    await _owned_owner(user_id, expert_id)
    if not title.strip() or not prompt.strip():
        raise ValueError("A routine needs a title and a prompt.")
    if bool(crons) == (run_at is not None):
        raise ValueError(
            "A routine runs either on a cadence or once at a time, not both "
            "and not neither."
        )
    if run_at is not None and run_at <= datetime.now(timezone.utc):
        raise ValueError("That time has already passed; give a future one.")
    mode = _session_mode(session_mode or "THREAD")
    data: prisma.types.ExpertRoutineCreateInput = {
        "title": title.strip(),
        "prompt": prompt.strip(),
        "crons": crons or [],
        "runAt": run_at,
        "sessionMode": mode,
        "sessionId": (
            session_id if mode == prisma.enums.ExpertRoutineSession.PINNED else None
        ),
        "source": prisma.enums.ExpertRoutineSource.OWNER,
        "grantsCredentials": grants_credentials,
        "customizedAt": datetime.now(timezone.utc),
    }
    if expert_id is None:
        data["userId"] = user_id
    else:
        data["expertId"] = expert_id
    created = await prisma.models.ExpertRoutine.prisma().create(data=data)
    return to_model(created)


async def disable_routine(
    user_id: str, expert_id: str | None, routine_id: str
) -> ExpertRoutine:
    """Switch a routine off, deleting its jobs but keeping the row.

    The resolved prompt and cadence survive, so switching it back on later
    restores what the owner set up rather than reverting to the proposal.
    """
    row = await _owned_routine(user_id, expert_id, routine_id)
    await delete_routine_schedules(user_id, row)
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


async def _drop_schedules(user_id: str, schedule_ids: list[str]) -> None:
    """Remove jobs that were created for a row write that never landed."""
    scheduler = get_scheduler_client()
    for schedule_id in schedule_ids:
        try:
            await scheduler.delete_schedule(schedule_id, user_id=user_id)
        except Exception as e:
            logger.warning(
                f"Leaked routine schedule #{schedule_id} after a failed "
                f"enable write: {type(e).__name__}: {e}"
            )


def _owner_where(
    user_id: str, expert_id: str | None
) -> prisma.types.ExpertRoutineWhereInput:
    """The one place that says what "yours" means for a routine.

    An expert's routines are reached through the expert, so the owner check is
    the join and an archived or template expert has none. The account's own are
    reached directly, and ``expertId: None`` is load-bearing: without it this
    would also match every routine belonging to every expert the user owns.
    """
    if expert_id is None:
        return {"userId": user_id, "expertId": None}
    return {
        "expertId": expert_id,
        "Expert": {
            "is": {
                "ownerUserId": user_id,
                "isTemplate": False,
                "isArchived": False,
            }
        },
    }


async def _owned_owner(user_id: str, expert_id: str | None) -> None:
    """The ownership check ``_owned_routine`` gets for free from its where
    clause, for the create path that has no routine to look up yet.

    The account is always its own owner, so only the expert case can fail.
    """
    if expert_id is None:
        return
    expert = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
        }
    )
    if expert is None:
        raise RoutineNotFoundError(expert_id)


async def _owned_routine(
    user_id: str, expert_id: str | None, routine_id: str
) -> prisma.models.ExpertRoutine:
    row = await prisma.models.ExpertRoutine.prisma().find_first(
        # Scoped rather than trusting the caller's ids: this is the only
        # ownership check between a routine id and somebody else's standing
        # work.
        where={"id": routine_id, **_owner_where(user_id, expert_id)}
    )
    if row is None:
        raise RoutineNotFoundError(routine_id)
    return row
