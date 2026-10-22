import asyncio
import concurrent.futures
import functools
import hashlib
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Protocol, Union
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from zoneinfo import ZoneInfoNotFoundError

from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MAX_INSTANCES,
    EVENT_JOB_MISSED,
)
from apscheduler.job import Job as JobObj
from apscheduler.jobstores.base import JobLookupError
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.util import ZoneInfo
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import MetaData, create_engine, or_

from backend.api.features.experts.models import ExpertRoutine
from backend.api.features.onboarding_dump.retention import (
    RecordingPurgeResult,
    purge_expired_recordings,
)
from backend.api.features.onboarding_dump.storage import RECORDING_PURGE_INTERVAL
from backend.blocks import get_blocks
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.at_reset import (
    AT_RESET_JOB_NAMES,
    AtResetKind,
    at_reset_schedule_id,
    fire_verdict,
    load_at_reset_chat,
)
from backend.copilot.at_reset_lock import at_reset_job_lock
from backend.copilot.config import CopilotLlmAuthProvider
from backend.copilot.continuation import RESUMED_AT_RESET_NOTE
from backend.copilot.credential_selection import CredentialPins
from backend.copilot.db import PaginatedMessages
from backend.copilot.dream.deadline import bridge_timeout
from backend.copilot.dream.scheduling import (
    COMMUNITY_REBUILD_JOB_PREFIX,
    DREAM_SYSTEM_JOBS,
    NIGHTLY_BATCH_JOB_PREFIX,
    SKILL_LEARNING_JOB_PREFIX,
    DreamSystemJob,
    dream_system_job,
    jobs_for,
)
from backend.copilot.dream.scope_crons import (
    forget_registration,
    log_community_outcome,
    log_nightly_outcome,
    memory_job_kwargs,
    memory_job_result,
    memory_job_skipped,
    memory_scope_may_fire,
    nightly_error_parts,
    record_scope_run,
)
from backend.copilot.dream.scope_jobs import scope_label
from backend.copilot.dream.scope_reconcile import (
    RECONCILE_INTERVAL_MINUTES,
    RECONCILED_SUFFIX,
    ListedCron,
    reconcile,
)
from backend.copilot.executor.utils import ScheduledTurnOrigin, schedule_turn
from backend.copilot.graphiti.communities import rebuild_communities_for_user
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.model import ChatSessionInfo, create_chat_session, get_chat_session
from backend.copilot.optimize_blocks import optimize_block_descriptions
from backend.copilot.permissions import CopilotPermissions, routine_disabled_tools
from backend.copilot.scheduled_chats.notices import (
    SkippedRun,
    post_notice,
    schedule_title,
    skip_notice,
)
from backend.copilot.session_permissions import resolve_session_permissions
from backend.copilot.transports import resolve_default_chat_route
from backend.copilot.turn_queue import TurnRefusal, turn_refusal
from backend.data.db_accessors import experts_db
from backend.data.execution import ExecutionTrigger, GraphExecutionWithNodes
from backend.data.memory_schedule import ScopeRunKind
from backend.data.model import CredentialsMetaInput, GraphInput
from backend.data.redis_client import get_redis_async
from backend.data.schedule import normalize_schedule_name
from backend.data.user import is_user_banned
from backend.executor import schedule_events
from backend.executor import utils as execution_utils
from backend.executor.jobstore import WRITE_TIMEOUT, ResilientSQLAlchemyJobStore
from backend.executor.schedule_index import ScheduleIndex, ScheduleIndexEntry
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.monitoring import (
    flush_matured_alerts,
    report_block_error_rates,
    report_execution_accuracy_alerts,
    report_late_executions,
    report_orphaned_auth_identities,
    send_due_briefings,
)
from backend.monitoring.instrumentation import SCHEDULER_JOBS
from backend.util import product_analytics
from backend.util.clients import (
    get_database_manager_async_client,
    get_database_manager_client,
    get_scheduler_client,
)
from backend.util.cloud_storage import cleanup_expired_files_async
from backend.util.exceptions import (
    ExpertNotFoundError,
    ExpertPrivateTenancyNotFoundError,
    ExpertRunPausedError,
    GraphNotFoundError,
    GraphNotInLibraryError,
    GraphValidationError,
    NotAuthorizedError,
    NotFoundError,
    UserBannedError,
    UserPaywalledError,
)
from backend.util.logging import PrefixFilter
from backend.util.process import freeze_startup_heap
from backend.util.retry import func_retry
from backend.util.service import (
    AppService,
    AppServiceClient,
    UnhealthyServiceError,
    endpoint_to_async,
    expose,
)
from backend.util.settings import Config

if TYPE_CHECKING:
    from backend.copilot.dream.nightly_batch import NightlyBatchResult


def _extract_schema_from_url(database_url) -> tuple[str, str]:
    """
    Extracts the schema from the DATABASE_URL and returns the schema and cleaned URL.
    """
    parsed_url = urlparse(database_url)
    query_params = parse_qs(parsed_url.query)

    # Extract the 'schema' parameter
    schema_list = query_params.pop("schema", None)
    schema = schema_list[0] if schema_list else "public"

    # Reconstruct the query string without the 'schema' parameter
    new_query = urlencode(query_params, doseq=True)
    new_parsed_url = parsed_url._replace(query=new_query)
    database_url_clean = str(urlunparse(new_parsed_url))

    return schema, database_url_clean


logger = logging.getLogger(__name__)
logger.addFilter(PrefixFilter("[Scheduler]"))
apscheduler_logger = logger.getChild("apscheduler")
apscheduler_logger.addFilter(PrefixFilter("[Scheduler] [APScheduler]"))

config = Config()

# Timeout constants
SCHEDULER_OPERATION_TIMEOUT_SECONDS = 300  # 5 minutes for scheduler operations
# Dream / nightly / community-rebuild operations call into OpenRouter
# with 50K-token prompts and FalkorDB-backed Leiden passes that
# legitimately take >5 min. Match the dream lock's 30 min TTL so the
# future resolves before the lock expires under the dream pass.
SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS = 1800


# The Stripe tier sweep pages through every active subscription, so it needs a
# generous ceiling relative to the per-op default.
STRIPE_RECONCILE_TIMEOUT_SECONDS = 1800  # 30 minutes

# The memory cron reconcile reads every registry row; it must finish well
# inside its own interval (``scope_reconcile.RECONCILE_INTERVAL_MINUTES``).
# Its listing of the dream-system crons runs inside that ceiling, and the
# database stops that query at its own limit: cancelling the wait leaves
# the thread running.
MEMORY_CRON_RECONCILE_TIMEOUT_SECONDS = 20 * 60
MEMORY_CRON_LISTING_TIMEOUT_SECONDS = 60
# Each of the reconcile's jobstore writes gives up after this on a row another
# transaction holds (``jobstore.WRITE_TIMEOUT``), failing only its scope.
MEMORY_CRON_WRITE_TIMEOUT_SECONDS = 10
_MEMORY_CRON_RECONCILE_JOB_ID = "reconcile_memory_scope_crons"


def job_listener(event):
    """Logs job execution outcomes for better monitoring."""
    if event.exception:
        logger.warning(
            f"Job {event.job_id} failed: {type(event.exception).__name__}: {event.exception}"
        )
    else:
        logger.info(f"Job {event.job_id} completed successfully.")


def job_missed_listener(event):
    """Logs when jobs are missed due to scheduling issues."""
    logger.warning(
        f"Job {event.job_id} was missed at scheduled time {event.scheduled_run_time}. "
        f"This can happen if the scheduler is overloaded or if previous executions are still running."
    )


def job_max_instances_listener(event):
    """Logs when jobs hit max instances limit."""
    logger.warning(
        f"Job {event.job_id} execution was SKIPPED - max instances limit reached. "
        f"Previous execution(s) are still running. "
        f"Consider increasing max_instances or check why previous executions are taking too long."
    )


_event_loop: asyncio.AbstractEventLoop | None = None
_event_loop_thread: threading.Thread | None = None

# The running Scheduler service instance, so module-level job functions
# (which APScheduler pickles by reference) can reach instance state — the
# same in-process shortcut the event-loop globals above provide.
_scheduler_instance: "Scheduler | None" = None


def reconcile_schedule_index():
    """Daily system task: rebuild the schedule index from the jobstore.

    Heals rows missed by best-effort writes and clears rows left behind by
    fired one-shots that only APScheduler's auto-removal saw.
    """
    if _scheduler_instance is None:
        logger.warning("Schedule index reconcile skipped: scheduler not running")
        return
    _scheduler_instance._reconcile_schedule_index()


@func_retry
def get_event_loop():
    """Get the shared event loop."""
    if _event_loop is None:
        raise RuntimeError("Event loop not initialized. Scheduler not started.")
    return _event_loop


def run_async(
    coro,
    timeout: float | None = SCHEDULER_OPERATION_TIMEOUT_SECONDS,
    *,
    cancel_on_timeout: bool = False,
):
    """Run a coroutine in the shared event loop and wait for completion.

    A ``TimeoutError`` does not prove the operation had no effect: it may
    have finished just as the wait ended, or be past a write it cannot take
    back. By default the coroutine is left running after a timeout.
    ``cancel_on_timeout`` requests its cancellation (requested, not awaited:
    it stops at its next await, possibly after this returns); only callers
    whose coroutine is safe to stop anywhere pass it, today the memory
    registry bridges. Graph dispatch is not: ``_add_graph_execution``
    creates the execution row before its remaining lookups, and cancelled
    there it leaves that row INCOMPLETE, never published or failed. Making
    it cancellation-safe is the follow-up that would let the default flip.
    """
    loop = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        if cancel_on_timeout:
            future.cancel()
        logger.warning(
            f"Async operation timed out after {timeout}s; "
            + ("cancellation requested" if cancel_on_timeout else "not cancelled")
        )
        raise
    except Exception as e:
        logger.warning(f"Async operation failed: {type(e).__name__}: {e}")
        raise


def execute_graph(**kwargs):
    """Execute graph in the shared event loop and wait for completion."""
    # Wait for completion to ensure job doesn't exit prematurely
    run_async(_execute_graph(**kwargs))


async def _execute_graph(**kwargs):
    args = GraphExecutionJobArgs(**kwargs)
    start_time = asyncio.get_event_loop().time()
    db = get_database_manager_async_client()
    try:
        logger.info(f"Executing recurring job for graph #{args.graph_id}")
        graph_exec: GraphExecutionWithNodes = await execution_utils.add_graph_execution(
            user_id=args.user_id,
            graph_id=args.graph_id,
            graph_version=args.graph_version,
            inputs=args.input_data,
            graph_credentials_inputs=args.input_credentials,
            organization_id=args.organization_id,
            team_id=args.team_id,
            expert_id=args.expert_id,
            trigger=ExecutionTrigger.SCHEDULE,
            trigger_ref=args.schedule_id,
            schedule_id=args.schedule_id,
        )
        await db.increment_onboarding_runs(args.user_id)
        product_analytics.track_schedule_fired(
            user_id=args.user_id,
            schedule_id=args.schedule_id,
            target=product_analytics.schedule_target(
                expert_id=args.expert_id, is_copilot_turn=False
            ),
            expert_id=args.expert_id,
            graph_id=args.graph_id,
            graph_exec_id=graph_exec.id,
        )
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(
            f"Graph execution started with ID {graph_exec.id} for graph {args.graph_id} "
            f"(took {elapsed:.2f}s to create and publish)"
        )
        if elapsed > 10:
            logger.warning(
                f"Graph execution {graph_exec.id} took {elapsed:.2f}s to create/publish - "
                f"this is unusually slow and may indicate resource contention"
            )
    except GraphNotFoundError as e:
        await _handle_graph_not_available(e, args, start_time)
    except GraphNotInLibraryError as e:
        await _handle_graph_not_available(e, args, start_time)
    except GraphValidationError:
        await _handle_graph_validation_error(args)
    except ExpertRunPausedError as e:
        # Expected while an expert is paused (budget/archive): skip quietly;
        # the schedule stays registered for one-click resume.
        logger.info(f"Skipping scheduled run for graph #{args.graph_id}: {e}")
    except UserPaywalledError as e:
        # Expected while the owner has no subscription: skip quietly. The
        # schedule stays registered so it resumes on its own once they
        # subscribe, and a recurring tick is not an error worth paging on.
        logger.info(f"Skipping scheduled run for graph #{args.graph_id}: {e}")
    except UserBannedError as e:
        # Kept registered, so the schedule resumes if the ban is lifted.
        logger.info(f"Skipping scheduled run for graph #{args.graph_id}: {e}")
    except ExpertPrivateTenancyNotFoundError:
        # Graph schedules are recurring, so the next cron tick is the retry.
        logger.warning(
            f"Skipping scheduled expert run for graph #{args.graph_id}: "
            "expert workspace unavailable; next schedule tick will retry"
        )
    except ExpertNotFoundError:
        # The schedule can outlive an archived, deleted, or no-longer-private
        # expert. Keep it registered for recovery without logging an error on
        # every tick.
        logger.info(
            f"Skipping scheduled expert run for graph #{args.graph_id}: "
            "expert unavailable"
        )
    except Exception as e:
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.error(
            f"Error executing graph {args.graph_id} after {elapsed:.2f}s: "
            f"{type(e).__name__}: {e}"
        )


def execute_held_wake(user_id: str, session_id: str):
    """Start the turn that carries a chat's answered cards."""
    # Deferred: the gate package imports this module (``gate/references.py``).
    from backend.copilot.gate.held import wake

    run_async(wake(user_id, session_id))


def execute_copilot_turn(**kwargs):
    """Enqueue a copilot turn on the executor queue and wait for completion.

    APScheduler dispatches scheduled copilot-turn jobs here. ``kwargs``
    is the serialized form of :class:`CopilotTurnJobArgs`.
    """
    run_async(_execute_copilot_turn(**kwargs))


async def _expert_scope_status(
    user_id: str, expert_id: str
) -> Literal["active", "archived", "paused", "missing", "unavailable"]:
    """Resolve an expert schedule's owner/lifecycle state without guessing.

    Mirrors ``enforce_expert_run_budget``'s gate: ``archived`` and ``paused``
    (``schedulesPausedAt`` — set by a manual pause or a budget breach) both
    block the firing, but reversibly, so the schedule must survive them.
    ``missing`` means the row is truly gone (deleted, or a wrong-owner
    probe): the strict lookup misses AND a lenient existence check (which
    ignores visibility/archive state) misses too — the only state allowed
    to delete the schedule. An expert that still exists but is hidden from
    the strict lookup (no longer PRIVATE) maps to ``unavailable`` so the
    schedule survives for recovery; ``unavailable`` also covers a
    transient DB/RPC failure for the same reason."""
    try:
        expert = await experts_db().get_expert(
            user_id, expert_id, include_workflows=False, include_archived=True
        )
        if expert is None:
            # The strict lookup hides non-PRIVATE experts. Only true
            # deletion may delete the schedule, so rule it out with the
            # visibility-blind existence check before returning "missing".
            if await experts_db().expert_row_exists(user_id, expert_id):
                return "unavailable"
            return "missing"
    except Exception:
        logger.warning(
            "Could not validate expert scope %s for scheduled copilot turn",
            expert_id[:12],
            exc_info=True,
        )
        return "unavailable"
    if expert.is_archived:
        return "archived"
    if expert.schedules_paused_at is not None:
        return "paused"
    return "active"


async def _skip_inactive_expert_scope(
    args: "CopilotTurnJobArgs",
    status: Literal["archived", "paused", "missing", "unavailable"],
) -> None:
    """Shared skip path for a firing whose expert scope is not active.

    Only ``missing`` deletes the schedule: archive, pause, and a visibility
    change are reversible, and copilot-turn schedules have no persisted
    cadence to revive from, so they must outlive all three. A transient
    lookup failure re-schedules a one-shot because APScheduler drops it
    after the fire regardless."""
    logger.warning(
        "Copilot turn schedule %s skipped — expert scope %s is %s",
        args.schedule_id,
        (args.expert_id or "?")[:12],
        status,
    )
    if status == "missing":
        await _self_delete_copilot_turn_schedule(args)
    elif status == "unavailable" and args.run_at is not None:
        await _reschedule_one_shot_after_expert_unavailable(args)


async def _routine_for_turn(args: "CopilotTurnJobArgs") -> ExpertRoutine | None:
    """Load the routine behind this job, if it is one.

    A failed lookup returns ``None``, which is the cautious side of both
    decisions it feeds: the turn lands in a fresh chat rather than one it
    cannot confirm belongs to this routine, and it runs muted rather than
    assuming an owner granted it anything.
    """
    if args.routine_id is None:
        return None
    try:
        routine = await experts_db().get_routine(args.routine_id)
    except Exception:
        logger.warning(
            "Could not load routine %s for scheduled turn %s; "
            "firing into a fresh chat with nothing granted",
            args.routine_id[:12],
            args.schedule_id,
            exc_info=True,
        )
        return None
    return routine


def _routine_turn_permissions(routine: ExpertRoutine | None) -> CopilotPermissions:
    """The capability filter a routine's turn runs under.

    Every routine turn gets one, even a fully granted one: no routine may
    schedule more of itself, because nobody is watching this turn read the page
    that might ask it to. Passing ``None`` would mean "whatever the session
    allows", and the point of this object is that the decision is made here, at
    the boundary, and is visible in the job.

    A routine loaded as ``None`` — the row is gone, or the lookup failed — is
    treated as ungranted, which is the right way to be wrong.
    """
    granted = routine is not None and routine.grants_credentials
    return CopilotPermissions(tools=sorted(routine_disabled_tools(granted=granted)))


async def _execute_copilot_turn(**kwargs):
    expert_scope_was_persisted = "expert_id" in kwargs
    args = CopilotTurnJobArgs(**kwargs)
    if args.purpose != "followup":
        await _execute_at_reset_turn(args, args.purpose)
        return
    routine = await _routine_for_turn(args)
    if routine is not None and not routine.enabled:
        # Deleting a routine's jobs is best effort — the scheduler can refuse,
        # and a spent one-shot's row outlives its job either way. So "off" has
        # to mean something at fire time too, or a job that survived being
        # switched off keeps running work its owner stopped. This is the
        # fire-time lookup ``delete_routine_schedules`` defers to.
        logger.info(
            "Copilot turn schedule %s skipped — routine %s is switched off; "
            "removing the schedule that outlived it",
            args.schedule_id,
            routine.id[:12],
        )
        await _self_delete_copilot_turn_schedule(args)
        return
    # A THREAD routine keeps one durable conversation: null until its first
    # fire mints it, reused by every fire after. Resolving it here means the
    # second fire takes the existing-session branch below and inherits that
    # branch's ownership and scope re-validation for free.
    if routine is not None and routine.session_id is not None:
        args = args.model_copy(update={"session_id": routine.session_id})
    start_time = asyncio.get_event_loop().time()
    try:
        # Resolve the target session.  ``session_id=None`` means "fire into
        # a fresh chat" — create one now so the user has somewhere visible
        # for the scheduled message to land.  For an explicit session_id
        # we still verify it exists and remains in the scope captured when
        # the schedule was created. The user may have deleted the chat, and
        # stale or tampered scope data must not route a turn into another
        # persona's memory. Self-clean either invalid schedule.
        if args.session_id is None:
            if not expert_scope_was_persisted:
                logger.info(
                    "Copilot turn schedule %s predates persisted memory scope; "
                    "preserving its legacy Otto behavior",
                    args.schedule_id,
                )
            if args.expert_id is not None:
                expert_status = await _expert_scope_status(args.user_id, args.expert_id)
                if expert_status != "active":
                    await _skip_inactive_expert_scope(args, expert_status)
                    return
            llm_auth_provider, llm_credential_id = await resolve_default_chat_route(
                args.user_id
            )
            if await _refuse_if_turn_may_not_start(args, llm_auth_provider, routine):
                return
            new_session = await create_chat_session(
                args.user_id,
                dry_run=False,
                organization_id=args.organization_id,
                team_id=args.team_id,
                expert_id=args.expert_id,
                # The message this fires is model-authored — the scheduling
                # turn wrote it, not the user. Without this the fresh session
                # defaults to "interactive" and schedule_followup becomes a
                # way to reach the staffing tools with nobody watching.
                origin="automation",
                # Model-authored or not, it still runs on the connection the
                # user chose — a scheduled turn should not quietly bill to a
                # different one than the chat it follows up on.
                llm_auth_provider=llm_auth_provider,
                llm_credential_id=llm_credential_id,
            )
            if args.expert_id and new_session.expert_id is None:
                # The scope check above passed, so the expert was archived or
                # deleted in the window before creation. `create_chat_session`
                # drops the attribution rather than failing, which would run an
                # expert's follow-up in Otto memory scope — fail closed
                # instead. Skip without deleting: archive is reversible, and
                # this window can't tell it apart from deletion. The next
                # firing's scope check routes authoritatively (missing →
                # delete, archived/paused → keep skipping).
                logger.warning(
                    f"Copilot turn schedule {args.schedule_id} skipped — expert "
                    f"{args.expert_id[:12]} stopped being active/owned while the "
                    f"session was being created"
                )
                return
            target_session_id = new_session.session_id
            target_session = new_session
            if routine is not None and routine.session_mode == "THREAD":
                # First fire of a THREAD routine: remember the conversation so
                # every later fire continues it instead of leaving a trail of
                # one-message chats. Best effort — losing this costs a thread,
                # not a run, and the next fire mints a fresh one.
                try:
                    await experts_db().record_routine_thread(
                        routine.id, target_session_id
                    )
                except Exception:
                    logger.warning(
                        "Could not record thread %s on routine %s",
                        target_session_id[:12],
                        routine.id[:12],
                        exc_info=True,
                    )
            # Nothing can be forged in a chat that has never existed before:
            # its ``origin="automation"`` refuses the staffing tools outright,
            # so no proposal can be parked here to approve. Persist the opener
            # as the user turn so the fresh session still gets a title and the
            # first-turn user context.
            persist_as_user_turn = True
            logger.info(
                f"Copilot turn schedule {args.schedule_id} creating fresh "
                f"session {target_session_id[:12]} (sentinel session_id=None)"
            )
        else:
            session = await get_chat_session(args.session_id, args.user_id)
            if session is None:
                logger.info(
                    f"Copilot turn schedule {args.schedule_id} skipped — session "
                    f"{args.session_id[:12]} no longer exists; removing schedule"
                )
                await _self_delete_copilot_turn_schedule(args)
                return
            if expert_scope_was_persisted and session.expert_id != args.expert_id:
                logger.warning(
                    f"Copilot turn schedule {args.schedule_id} skipped — session "
                    f"{args.session_id[:12]} memory scope no longer matches the "
                    "persisted schedule scope; removing schedule"
                )
                await _self_delete_copilot_turn_schedule(args)
                return
            if not expert_scope_was_persisted:
                # Legacy explicit-session jobs predate the scope field. The
                # owned target session is the only authoritative provenance
                # available, so recover from it rather than interpreting the
                # missing field as Otto.
                args = args.model_copy(update={"expert_id": session.expert_id})
            if args.expert_id is not None:
                expert_status = await _expert_scope_status(args.user_id, args.expert_id)
                if expert_status != "active":
                    await _skip_inactive_expert_scope(args, expert_status)
                    return
            if await _refuse_if_turn_may_not_start(
                args, session.metadata.llm_auth_provider, routine
            ):
                return
            target_session_id = args.session_id
            target_session = session
            # The target may be the user's own interactive Otto chat,
            # where ``origin`` says nothing about who wrote *this* turn. A
            # role="user" row here would raise the confirm watermark
            # ``expert_proposal`` gates on, letting a scheduled follow-up
            # approve the expert change the scheduling turn previewed. The
            # message is model-authored either way, so persist it as one.
            persist_as_user_turn = False

        assert target_session_id is not None
        # `schedule_turn` (not raw `enqueue_copilot_turn`) is the right entry
        # point: it acquires a per-user concurrency slot AND registers the
        # session in the stream registry before queue-publishing, so the
        # executor's streamed output reaches a known consumer instead of
        # being orphaned.
        await schedule_turn(
            session_id=target_session_id,
            user_id=args.user_id,
            turn_id=str(uuid.uuid4()),
            message=args.message,
            is_user_message=persist_as_user_turn,
            tool_call_id="scheduled_followup",
            tool_name="schedule_followup",
            # Even when the target is the user's own chat, nobody is there to
            # answer this turn's questions, e.g. which of two accounts to use.
            unattended=True,
            # So the accounts are the ones chosen when the schedule was made.
            credential_pins=_credential_pins_for_turn(args, routine),
            organization_id=args.organization_id,
            team_id=args.team_id,
            llm_auth_provider=target_session.metadata.llm_auth_provider,
            llm_credential_id=target_session.metadata.llm_credential_id,
            # Per turn, not per session: a PINNED routine fires into a chat the
            # user also drives themselves, and muting the conversation would
            # take capabilities away from the person sitting in it. What is
            # unattended is this turn.
            permissions=(
                _routine_turn_permissions(routine)
                if args.routine_id is not None
                else None
            ),
            # Nobody watches a scheduled turn, so the executor alerts when it
            # fails after this dispatch succeeded (SECRT-2799).
            scheduled=ScheduledTurnOrigin(
                schedule_id=args.schedule_id,
                routine_id=args.routine_id,
                cron=args.cron,
            ),
        )
        if routine is not None and routine.run_at is not None:
            # APScheduler drops a one-shot job once it fires, so without this
            # the row would go on describing itself as scheduled for a time
            # that has passed. After dispatch, not before: a turn that never
            # reached the queue has not run.
            try:
                await experts_db().record_routine_fired(routine.id)
            except Exception:
                logger.warning(
                    "Could not mark one-shot routine %s fired",
                    routine.id[:12],
                    exc_info=True,
                )
        product_analytics.track_schedule_fired(
            user_id=args.user_id,
            schedule_id=args.schedule_id,
            target=product_analytics.schedule_target(
                expert_id=args.expert_id, is_copilot_turn=True
            ),
            expert_id=args.expert_id,
            session_id=target_session_id,
        )
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(
            f"Dispatched scheduled copilot turn for session "
            f"{target_session_id[:12]} (took {elapsed:.2f}s)"
        )
    except (ExpertPrivateTenancyNotFoundError, ExpertNotFoundError):
        # ExpertNotFoundError covers the race where the expert is archived,
        # deleted, or loses PRIVATE visibility between the scope pre-check
        # and create_chat_session's own tenancy resolution — same reversible
        # skip as the pre-check: never delete the schedule from this window.
        logger.warning(
            f"Scheduled copilot turn for session {_session_id_label(args)} "
            "skipped because the expert workspace is unavailable"
        )
        if args.run_at is not None:
            await _reschedule_one_shot_after_expert_unavailable(args)
    except ConcurrentTurnLimitError as e:
        # User is at their per-user concurrency cap. For cron schedules the
        # next tick retries automatically; for one-shot (run_at) schedules
        # APScheduler removes the job after fire regardless of our error,
        # so re-schedule a copy 5 min out to give the slot time to free.
        logger.warning(
            f"Scheduled copilot turn for session {_session_id_label(args)} "
            f"hit concurrency cap; cron={args.cron is not None}: {e}"
        )
        if args.run_at is not None:
            await _reschedule_one_shot_after_cap(args)
    except Exception as e:
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.error(
            f"Error dispatching copilot turn for session "
            f"{_session_id_label(args)} after {elapsed:.2f}s: "
            f"{type(e).__name__}: {e}"
        )


async def _refuse_if_turn_may_not_start(
    job_args: "CopilotTurnJobArgs",
    llm_auth_provider: CopilotLlmAuthProvider,
    routine: ExpertRoutine | None,
) -> bool:
    """Refuse a fire that may not start, as ``dispatch_next_for_user`` refuses
    a queued turn, and say whether it did: the owner has no access to the route
    it bills to, or has used up their AutoPilot usage for the window.

    A cron schedule stays registered and resumes once the owner has access or
    the window resets. APScheduler drops a one-shot once it fires, so one
    refused by an outage (unreadable usage limits or plan) is retried, one
    refused by a daily or weekly limit waits for that limit to reset, and any
    other is dropped with its routine switched off rather than left pending for
    a time that has passed. Unless it was our outage, the chat says why
    (SECRT-2879).
    """
    retry = None
    try:
        refusal = await turn_refusal(job_args.user_id, llm_auth_provider)
        if refusal is not None and refusal.transient:
            retry = _reschedule_one_shot_after_limits_unreadable
    except Exception as exc:
        # Unknown access is no access: skip the fire, and retry a one-shot.
        refusal = TurnRefusal(
            reason=f"has an unreadable plan ({type(exc).__name__}: {exc})",
            kind="outage",
            transient=True,
        )
        retry = _reschedule_one_shot_after_plan_unreadable
    if refusal is None:
        return False
    waits = refusal.kind == "usage" and refusal.window in ("daily", "weekly")
    if waits:
        retry = functools.partial(
            _reschedule_one_shot_after_usage, resets_at=refusal.resets_at
        )
    run: SkippedRun
    if job_args.run_at is None:
        run, outcome = "recurring", "the schedule stays registered"
    elif retry is None:
        await _drop_job_from_routine(job_args)
        run, outcome = "dropped", "the one-shot is dropped"
    elif await retry(job_args):
        run = "waits_for_reset"
        outcome = (
            "the one-shot waits for the limit to reset"
            if waits
            else "the one-shot is retried"
        )
    else:
        run, outcome = "dropped", "the one-shot is dropped"
    logger.log(
        logging.WARNING if refusal.transient else logging.INFO,
        f"Skipping scheduled copilot turn {job_args.schedule_id}: owner "
        f"{refusal.reason}; {outcome}",
    )
    if not refusal.transient:
        await _tell_owner_why_skipped(job_args, routine, refusal, run=run)
    return True


async def _tell_owner_why_skipped(
    args: "CopilotTurnJobArgs",
    routine: ExpertRoutine | None,
    refusal: TurnRefusal,
    *,
    run: SkippedRun,
) -> None:
    """Best effort, and bounded: a note that cannot be posted must not hold up
    the scheduler's thread, and the skip itself already happened."""
    schedule_key = args.routine_id or args.routine_schedule_id or args.schedule_id
    if schedule_key is None:
        return
    title = schedule_title(
        routine_title=routine.title if routine is not None else None,
        message=args.message,
    )
    notice = skip_notice(
        refusal,
        title=title,
        now=datetime.now(tz=timezone.utc),
        run=run,
        user_timezone=args.user_timezone,
        # A routine's every fire time, not just this job's.
        crons=(
            list(routine.crons)
            if routine is not None
            else [args.cron] if args.cron else None
        ),
    )
    if notice is None:
        return
    try:
        await asyncio.wait_for(
            post_notice(
                notice,
                user_id=args.user_id,
                schedule_key=schedule_key,
                title=title,
                session_id=args.session_id,
                expert_id=args.expert_id,
            ),
            timeout=_NOTICE_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning(
            f"Could not tell the owner why scheduled copilot turn "
            f"{args.schedule_id} was skipped",
            exc_info=True,
        )


def _credential_pins_for_turn(
    args: "CopilotTurnJobArgs", routine: ExpertRoutine | None
) -> CredentialPins:
    """The accounts this fire runs on, chosen in the chat that made it.

    A routine keeps them on its row, so switching it off and on again, or
    rewording it, keeps them; a follow-up keeps them in its job. A schedule
    made before pins existed has none, and its turns take the first saved
    credential when several fit (SECRT-2804).
    """
    pins = routine.credential_pins if routine is not None else args.credential_pins
    if not pins:
        logger.info(
            "Copilot turn schedule %s has no pinned credentials; where several "
            "fit, its turn uses the first saved one",
            args.schedule_id,
        )
    return pins


def _session_id_label(args: "CopilotTurnJobArgs") -> str:
    """Log-safe short label for a (possibly None) session_id."""
    return args.session_id[:12] if args.session_id else "<new>"


# One-shot schedules that hit the per-user concurrency cap are pushed out
# this many seconds and retried at most _MAX_CAP_RETRIES times. We don't
# loop indefinitely because the original use case ("check CI in 20 min")
# is time-sensitive — a multi-hour delay defeats the purpose.
_CONCURRENCY_RETRY_DELAY_SECONDS = 300
_MAX_CAP_RETRIES = 1
_MAX_EXPERT_LOOKUP_RETRIES = 1
_MAX_PLAN_LOOKUP_RETRIES = 1
_MAX_LIMITS_RETRIES = 1
# A one-shot refused by a used-up limit waits for the reset rather than being
# dropped. Twice, because the daily reset can still find the week used up.
_MAX_USAGE_WAITS = 2
# Spreads the one-shots waiting for one reset over a few minutes, rather than
# firing them all on the stroke of midnight.
_USAGE_WAIT_SPREAD_SECONDS = 300
_RetryKind = Literal["cap", "expert_lookup", "plan_lookup", "limits", "usage"]
_MAX_RETRIES: dict[_RetryKind, int] = {
    "cap": _MAX_CAP_RETRIES,
    "expert_lookup": _MAX_EXPERT_LOOKUP_RETRIES,
    "plan_lookup": _MAX_PLAN_LOOKUP_RETRIES,
    "limits": _MAX_LIMITS_RETRIES,
    "usage": _MAX_USAGE_WAITS,
}
# How long a skipped run may spend telling its chat why. It holds a pooled
# scheduler worker meanwhile, and the skip itself has already happened.
_NOTICE_TIMEOUT_SECONDS = 10


async def _reschedule_one_shot_after_cap(args: "CopilotTurnJobArgs") -> None:
    await _reschedule_one_shot(
        args,
        reason="concurrency cap",
        name_suffix="cap-retry",
        retry_kind="cap",
    )


async def _reschedule_one_shot_after_plan_unreadable(
    args: "CopilotTurnJobArgs",
) -> bool:
    return await _reschedule_one_shot(
        args,
        reason="unreadable owner plan",
        name_suffix="plan-retry",
        retry_kind="plan_lookup",
    )


async def _reschedule_one_shot_after_limits_unreadable(
    args: "CopilotTurnJobArgs",
) -> bool:
    return await _reschedule_one_shot(
        args,
        reason="unreadable usage limits",
        name_suffix="limits-retry",
        retry_kind="limits",
    )


async def _reschedule_one_shot_after_expert_unavailable(
    args: "CopilotTurnJobArgs",
) -> None:
    await _reschedule_one_shot(
        args,
        reason="transient expert lookup failure",
        name_suffix="expert-lookup-retry",
        retry_kind="expert_lookup",
    )


async def _reschedule_one_shot_after_usage(
    args: "CopilotTurnJobArgs", *, resets_at: datetime | None
) -> bool:
    """Move a one-shot refused by a used-up limit to when that limit resets."""
    if resets_at is None:
        return False
    key = args.routine_schedule_id or args.schedule_id or args.message
    spread = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return await _reschedule_one_shot(
        args,
        reason="used-up usage limit",
        name_suffix="usage-wait",
        retry_kind="usage",
        run_at=resets_at + timedelta(seconds=spread % _USAGE_WAIT_SPREAD_SECONDS),
    )


async def _reschedule_one_shot(
    args: "CopilotTurnJobArgs",
    *,
    reason: str,
    name_suffix: str,
    retry_kind: _RetryKind,
    run_at: datetime | None = None,
) -> bool:
    """Re-create a one-shot copilot-turn schedule after a transient failure,
    and say whether it did.

    Best-effort: failures are logged. Schedules that have already been
    retried the limit for this failure kind are dropped to avoid loops. Retry
    depths round-trip independently through APScheduler's persisted kwargs so
    a transient expert lookup does not consume the concurrency-cap budget.
    The retry fires a few minutes out, or at *run_at* when given.
    """
    counts: dict[_RetryKind, int] = {
        "cap": args.cap_retry_count,
        "expert_lookup": args.expert_lookup_retry_count,
        "plan_lookup": args.plan_lookup_retry_count,
        "limits": args.limits_retry_count,
        "usage": args.usage_wait_count,
    }
    retry_count = counts[retry_kind]
    max_retries = _MAX_RETRIES[retry_kind]
    counts[retry_kind] = retry_count + 1

    if retry_count >= max_retries:
        logger.log(
            # A limit that is still used up is the owner's state, not a fault.
            logging.INFO if retry_kind == "usage" else logging.ERROR,
            f"Dropping one-shot copilot turn for session "
            f"{_session_id_label(args)} — exhausted {max_retries} "
            f"retry/retries after {reason}",
        )
        await _drop_job_from_routine(args)
        return False
    try:
        new_run_at = run_at or datetime.now(tz=timezone.utc) + timedelta(
            seconds=_CONCURRENCY_RETRY_DELAY_SECONDS
        )
        await get_scheduler_client().add_copilot_turn_schedule(
            user_id=args.user_id,
            session_id=args.session_id,
            message=args.message,
            run_at=new_run_at,
            name=f"{args.schedule_id or 'copilot'}-{name_suffix}",
            cap_retry_count=counts["cap"],
            expert_lookup_retry_count=counts["expert_lookup"],
            plan_lookup_retry_count=counts["plan_lookup"],
            limits_retry_count=counts["limits"],
            usage_wait_count=counts["usage"],
            # Preserve the user's timezone across the reschedule so the new
            # one-shot job's trigger/timezone matches the original request.
            user_timezone=args.user_timezone,
            # Same for tenancy — dropping these would re-home a fresh-session
            # turn to the user's default org after a cap retry.
            organization_id=args.organization_id,
            team_id=args.team_id,
            # And for expert attribution — dropping it would fire the retried
            # turn into a plain session, escaping the expert's thread/budget
            # and its isolated memory scope.
            expert_id=args.expert_id,
            # And for the routine behind it. Dropping this was the worst of
            # the three: the retried turn resolves no routine, so it runs
            # with the session's own permissions instead of the routine's —
            # an ungranted routine that merely lost a race to the concurrency
            # cap would come back with everything the mute exists to withhold.
            routine_id=args.routine_id,
            # The retry gets a new schedule_id; this is the one the routine's
            # row holds, so the retry can still switch the routine off.
            routine_schedule_id=args.routine_schedule_id or args.schedule_id,
            # And the accounts it was set up to run on, or the retry would
            # take the first saved one instead.
            credential_pins=args.credential_pins,
        )
        logger.info(
            f"Rescheduled one-shot copilot turn for session "
            f"{_session_id_label(args)} to {new_run_at.isoformat()} after "
            f"{reason} (retry {retry_count + 1}/{max_retries})"
        )
        return True
    except Exception:
        logger.warning(
            f"Failed to reschedule one-shot copilot turn for session "
            f"{_session_id_label(args)} after {reason}",
            exc_info=True,
        )
        await _drop_job_from_routine(args)
        return False


async def _best_effort_unschedule(
    schedule_id: str | None, graph_id: str | None, user_id: str, *, reason: str
) -> None:
    """Self-delete a schedule whose firing condition is no longer satisfiable
    (graph deleted, session deleted, validation failure, etc.).

    ``graph_id`` enables targeted cleanup of legacy jobs that predate
    ``schedule_id``; copilot-turn schedules have no graph to target and pass
    ``None``. Best-effort: failures are logged and swallowed. For recurring
    schedules the next cron tick will re-attempt the cleanup; for one-shot
    schedules APScheduler removes the job after fire anyway, so a missed
    delete here doesn't accumulate orphans indefinitely.
    """
    try:
        if schedule_id:
            await get_scheduler_client().delete_schedule(
                schedule_id=schedule_id, user_id=user_id
            )
        elif graph_id is not None:
            logger.warning(
                f"Old scheduled job for graph {graph_id} (user {user_id}) "
                f"has no schedule_id, attempting targeted cleanup"
            )
            await _cleanup_old_schedules_without_id(graph_id, user_id=user_id)
        else:
            logger.warning(
                f"Cannot unschedule (reason: {reason}) — no schedule_id "
                f"available; this is an old job, remove manually"
            )
            return
        logger.info(f"Unscheduled job {schedule_id} (reason: {reason})")
    except Exception:
        logger.warning(
            f"Failed to unschedule job {schedule_id} (reason: {reason})",
            exc_info=True,
        )


async def _self_delete_copilot_turn_schedule(args: "CopilotTurnJobArgs") -> None:
    """Remove a copilot schedule whose target is unavailable or out of scope."""
    # Copilot-turn schedules aren't graph-bound — no graph target for legacy
    # cleanup, so a schedule_id-less job can only be removed manually.
    await _best_effort_unschedule(
        args.schedule_id,
        None,
        args.user_id,
        reason="session unavailable or scope mismatch",
    )
    await _drop_job_from_routine(args)


async def _drop_job_from_routine(args: "CopilotTurnJobArgs") -> None:
    # A job with no schedule_id predates the field and cannot be matched
    # against the ids a routine row holds, so there is nothing to drop.
    schedule_id = args.routine_schedule_id or args.schedule_id
    if args.routine_id is None or schedule_id is None:
        return
    # The row outlives the job it lost, and a routine still listed as switched
    # on with nothing scheduled behind it is the one state the owner cannot act
    # on: the UI offers to switch off something that is already not running.
    # Most often this is a PINNED routine whose chat the owner deleted, or a
    # one-shot that fired while its owner could not pay.
    try:
        await experts_db().mark_routine_unscheduled(args.routine_id, schedule_id)
    except Exception:
        logger.warning(
            f"Could not switch off routine {args.routine_id[:12]} after removing "
            f"its schedule {schedule_id}",
            exc_info=True,
        )


# When a chat's resume or send at the reset was cancelled, by job id. A job
# cancelled while it runs is already out of the job store, so the cancel
# finds nothing to remove; before firing or putting itself back the job looks
# here. Kept in Redis, since the replica that takes the cancel need not be the
# one running the job, and in this process too, for when Redis can't be read.
_at_reset_cancels: dict[str, datetime] = {}
_at_reset_cancels_lock = threading.Lock()
_AT_RESET_CANCEL_MEMORY = timedelta(hours=2)
_AT_RESET_CANCEL_KEY_PREFIX = "copilot:at_reset_cancel:"
# Kept as a sorted set's score, in microseconds since the epoch, and only
# ever moved later: a write that stalled and lands after a newer cancel can't
# put the older time back.
_AT_RESET_CANCEL_MEMBER = "cancelled_at"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
# How long writing or reading a cancel waits on Redis, whose client retries
# for far longer, before going on with only this replica's memory of it.
_AT_RESET_CANCEL_SHARE_TIMEOUT_SECONDS = 2

# An at-reset turn that can't fire yet (the chat is busy, the budget or the
# expert can't be read, the queue turned it away) tries again five minutes
# later, this many times: an hour, so a long turn already running in the chat
# can finish. One still out of budget moves to the next reset instead, which
# starts the count again.
_MAX_AT_RESET_RETRIES = 12


async def _execute_at_reset_turn(args: "CopilotTurnJobArgs", kind: AtResetKind) -> None:
    """Fire a "Resume at reset" or "Send at reset" the user set on a chat.

    ``fire_verdict`` decides whether it still should; this carries that out.
    Whenever it can't fire yet it goes back under its own id, so the chat
    keeps at most one of each kind.
    """
    label = _session_id_label(args)
    try:
        chat = await _at_reset_chat(args)
        if chat is None:
            return
        verdict = await fire_verdict(kind, chat, args.created_at)
        if await _cancelled_after_set(args):
            logger.info(
                f"{kind}-at-reset turn for session {label} was cancelled "
                "while it ran; dropping it"
            )
            return
        if verdict.action == "fire":
            await _dispatch_at_reset_turn(args, kind, chat.session)
        elif verdict.action == "wait" and verdict.run_at is not None:
            logger.info(
                f"{kind}-at-reset turn for session {label} is still out of "
                f"budget; moving it to {verdict.run_at.isoformat()}"
            )
            await _readd_at_reset_turn(args, run_at=verdict.run_at, retries=0)
        elif verdict.action == "retry":
            await _rearm_at_reset_turn(args, reason=verdict.why)
        else:
            logger.info(
                f"Dropping {kind}-at-reset turn for session {label}: {verdict.why}"
            )
    except Exception as e:
        logger.warning(
            f"{kind}-at-reset turn for session {label} could not be dispatched",
            exc_info=True,
        )
        await _rearm_at_reset_turn(args, reason=f"{type(e).__name__}: {e}")


async def _at_reset_chat(args: "CopilotTurnJobArgs") -> PaginatedMessages | None:
    """The chat an at-reset turn fires into, or None once that's been handled.

    A chat that is gone, someone else's, or out of the job's scope removes the
    job, as for a follow-up; an expert that can't be looked up tries again.
    """
    chat = (
        await load_at_reset_chat(args.session_id, args.user_id)
        if args.session_id
        else None
    )
    if chat is None or chat.session.expert_id != args.expert_id:
        logger.info(
            f"At-reset turn {args.schedule_id} skipped — session "
            f"{_session_id_label(args)} is gone or out of scope; removing it"
        )
        await _self_delete_copilot_turn_schedule(args)
        return None
    if args.expert_id is None:
        return chat
    status = await _expert_scope_status(args.user_id, args.expert_id)
    if status == "unavailable":
        await _rearm_at_reset_turn(args, reason="the expert could not be looked up")
        return None
    if status != "active":
        await _skip_inactive_expert_scope(args, status)
        return None
    return chat


async def _dispatch_at_reset_turn(
    args: "CopilotTurnJobArgs", kind: AtResetKind, session: ChatSessionInfo
) -> None:
    """Start the turn the way a follow-up into an existing chat starts.

    A resume is a system note: grey in the chat, an instruction to the model.
    A send is the user's own words, written before the reset, so unlike a
    model-authored follow-up it lands as their message.
    """
    is_send = kind == "send"
    await schedule_turn(
        session_id=session.session_id,
        user_id=args.user_id,
        turn_id=str(uuid.uuid4()),
        message=args.message if is_send else RESUMED_AT_RESET_NOTE,
        is_user_message=is_send,
        tool_call_id="scheduled_followup",
        tool_name="schedule_followup",
        unattended=True,
        credential_pins=args.credential_pins,
        organization_id=args.organization_id,
        team_id=args.team_id,
        llm_auth_provider=session.metadata.llm_auth_provider,
        llm_credential_id=session.metadata.llm_credential_id,
        # It carries on the user's own chat, so under the chat's own
        # limits, as a message sent there now would be (a builder chat's).
        permissions=resolve_session_permissions(session),
        scheduled=ScheduledTurnOrigin(schedule_id=args.schedule_id, purpose=kind),
    )
    logger.info(
        f"Dispatched {kind}-at-reset turn for session {session.session_id[:12]}"
    )


async def _rearm_at_reset_turn(args: "CopilotTurnJobArgs", *, reason: str) -> None:
    """Try an at-reset turn again in five minutes, a bounded number of times."""
    label = _session_id_label(args)
    if args.cap_retry_count >= _MAX_AT_RESET_RETRIES:
        logger.error(
            f"Dropping {args.purpose}-at-reset turn for session {label} after "
            f"{_MAX_AT_RESET_RETRIES} retries ({reason})"
        )
        return
    run_at = datetime.now(tz=timezone.utc) + timedelta(
        seconds=_CONCURRENCY_RETRY_DELAY_SECONDS
    )
    logger.info(
        f"{args.purpose}-at-reset turn for session {label} can't fire yet "
        f"({reason}); retry {args.cap_retry_count + 1}/{_MAX_AT_RESET_RETRIES} "
        f"at {run_at.isoformat()}"
    )
    await _readd_at_reset_turn(args, run_at=run_at, retries=args.cap_retry_count + 1)


async def _readd_at_reset_turn(
    args: "CopilotTurnJobArgs", *, run_at: datetime, retries: int
) -> None:
    """Put an at-reset turn back under its own id, keeping when it was set.

    Under the job's lock, so a send the user writes or a cancel they make
    meanwhile, on whichever replica, lands wholly before or after it.
    """
    job_lock = (
        at_reset_job_lock(args.schedule_id) if args.schedule_id else nullcontext(True)
    )
    async with job_lock as locked:
        if not locked:
            # Losing the user's turn would be worse than racing their write.
            logger.warning(
                f"Putting back the {args.purpose}-at-reset turn for session "
                f"{_session_id_label(args)} without its lock"
            )
        if await _cancelled_after_set(args):
            logger.info(
                f"Not putting back the {args.purpose}-at-reset turn for session "
                f"{_session_id_label(args)}: it was cancelled while it ran"
            )
            return
        await _put_back_at_reset_turn(args, run_at=run_at, retries=retries)


async def _put_back_at_reset_turn(
    args: "CopilotTurnJobArgs", *, run_at: datetime, retries: int
) -> None:
    try:
        # Under its own id this is safe to repeat, and a put-back lost on a
        # connection the scheduler had just closed would drop the user's turn.
        await get_scheduler_client(should_retry=True).add_copilot_turn_schedule(
            user_id=args.user_id,
            session_id=args.session_id,
            message=args.message,
            run_at=run_at,
            user_timezone=args.user_timezone,
            cap_retry_count=retries,
            organization_id=args.organization_id,
            team_id=args.team_id,
            expert_id=args.expert_id,
            credential_pins=args.credential_pins,
            purpose=args.purpose,
            schedule_id=args.schedule_id,
            created_at=args.created_at,
        )
    except Exception:
        logger.warning(
            f"Failed to put back the {args.purpose}-at-reset turn for session "
            f"{_session_id_label(args)}",
            exc_info=True,
        )


def _put_back_onto(
    newer: "CopilotTurnJobInfo", message: str
) -> tuple[str, datetime | None]:
    """A send put back after the user wrote another while it ran: both go,
    this one first, as the newer job the user set. One already put back there
    (a repeated call whose first try landed) is left as it is."""
    if newer.message.startswith(f"{message}\n\n"):
        return newer.message, newer.created_at
    return f"{message}\n\n{newer.message}", newer.created_at


def _remember_at_reset_cancel(job_id: str) -> None:
    now = datetime.now(tz=timezone.utc)
    with _at_reset_cancels_lock:
        stale = [
            key
            for key, at in _at_reset_cancels.items()
            if now - at > _AT_RESET_CANCEL_MEMORY
        ]
        for key in stale:
            del _at_reset_cancels[key]
        _at_reset_cancels[job_id] = now
    try:
        # Waited for: the cancel route holds the job's lock until this
        # returns, and a put-back that takes the lock next must find it.
        run_async(
            _share_at_reset_cancel(job_id, now),
            # The write keeps its own deadline; this one only backs it up.
            timeout=_AT_RESET_CANCEL_SHARE_TIMEOUT_SECONDS + 1,
        )
    except Exception:
        logger.warning(
            f"Could not share the cancel of at-reset job {job_id}; only this "
            "replica knows of it",
            exc_info=True,
        )


async def _share_at_reset_cancel(job_id: str, cancelled_at: datetime) -> None:
    key = f"{_AT_RESET_CANCEL_KEY_PREFIX}{job_id}"
    micros = (cancelled_at - _EPOCH) // timedelta(microseconds=1)
    async with asyncio.timeout(_AT_RESET_CANCEL_SHARE_TIMEOUT_SECONDS):
        redis = await get_redis_async()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.zadd(key, {_AT_RESET_CANCEL_MEMBER: micros}, gt=True)
            pipe.expire(key, _AT_RESET_CANCEL_MEMORY)
            await pipe.execute()


async def _cancelled_after_set(args: "CopilotTurnJobArgs") -> bool:
    """Whether the user cancelled this job after it was set: setting it
    again after a cancel makes a new job, set later than the cancel."""
    if args.schedule_id is None:
        return False
    cancelled_at = await _at_reset_cancelled_at(args.schedule_id)
    if cancelled_at is None:
        return False
    return args.created_at is None or args.created_at <= cancelled_at


async def _at_reset_cancelled_at(job_id: str) -> datetime | None:
    """When the job was last cancelled, on any replica. This replica's own
    memory stands in for Redis when Redis can't be read."""
    with _at_reset_cancels_lock:
        here = _at_reset_cancels.get(job_id)
    try:
        async with asyncio.timeout(_AT_RESET_CANCEL_SHARE_TIMEOUT_SECONDS):
            redis = await get_redis_async()
            micros = await redis.zscore(
                f"{_AT_RESET_CANCEL_KEY_PREFIX}{job_id}", _AT_RESET_CANCEL_MEMBER
            )
    except Exception:
        logger.warning(
            f"Could not read the cancels of at-reset job {job_id}", exc_info=True
        )
        return here
    anywhere = None if micros is None else _EPOCH + timedelta(microseconds=micros)
    return max((at for at in (here, anywhere) if at is not None), default=None)


async def _handle_graph_validation_error(args: "GraphExecutionJobArgs") -> None:
    logger.warning(
        f"Scheduled Graph {args.graph_id} failed validation. Unscheduling graph"
    )
    await _best_effort_unschedule(
        args.schedule_id,
        args.graph_id,
        args.user_id,
        reason=f"graph {args.graph_id} validation failed",
    )


async def _handle_graph_not_available(
    e: Exception, args: "GraphExecutionJobArgs", start_time: float
) -> None:
    elapsed = asyncio.get_event_loop().time() - start_time
    logger.warning(
        f"Scheduled execution blocked for deleted/archived graph {args.graph_id} "
        f"(user {args.user_id}) after {elapsed:.2f}s: {e}"
    )
    # Clean up orphaned schedules for this graph
    await _cleanup_orphaned_schedules_for_graph(args.graph_id, args.user_id)


async def _cleanup_orphaned_schedules_for_graph(graph_id: str, user_id: str) -> None:
    """
    Clean up orphaned schedules for a specific graph when execution fails with GraphNotAccessibleError.
    This happens when an agent is pulled from the Marketplace or deleted
    but schedules still exist.
    """
    # Use scheduler client to access the scheduler service
    scheduler_client = get_scheduler_client()

    # Find all schedules for this graph and user
    schedules = await scheduler_client.get_execution_schedules(
        graph_id=graph_id, user_id=user_id
    )

    for schedule in schedules:
        try:
            await scheduler_client.delete_schedule(
                schedule_id=schedule.id, user_id=user_id
            )
            logger.info(
                f"Cleaned up orphaned schedule {schedule.id} for deleted/archived graph {graph_id}"
            )
        except Exception:
            logger.exception(
                f"Failed to delete orphaned schedule {schedule.id} for graph {graph_id}"
            )


async def _cleanup_old_schedules_without_id(graph_id: str, user_id: str) -> None:
    """Remove only schedules that have no schedule_id in their job args.

    Unlike _cleanup_orphaned_schedules_for_graph (which removes ALL schedules
    for a graph), this only targets legacy jobs created before schedule_id was
    added to GraphExecutionJobArgs, preserving any valid newer schedules.
    """
    scheduler_client = get_scheduler_client()
    schedules = await scheduler_client.get_execution_schedules(
        graph_id=graph_id, user_id=user_id
    )

    for schedule in schedules:
        if schedule.schedule_id is not None:
            continue
        try:
            await scheduler_client.delete_schedule(
                schedule_id=schedule.id, user_id=user_id
            )
            logger.info(
                f"Cleaned up old schedule {schedule.id} (no schedule_id) "
                f"for graph {graph_id}"
            )
        except Exception:
            logger.exception(
                f"Failed to delete old schedule {schedule.id} for graph {graph_id}"
            )


def cleanup_expired_files():
    """Clean up expired files from cloud storage."""
    # Wait for completion
    run_async(cleanup_expired_files_async())


def purge_expired_recordings_job():
    """Delete brain-dump voice recordings that reached their max age."""
    run_async(_purge_expired_recordings())


async def _purge_expired_recordings() -> RecordingPurgeResult:
    db = get_database_manager_async_client()
    return await purge_expired_recordings(db.clear_brain_dump_audio_paths)


def execute_community_rebuild(user_id: str, expert_id: str | None = None):
    """Weekly community-rebuild cron body of one memory scope (P-1.7).

    The job is ``community_rebuild_{scope_key}``. The account's jobs pass
    only ``user_id``, as they did when the crons were per user. Runs only
    while ``_memory_scope_may_fire`` lets it, and stamps
    ``lastCommunityRunAt`` when the rebuild finishes without an error.
    """
    scope = MemoryScope.build(user_id, expert_id)
    if not _memory_scope_may_fire(scope):
        return
    result = _rebuild_scope_communities(scope)
    if not result.get("error"):
        _stamp_scope_run(scope, "community")


def _rebuild_scope_communities(scope: MemoryScope) -> dict:
    """Community rebuild of one scope, with its outcome logged.

    Sync wrapper around the async ``rebuild_communities_for_user`` so it
    can run on the APScheduler thread pool. Failures are caught inside
    the coroutine; this wrapper logs the outcome.
    """
    result = run_async(
        rebuild_communities_for_user(scope.owner_user_id, expert_id=scope.expert_id),
        timeout=SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS,
    )
    log_community_outcome(scope, result)
    return result


def _memory_scope_may_fire(scope: MemoryScope) -> bool:
    """The registry gate of a dream-system cron (``memory_scope_may_fire``),
    run on the scheduler loop. The gate bounds itself; if the bridge still
    gives up, the gate's cancellation is requested and the run skipped."""
    # A banned owner's crons stay registered and resume if the ban is lifted.
    try:
        banned = run_async(
            is_user_banned(scope.owner_user_id),
            timeout=bridge_timeout(),
            cancel_on_timeout=True,
        )
    except Exception:
        logger.warning(f"Memory cron of {scope_label(scope)} skipped: ban unreadable")
        return False
    if banned:
        logger.info(f"Memory cron of {scope_label(scope)} skipped: owner is banned")
        return False
    try:
        return run_async(
            memory_scope_may_fire(scope, _expert_scope_status),
            timeout=bridge_timeout(),
            cancel_on_timeout=True,
        )
    except concurrent.futures.TimeoutError:
        logger.warning(f"Memory cron of {scope_label(scope)} skipped: gate timed out")
        return False


def _stamp_scope_run(scope: MemoryScope, kind: ScopeRunKind) -> None:
    """Record a clean run on the scope's registry row; best-effort."""
    try:
        run_async(
            record_scope_run(scope, kind),
            timeout=bridge_timeout(),
            cancel_on_timeout=True,
        )
    except concurrent.futures.TimeoutError:
        logger.warning(f"Could not stamp the {kind} run of {scope_label(scope)}")


def _morning_briefing_crontab(user_id: str) -> str:
    """Daily 09:00-local cron, with the minute spread across the hour.

    A fixed ``0 9 * * *`` fires every user in a timezone at the same instant;
    deriving the minute from the user id spreads that batch over the hour.
    Hashed with sha256 rather than ``hash()`` because the latter is salted
    per process, which would move a user's slot on every restart.
    """
    minute = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest(), 16) % 60
    return f"{minute} 9 * * *"


_POSTHOG_LIFECYCLE_JOB_ID = "sync_posthog_lifecycles"
RECORDING_PURGE_JOB_ID = "purge_expired_brain_dump_recordings"


def _job_timezone_name(job: JobObj) -> str | None:
    """IANA name of a cron job's trigger timezone, if it has one."""
    if isinstance(job.trigger, CronTrigger):
        return str(job.trigger.timezone)
    return None


def execute_morning_briefing(user_id: str) -> None:
    """Per-user morning briefing cron body.

    Sync wrapper around the async ``generate_and_deliver_briefing`` so it
    can run on the APScheduler thread pool. Unlike
    ``execute_community_rebuild``, the coroutine does not guarantee it
    swallows every failure internally (DB / LLM calls can raise), so this
    wrapper must catch failures itself — an exception escaping a job body
    is fatal to that run, and one user's briefing must never affect
    scheduler health.
    """
    from backend.copilot.briefing.generate import generate_and_deliver_briefing

    try:
        result = run_async(
            generate_and_deliver_briefing(user_id),
            timeout=SCHEDULER_OPERATION_TIMEOUT_SECONDS,
        )
        logger.info("Morning briefing for user %s: %s", user_id[:12], result)
    except Exception as e:
        logger.error("Morning briefing failed for user %s: %s", user_id[:12], e)


def execute_nightly_batch_sync(
    user_id: str, expert_id: str | None = None
) -> "NightlyBatchResult | None":
    """Nightly batch-family cron body of one memory scope.

    The job is ``dream_nightly_batch_{scope_key}``. The account's jobs pass
    only ``user_id``, as they did when the crons were per user. Runs only
    while ``_memory_scope_may_fire`` lets it, and stamps ``lastNightlyRunAt``
    when the fan-out reports no error. Returns the ``NightlyBatchResult``,
    or ``None`` when a gate short-circuits.
    """
    scope = MemoryScope.build(user_id, expert_id)
    if not _memory_scope_may_fire(scope):
        return None
    result = _submit_nightly_batch(user_id, expert_id)
    if result is not None and not nightly_error_parts(result):
        _stamp_scope_run(scope, "nightly")
    return result


def _submit_nightly_batch(
    user_id: str,
    expert_id: str | None = None,
    *,
    trigger: Literal["cron", "admin"] = "cron",
) -> "NightlyBatchResult | None":
    """Flag-gated nightly batch-family fan-out of one scope, outcome logged.

    Sync wrapper for APScheduler. The body inside
    ``run_nightly_batch_submit`` sequentially invokes each enabled
    submitter (dream pass, ratification supersession sweep, plus
    future P2 / P3 / P4 / P11 stages). In real-batch mode each
    submitter enqueues to the provider's batch API and returns in
    seconds; results land asynchronously via the
    ``copilot_batch_executor`` poller. In sync_baseline mode (today),
    each submitter inlines its work and the function takes ~30-60s.

    Runtime flag gate (layer 3 of the 3-layer design): if
    ``DREAM_PASS_ENABLED`` flipped off after the cron was registered,
    this short-circuits before any submitter runs. The budget check
    inside the fan-out stays on the owner, whichever scope runs.

    Returns the typed ``NightlyBatchResult`` so the admin
    ``*_with_status`` wrapper can persist it on the JobStatus row.
    Returns ``None`` only when the runtime flag gate short-circuits
    before the submitter runs. ``trigger`` is what the dream pass's
    record says started it: the cron, or the admin wrapper.
    """
    from backend.copilot.dream.nightly_batch import (
        NightlyBatchResult,
        run_nightly_batch_submit,
    )
    from backend.util.feature_flag import Flag, is_feature_enabled

    scope = MemoryScope.build(user_id, expert_id)
    if not run_async(is_feature_enabled(Flag.DREAM_PASS_ENABLED, user_id)):
        logger.info(
            "Nightly batch skipped for %s — DREAM_PASS_ENABLED flipped off",
            scope_label(scope),
        )
        return None

    result: NightlyBatchResult = run_async(
        run_nightly_batch_submit(user_id, expert_id=expert_id, trigger=trigger),
        timeout=SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS,
    )
    log_nightly_outcome(scope, result)
    return result


def execute_skill_learning_sync(user_id: str):
    """The account's nightly skill-learning cron body (independent of the
    dream pass); the job is ``skill_learning_nightly_{user_id}``.

    Runs only while ``_memory_scope_may_fire`` lets the account's scope, as
    the other dream-system crons do: a cron that straggles past a pause or a
    wipe of the account's memory does nothing. Runtime flag gate (layer 3):
    if ``DREAM_SKILL_LEARNING_ENABLED`` flipped off after the cron was
    registered, the body returns before any work. The pass itself never
    raises; its outcome is logged here and kept in the per-source review
    ledger.
    """
    from backend.copilot.learning.nightly import run_skill_learning_pass
    from backend.util.feature_flag import Flag, is_feature_enabled

    if not _memory_scope_may_fire(MemoryScope.for_user(user_id)):
        return None
    if not run_async(is_feature_enabled(Flag.DREAM_SKILL_LEARNING_ENABLED, user_id)):
        logger.info(
            "Skill learning skipped for user %s — flag flipped off", user_id[:12]
        )
        return None

    result = run_async(
        run_skill_learning_pass(user_id, trigger="nightly"),
        timeout=SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS,
    )
    if result.error:
        logger.warning(
            "Skill learning errored for user %s (run %s): %s",
            user_id[:12],
            result.run_id,
            result.error,
        )
    elif result.skipped:
        logger.info(
            "Skill learning skipped for user %s (run %s): %s",
            user_id[:12],
            result.run_id,
            result.skip_reason,
        )
    else:
        logger.info(
            "Skill learning completed for user %s in %.1fs (run %s): "
            "reviewed=%d applied=%d proposed=%d",
            user_id[:12],
            result.elapsed_seconds or 0.0,
            result.run_id,
            result.reviewed,
            result.applied,
            result.proposed,
        )
    return result


# ---------------------------------------------------------------------------
# *_with_status wrappers — used by fire-and-forget admin triggers
# ---------------------------------------------------------------------------
#
# The original ``execute_*_sync`` bodies above are called from cron
# triggers and return a result that the cron-fired path just logs.
# The admin trigger pattern (POST returns 202 + job_id, frontend
# polls) needs each work body to also write status transitions to the
# Redis-backed JobStatus registry so the frontend can show
# "Consolidating..." → "Recombining..." → "Complete (3.2 min)".
#
# Rather than thread ``job_id`` through every internal call, we wrap
# the sync bodies with a thin status-recording wrapper. The wrapper
# uses ``run_async`` to call the async JobStatus helpers from this
# sync context (same bridge to the shared event loop the existing
# bodies use).
#
# The cron path keeps calling the original ``execute_*_sync`` bodies
# (no status writes). The admin path goes through these wrappers.


def execute_nightly_batch_with_status(user_id: str, job_id: str):
    """Run the nightly batch and record JobStatus transitions.

    Used by the admin trigger pattern (POST /api/admin/memory/{user}/nightly
    returns 202 + job_id; this fires asynchronously via
    ``Scheduler.schedule_immediate_nightly_batch``).
    """
    from backend.copilot.dream.job_status import (
        mark_complete,
        mark_errored,
        update_status_phase,
    )

    try:
        run_async(
            update_status_phase(kind="nightly", job_id=job_id, state="running"),
            timeout=10,
        )
    except Exception:
        # Status update failures must never crash the work body — the
        # admin frontend will just see the row stuck at "queued" until
        # mark_complete / mark_errored writes the terminal state.
        logger.warning(
            "Failed to mark nightly batch %s as running for user %s",
            job_id[:12],
            user_id[:12],
            exc_info=True,
        )

    try:
        # The admin trigger bypasses the registry gate of the cron body:
        # it runs the account's pass whatever state its scope row is in.
        result = _submit_nightly_batch(user_id, trigger="admin")
    except Exception as exc:
        logger.exception(
            "Admin-triggered nightly batch crashed for user %s job %s",
            user_id[:12],
            job_id[:12],
        )
        try:
            run_async(
                mark_errored(
                    kind="nightly",
                    job_id=job_id,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                timeout=10,
            )
        except Exception:
            logger.warning(
                "Failed to record errored status for nightly %s", job_id[:12]
            )
        return

    if result is None:
        # Runtime flag gate short-circuit — record a minimal envelope
        # so the polling client sees the run completed-as-skipped.
        from backend.copilot.dream.nightly_batch import NightlyBatchResult

        now = datetime.now(timezone.utc)
        result = NightlyBatchResult(
            user_id=user_id,
            nightly_id="",
            started_at=now,
            completed_at=now,
            elapsed_seconds=0.0,
            skipped=True,
            skip_reason="dream_pass_disabled_runtime",
        )

    # The admin row must read errored for every error the result carries
    # (see ``nightly_error_parts``) — otherwise the Memory Visualizer
    # renders a failed dream as a successful nightly run (same contract as
    # the dream-pass wrapper below).
    error_parts = nightly_error_parts(result)
    if error_parts:
        try:
            run_async(
                mark_errored(
                    kind="nightly", job_id=job_id, error=" | ".join(error_parts)
                ),
                timeout=10,
            )
        except Exception:
            logger.warning(
                "Failed to mark nightly batch %s errored for user %s",
                job_id[:12],
                user_id[:12],
                exc_info=True,
            )
        return

    # The nightly fan-out is complete once every submitter has run —
    # even when the dream submitter took a provider batch path and only
    # ENQUEUED its pass (``result.dream_in_flight``). The dream batch
    # callbacks finalize ``dream_pass`` JobStatus rows only, never this
    # nightly row, so parking it at 'submitted' would leave it stuck
    # with nothing ever writing a terminal state until the 6h TTL
    # reaps it. Close it out as complete; ``dream_in_flight`` on the
    # persisted envelope tells consumers the dream's apply step is
    # still landing asynchronously via the BatchExecutor.
    try:
        run_async(
            mark_complete(kind="nightly", job_id=job_id, result=result),
            timeout=10,
        )
    except Exception:
        logger.warning(
            "Failed to mark nightly batch %s complete for user %s",
            job_id[:12],
            user_id[:12],
            exc_info=True,
        )


def execute_dream_pass_with_status(user_id: str, job_id: str, force: bool = False):
    """Run the dream pass in isolation and record JobStatus transitions.

    ``force`` has the pass's guard expire a fresh open pass of the account
    instead of skipping behind it (the admin trigger's ``?force=true``)."""
    from backend.copilot.dream.job_status import (
        mark_complete,
        mark_errored,
        update_status_phase,
    )
    from backend.copilot.dream.orchestrator import execute_dream_pass

    try:
        run_async(
            update_status_phase(kind="dream_pass", job_id=job_id, state="running"),
            timeout=10,
        )
    except Exception:
        logger.warning(
            "Failed to mark dream pass %s as running for user %s",
            job_id[:12],
            user_id[:12],
            exc_info=True,
        )

    try:
        result = run_async(
            execute_dream_pass(user_id, status_id=job_id, trigger="admin", force=force),
            timeout=SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.exception(
            "Admin-triggered dream pass crashed for user %s job %s",
            user_id[:12],
            job_id[:12],
        )
        try:
            run_async(
                mark_errored(
                    kind="dream_pass",
                    job_id=job_id,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                timeout=10,
            )
        except Exception:
            logger.warning(
                "Failed to record errored status for dream pass %s", job_id[:12]
            )
        return

    # The Anthropic batch path returns ``DreamPassResult`` as soon as
    # the first phase submission is queued — the actual apply step
    # runs in the BatchExecutor's callback chain when phase 3 lands
    # (anywhere from minutes to ~1h later). In that case the dream
    # batch_callbacks own the final ``mark_complete`` / ``mark_errored``
    # transition; this wrapper must NOT close the row out early or
    # the GET status endpoint will report ``complete`` while the batch
    # is still processing.
    if (
        result.execution_path == "anthropic_batch"
        and not result.skipped
        and not result.error
    ):
        try:
            run_async(
                update_status_phase(
                    kind="dream_pass",
                    job_id=job_id,
                    state="submitted",
                    current_phase="consolidate",
                ),
                timeout=10,
            )
        except Exception:
            logger.warning(
                "Failed to flip dream pass %s to submitted for user %s",
                job_id[:12],
                user_id[:12],
                exc_info=True,
            )
        return

    if result.error:
        # A sync pass that returned an error result must surface as
        # errored, not complete — otherwise the Memory Visualizer renders
        # a failed dream as a successful one.
        try:
            run_async(
                mark_errored(
                    kind="dream_pass", job_id=job_id, error=result.error, result=result
                ),
                timeout=10,
            )
        except Exception:
            logger.warning(
                "Failed to mark dream pass %s errored for user %s",
                job_id[:12],
                user_id[:12],
                exc_info=True,
            )
        return

    try:
        run_async(
            mark_complete(kind="dream_pass", job_id=job_id, result=result),
            timeout=10,
        )
    except Exception:
        logger.warning(
            "Failed to mark dream pass %s complete for user %s",
            job_id[:12],
            user_id[:12],
            exc_info=True,
        )


def execute_community_rebuild_with_status(user_id: str, job_id: str):
    """Run a community rebuild and record JobStatus transitions."""
    from backend.copilot.dream.job_status import (
        mark_complete,
        mark_errored,
        update_status_phase,
    )

    try:
        run_async(
            update_status_phase(kind="rebuild", job_id=job_id, state="running"),
            timeout=10,
        )
    except Exception:
        logger.warning(
            "Failed to mark community rebuild %s as running for user %s",
            job_id[:12],
            user_id[:12],
            exc_info=True,
        )

    try:
        result = run_async(
            rebuild_communities_for_user(user_id),
            timeout=SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.exception(
            "Admin-triggered community rebuild crashed for user %s job %s",
            user_id[:12],
            job_id[:12],
        )
        try:
            run_async(
                mark_errored(
                    kind="rebuild",
                    job_id=job_id,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                timeout=10,
            )
        except Exception:
            logger.warning(
                "Failed to record errored status for rebuild %s", job_id[:12]
            )
        return

    error = result.get("error")
    if error:
        # ``rebuild_communities_for_user`` never raises — failures land
        # in ``result['error']``. Same contract as the dream/nightly
        # wrappers above: an errored rebuild must surface as errored,
        # otherwise the Memory Visualizer toasts success on a rebuild
        # that deleted every :Community node and then crashed.
        try:
            run_async(
                mark_errored(kind="rebuild", job_id=job_id, error=error),
                timeout=10,
            )
        except Exception:
            logger.warning(
                "Failed to mark community rebuild %s errored for user %s",
                job_id[:12],
                user_id[:12],
                exc_info=True,
            )
        return

    try:
        run_async(
            mark_complete(kind="rebuild", job_id=job_id, result=result),
            timeout=10,
        )
    except Exception:
        logger.warning(
            "Failed to mark community rebuild %s complete for user %s",
            job_id[:12],
            user_id[:12],
            exc_info=True,
        )


# Each dream-system cron's body and crontab, by job-id prefix: weekly
# community rebuild at owner-local 04:00 on Mondays (APScheduler counts
# weekdays from 0 = Monday), nightly batch at owner-local 03:00.
_MEMORY_CRONS: dict[str, tuple[Callable[..., object], str]] = {
    COMMUNITY_REBUILD_JOB_PREFIX: (execute_community_rebuild, "0 4 * * 0"),
    NIGHTLY_BATCH_JOB_PREFIX: (execute_nightly_batch_sync, "0 3 * * *"),
    SKILL_LEARNING_JOB_PREFIX: (execute_skill_learning_sync, "0 3 * * *"),
}
# A missed run is caught up only within this many seconds; None (the
# scheduler's default) catches it up however late. Skill learning's catch-up
# after downtime is capped to the same morning: at most one run (coalesce),
# and the pass re-checks every source's eligibility anyway.
_MEMORY_CRON_MISFIRE_GRACE: dict[str, int | None] = {
    SKILL_LEARNING_JOB_PREFIX: 6 * 3600,
}


def execute_expert_memory_cron(
    func: Callable[..., object], user_id: str, expert_id: str
) -> object:
    """A distinct reference so old binaries park expert-only job payloads."""
    return func(user_id=user_id, expert_id=expert_id)


def reconcile_memory_scope_crons() -> None:
    """System task, every ``RECONCILE_INTERVAL_MINUTES``: the dream-system
    crons back in step with the registry rows (``copilot/dream/
    scope_reconcile.py``). Safe on two schedulers at once."""
    if _scheduler_instance is None:
        logger.warning("Memory cron reconcile skipped: scheduler not running")
        return
    _scheduler_instance._reconcile_memory_crons()


def listed_cron(job: JobObj) -> ListedCron | None:
    """The scope a dream-system job runs for, from its kwargs, and the
    timezone its trigger runs in; None when the kwargs name no scope."""
    try:
        scope = MemoryScope.build(job.kwargs["user_id"], job.kwargs.get("expert_id"))
    except (KeyError, ValueError):
        logger.warning("Memory cron %s names no memory scope; left alone", job.id)
        return None
    return ListedCron(
        job_id=job.id,
        scope=scope,
        timezone=_job_timezone_name(job),
        reconciled=job.name.endswith(RECONCILED_SUFFIX),
    )


class _SchedulerCrons:
    """``scope_reconcile.CronStore`` over this scheduler's own jobstore, off
    the event loop. Each write gives up after
    ``MEMORY_CRON_WRITE_TIMEOUT_SECONDS`` on a row another transaction holds,
    failing only its scope: cancelling the wait for it would leave the
    thread, and APScheduler's lock, held."""

    def __init__(self, scheduler: "Scheduler") -> None:
        self._scheduler = scheduler

    async def listing(self) -> list[ListedCron]:
        return await asyncio.to_thread(self._scheduler._memory_crons)

    async def add(
        self, job: DreamSystemJob, scope: MemoryScope, user_timezone: str
    ) -> None:
        """Register ``job`` of ``scope``, tagged as the reconcile's."""
        await _bounded_write(
            self._scheduler._add_memory_job,
            *_MEMORY_CRONS[job.job_id_prefix],
            job,
            scope,
            user_timezone,
            True,
        )

    async def drop(self, job: DreamSystemJob, scope: MemoryScope) -> None:
        await _bounded_write(self._scheduler._drop_memory_job, job, scope)

    async def remove(self, scope: MemoryScope) -> None:
        await _bounded_write(self._scheduler.remove_scope_memory_jobs, scope)


async def _bounded_write(call: Callable[..., object], *args: object) -> None:
    """``call`` off the event loop, its jobstore writes limited to
    ``MEMORY_CRON_WRITE_TIMEOUT_SECONDS`` (``asyncio.to_thread`` carries
    ``WRITE_TIMEOUT`` into the thread)."""
    token = WRITE_TIMEOUT.set(MEMORY_CRON_WRITE_TIMEOUT_SECONDS)
    try:
        await asyncio.to_thread(call, *args)
    finally:
        WRITE_TIMEOUT.reset(token)


def _removed(job: JobObj) -> bool:
    """Remove ``job``; False when another scheduler removed it first."""
    try:
        job.remove()
    except JobLookupError:
        return False
    return True


def _forget_dream_registration(scope: MemoryScope, job: DreamSystemJob) -> bool:
    """Bridge ``forget_registration`` onto the scheduler's event loop: whether
    the scope's row no longer records the cron. Never raises: a delete that
    cannot forget keeps its cron."""
    try:
        return run_async(
            forget_registration(scope, job),
            timeout=bridge_timeout(),
            cancel_on_timeout=True,
        )
    except Exception:
        logger.warning(
            "Failed to forget %s of %s", job.name, scope_label(scope), exc_info=True
        )
        return False


def refresh_near_expiry_credentials() -> None:
    """Keep max_instances guarding the sweep until every refresh finishes."""
    run_async(IntegrationCredentialsManager().refresh_idle_credentials(), timeout=None)


def cleanup_oauth_tokens():
    """Clean up expired OAuth tokens from the database."""

    # Wait for completion
    async def _cleanup():
        db = get_database_manager_async_client()
        return await db.cleanup_expired_oauth_tokens()

    run_async(_cleanup())


def cleanup_auth_ip_events():
    """Delete sign-up and sign-in IP records past their retention."""

    async def _cleanup():
        db = get_database_manager_async_client()
        return await db.delete_expired_ip_events(config.auth_ip_event_retention_days)

    run_async(_cleanup())


def cleanup_failed_push_subscriptions():
    """Delete push subscriptions that have exceeded the failure threshold."""

    async def _cleanup():
        db = get_database_manager_async_client()
        return await db.cleanup_failed_push_subscriptions()

    run_async(_cleanup())


def cleanup_platform_link_tokens():
    """Delete PlatformLinkToken rows expired beyond the retention window."""

    async def _cleanup():
        db = get_database_manager_async_client()
        return await db.cleanup_expired_platform_link_tokens()

    run_async(_cleanup())


def reconcile_stripe_tiers():
    """Reconcile all STRIPE-sourced user tiers against live Stripe state."""

    async def _reconcile():
        db = get_database_manager_async_client()
        return await db.reconcile_all_stripe_tiers()

    run_async(_reconcile(), timeout=STRIPE_RECONCILE_TIMEOUT_SECONDS)


def sync_posthog_lifecycles():
    """Resend subscription status and lifecycle dates to PostHog (SECRT-2778).

    Only starts the sweep; it runs inside the database manager.
    """

    async def _start():
        db = get_database_manager_async_client()
        return await db.start_posthog_lifecycle_sweep()

    run_async(_start())


def execution_accuracy_alerts():
    """Check execution accuracy and send alerts if drops are detected."""
    return report_execution_accuracy_alerts()


# The dream modules import the batch executor, whose package imports this
# module, so the dream jobs below import them where they run, as the other
# dream job bodies in this module do.


def execute_dream_pass_reaper() -> None:
    """Close the dream passes of every user that outlived their lease
    (``copilot/dream/reaper.py``). The run bounds itself, and that budget is
    what stops it: ``run_async``'s timeout frees the job thread should the
    event loop stall, but does not cancel the run."""
    from backend.copilot.dream.reaper import REAPER_BUDGET_SECONDS, reap_expired_passes

    run_async(reap_expired_passes(), timeout=REAPER_BUDGET_SECONDS + 30)


def execute_dream_pass_retention() -> None:
    """Delete the closed dream passes past their retention
    (``copilot/dream/retention.py``), bounded the same way."""
    from backend.copilot.dream.retention import (
        RETENTION_BUDGET_SECONDS,
        delete_expired_records,
    )

    run_async(
        delete_expired_records(config.dream_pass_retention_days),
        timeout=RETENTION_BUDGET_SECONDS + 60,
    )


class _SystemJobs(Protocol):
    """The call the dream pass jobs make on the scheduler, typed:
    ``BackgroundScheduler.add_job`` itself is unannotated."""

    def add_job(
        self,
        func: Callable[[], None],
        trigger: str | CronTrigger,
        *,
        id: str,
        replace_existing: bool,
        max_instances: int,
        jobstore: str,
        coalesce: bool = ...,
        minutes: int = ...,
    ) -> JobObj: ...


def _register_dream_pass_jobs(scheduler: _SystemJobs) -> None:
    """The two jobs over every user's DreamPass rows, registered once at
    start like the other system jobs: the reaper every
    ``REAPER_INTERVAL_MINUTES``, retention weekly (Sunday 05:30 UTC)."""
    from backend.copilot.dream.reaper import REAPER_INTERVAL_MINUTES

    scheduler.add_job(
        execute_dream_pass_reaper,
        id="dream_pass_reaper",
        trigger="interval",
        minutes=REAPER_INTERVAL_MINUTES,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        jobstore=Jobstores.EXECUTION.value,
    )
    scheduler.add_job(
        execute_dream_pass_retention,
        CronTrigger(day_of_week="sun", hour=5, minute=30, timezone="UTC"),
        id="dream_pass_retention",
        replace_existing=True,
        max_instances=1,
        jobstore=Jobstores.EXECUTION.value,
    )


def ensure_embeddings_coverage():
    """
    Ensure all content types (store agents, blocks, docs) have embeddings for search.

    Processes ALL missing embeddings in batches of 10 per content type until 100% coverage.
    Missing embeddings = content invisible in hybrid search.

    Schedule: Runs every 6 hours (balanced between coverage and API costs).
    - Catches new content added between scheduled runs
    - Batch size 10 per content type: gradual processing to avoid rate limits
    - Manual trigger available via execute_ensure_embeddings_coverage endpoint
    """
    db_client = get_database_manager_client()
    stats = db_client.get_embedding_stats()

    # Check for error from get_embedding_stats() first
    if "error" in stats:
        logger.error(
            f"Failed to get embedding stats: {stats['error']} - skipping backfill"
        )
        return {
            "backfill": {"processed": 0, "success": 0, "failed": 0},
            "cleanup": {"deleted": 0},
            "error": stats["error"],
        }

    # Extract totals from new stats structure
    totals = stats.get("totals", {})
    without_embeddings = totals.get("without_embeddings", 0)
    coverage_percent = totals.get("coverage_percent", 0)

    total_processed = 0
    total_success = 0
    total_failed = 0

    if without_embeddings == 0:
        logger.info("All content has embeddings, skipping backfill")
    else:
        # Log per-content-type stats for visibility
        by_type = stats.get("by_type", {})
        for content_type, type_stats in by_type.items():
            if type_stats.get("without_embeddings", 0) > 0:
                logger.info(
                    f"{content_type}: {type_stats['without_embeddings']} items without embeddings "
                    f"({type_stats['coverage_percent']}% coverage)"
                )

        logger.info(
            f"Total: {without_embeddings} items without embeddings "
            f"({coverage_percent}% coverage) - processing all"
        )

        # Process in batches until no more missing embeddings
        while True:
            result = db_client.backfill_missing_embeddings(batch_size=100)

            total_processed += result["processed"]
            total_success += result["success"]
            total_failed += result["failed"]

            if result["processed"] == 0:
                # No more missing embeddings
                break

            if result["success"] == 0 and result["processed"] > 0:
                # All attempts in this batch failed - stop to avoid infinite loop
                logger.error(
                    f"All {result['processed']} embedding attempts failed - stopping backfill"
                )
                break

            # Small delay between batches to avoid rate limits
            time.sleep(1)

        logger.info(
            f"Embedding backfill completed: {total_success}/{total_processed} succeeded, "
            f"{total_failed} failed"
        )

    # Clean up orphaned embeddings for blocks and docs
    logger.info("Running cleanup for orphaned embeddings (blocks/docs)...")
    cleanup_result = db_client.cleanup_orphaned_embeddings()
    cleanup_totals = cleanup_result.get("totals", {})
    cleanup_deleted = cleanup_totals.get("deleted", 0)

    if cleanup_deleted > 0:
        logger.info(f"Cleanup completed: deleted {cleanup_deleted} orphaned embeddings")
        by_type = cleanup_result.get("by_type", {})
        for content_type, type_result in by_type.items():
            if type_result.get("deleted", 0) > 0:
                logger.info(
                    f"{content_type}: deleted {type_result['deleted']} orphaned embeddings"
                )
    else:
        logger.info("Cleanup completed: no orphaned embeddings found")

    return {
        "backfill": {
            "processed": total_processed,
            "success": total_success,
            "failed": total_failed,
        },
        "cleanup": {
            "deleted": cleanup_deleted,
        },
    }


# Monitoring functions are now imported from monitoring module


# Paused and fired-once rows are never deleted, so every scan over them is
# bounded rather than left to grow with the backlog.
_PARKED_SCAN_LIMIT = 1000

# A rolling deploy runs the outgoing pod beside this one until it exits (prod:
# ~10 s to ready, a 15 s preStop, a 300 s grace). The index is trusted only after
# a reconcile that began later, so rows that pod wrote unindexed are in it.
_SCHEDULE_INDEX_SETTLE_S = 600


class Jobstores(Enum):
    EXECUTION = "execution"
    BATCHED_NOTIFICATIONS = "batched_notifications"
    WEEKLY_NOTIFICATIONS = "weekly_notifications"
    MEMORY_CRON_RECONCILE = "memory_cron_reconcile"


class GraphExecutionJobArgs(BaseModel):
    # ``kind`` defaults to ``"graph"`` so existing persisted job kwargs
    # (which predate the discriminator) deserialize as graph schedules.
    kind: Literal["graph"] = "graph"
    schedule_id: str | None = None
    user_id: str
    graph_id: str
    graph_version: int
    agent_name: str | None = None
    cron: str
    input_data: GraphInput
    input_credentials: dict[str, CredentialsMetaInput] = Field(default_factory=dict)
    organization_id: str = ""
    team_id: str | None = None
    # Expert attribution: set when the schedule belongs to a hired expert
    # (install-time schedule, manual schedule of an expert-installed
    # workflow, or a schedule created from the expert's chat). Stamped onto
    # every execution this schedule fires. Optional for backward compat
    # with rows persisted before expert attribution.
    expert_id: str | None = None


class CopilotTurnJobArgs(BaseModel):
    kind: Literal["copilot_turn"] = "copilot_turn"
    schedule_id: str | None = None
    user_id: str
    # ``None`` means "create a fresh chat at fire-time" — the executor calls
    # ``create_chat_session`` and routes the turn into the newly-minted
    # session. A non-null value pins the followup to an existing session
    # owned by the same user and in the persisted persona scope.
    session_id: str | None = None
    message: str
    cron: str | None = None
    run_at: datetime | None = None
    # Independent persisted retry depths for transient failures. Keeping these
    # separate prevents an expert-lookup retry from consuming the later
    # concurrency-cap retry (or vice versa).
    cap_retry_count: int = 0
    expert_lookup_retry_count: int = 0
    plan_lookup_retry_count: int = 0
    limits_retry_count: int = 0
    # How often a one-shot has already waited for a used-up limit to reset.
    usage_wait_count: int = 0
    # Persisted so ``_reschedule_one_shot_after_cap`` can preserve the user's
    # timezone when re-creating a one-shot job after a concurrency-cap miss —
    # otherwise the rescheduled job's trigger defaults to UTC and the timezone
    # surfaced in ``CopilotTurnJobInfo`` / logs / UI no longer matches what
    # the user originally requested. Optional for backward compat with rows
    # persisted before this field was added.
    user_timezone: str | None = None
    # Org/team captured at schedule time so a fresh session minted at
    # fire time lands in the same tenant as the chat that scheduled it.
    # Optional for backward compat with rows persisted before org tagging.
    organization_id: str | None = None
    team_id: str | None = None
    # Expert captured at schedule time so a fresh session minted at fire time
    # is scoped to the same expert as the chat that scheduled the follow-up —
    # its runs then count toward the expert's budget, surface on her thread,
    # and read/write her isolated memory scope. None keeps legacy schedules
    # and Otto follow-ups in the user's account scope. Optional for
    # backward compat with rows persisted before this field was added.
    expert_id: str | None = None
    # Set when this job is one fire time of an ``ExpertRoutine``. The row is
    # what makes a routine more than a followup: it owns the durable thread a
    # THREAD routine reuses (minted here on the first fire) and the flag that
    # decides whether the turn may touch a connected service at all. None keeps
    # ordinary ``schedule_followup`` jobs on their existing path.
    routine_id: str | None = None
    # The routine row's id for this job when it is a one-shot's retry, which
    # gets a schedule_id of its own that the row does not hold.
    routine_schedule_id: str | None = None
    # ``{provider: pin}``: the account the user chose for each provider when
    # the follow-up was made, which every fire runs on (SECRT-2804). A routine
    # keeps its pins on its row instead. Empty on rows persisted before pins.
    credential_pins: CredentialPins = Field(default_factory=dict)
    # Every job is a follow-up but the two a user sets on an out-of-budget
    # chat: "Resume at reset" and "Send at reset" (``copilot/at_reset.py``).
    # Those keep one id per chat, stay out of the agent's schedule tools, and
    # check the budget when they fire. Rows from before this load as followups.
    purpose: Literal["followup", "resume", "send"] = "followup"
    # When the job was first set, kept when it is put back for a retry; a
    # resume stands down if the user carried the chat on after it.
    created_at: datetime | None = None


def _timezone_from_job(job_obj: JobObj) -> str:
    if hasattr(job_obj.trigger, "timezone"):
        return str(job_obj.trigger.timezone)
    return "UTC"


def _next_run_time_iso(job_obj: JobObj) -> str:
    """Render APScheduler's next_run_time. Returns "" for jobs already fired
    (one-shot DateTrigger jobs have ``next_run_time=None`` post-fire)."""
    return job_obj.next_run_time.isoformat() if job_obj.next_run_time else ""


def _record_graph_schedule_created(
    job_args: GraphExecutionJobArgs, job_obj: JobObj, *, title: str
) -> None:
    """Record a new agent/expert schedule. See ``schedule_events``."""
    if not job_args.schedule_id:
        return
    schedule_events.record_schedule_created(
        schedule_events.ScheduleCreatedRecord(
            user_id=job_args.user_id,
            schedule_id=job_args.schedule_id,
            title=title,
            target=product_analytics.schedule_target(
                expert_id=job_args.expert_id, is_copilot_turn=False
            ),
            expert_id=job_args.expert_id,
            organization_id=job_args.organization_id or None,
            cron=job_args.cron,
            graph_id=job_args.graph_id,
            next_run_time=_next_run_time_iso(job_obj) or None,
        )
    )


def _record_copilot_turn_schedule_created(
    job_args: CopilotTurnJobArgs, job_obj: JobObj, *, title: str
) -> None:
    """Record a new Autopilot/expert follow-up schedule. See ``schedule_events``."""
    if not job_args.schedule_id:
        return
    schedule_events.record_schedule_created(
        schedule_events.ScheduleCreatedRecord(
            user_id=job_args.user_id,
            schedule_id=job_args.schedule_id,
            title=title,
            target=product_analytics.schedule_target(
                expert_id=job_args.expert_id, is_copilot_turn=True
            ),
            expert_id=job_args.expert_id,
            organization_id=job_args.organization_id or None,
            cron=job_args.cron,
            run_at=job_args.run_at,
            session_id=job_args.session_id,
            next_run_time=_next_run_time_iso(job_obj) or None,
        )
    )


def _job_info_fields(job_obj: JobObj) -> dict[str, str]:
    """The id/name/next_run_time/timezone block both JobInfo classes need.

    Extracted so the per-kind ``from_db`` factories don't duplicate it.
    """
    return {
        "id": job_obj.id,
        "name": job_obj.name,
        "next_run_time": _next_run_time_iso(job_obj),
        "timezone": _timezone_from_job(job_obj),
    }


class GraphExecutionJobInfo(GraphExecutionJobArgs):
    id: str
    name: str
    next_run_time: str
    timezone: str = Field(default="UTC", description="Timezone used for scheduling")

    @staticmethod
    def from_db(
        job_args: GraphExecutionJobArgs, job_obj: JobObj
    ) -> "GraphExecutionJobInfo":
        return GraphExecutionJobInfo(
            **_job_info_fields(job_obj), **job_args.model_dump()
        )


class CopilotTurnJobInfo(CopilotTurnJobArgs):
    id: str
    name: str
    next_run_time: str
    timezone: str = Field(default="UTC", description="Timezone used for scheduling")

    @staticmethod
    def from_db(job_args: CopilotTurnJobArgs, job_obj: JobObj) -> "CopilotTurnJobInfo":
        return CopilotTurnJobInfo(**_job_info_fields(job_obj), **job_args.model_dump())


# Polymorphic schedule info — the kind discriminator picks the right shape
# during deserialization on the client side.
ScheduleInfo = Annotated[
    Union[GraphExecutionJobInfo, CopilotTurnJobInfo],
    Field(discriminator="kind"),
]


def _resolve_timezone(user_timezone: str | None, user_id: str) -> str:
    if user_timezone:
        return user_timezone
    logger.warning(
        f"No timezone provided for user {user_id}, using UTC for scheduling. "
        f"Client should pass user's timezone for correct scheduling."
    )
    return "UTC"


_DOW_DIGIT_RE = re.compile(r"\d")


def _unix_dow_to_apscheduler(n: int) -> int:
    """Map Unix-cron day-of-week (0=Sun..6=Sat, 7=Sun) to APScheduler's
    (0=Mon..6=Sun). APScheduler's ``CronTrigger.from_crontab`` does NOT
    perform this translation despite the docstring implying Unix semantics,
    so numeric day-of-week fields silently fire one day late.
    """
    return (n - 1) % 7


def _convert_dow_token(token: str) -> str:
    """Convert a single comma-separated token of a day-of-week field from
    Unix-cron numbering to APScheduler numbering. Handles ``*``, plain
    numbers, ranges (including wrap-around like ``5-2``), step modifiers,
    and lets named tokens (``mon-fri``) pass through unchanged."""
    if "/" in token:
        body, step = token.split("/", 1)
    else:
        body, step = token, None

    if body in ("*", "?"):
        result = body
    elif "-" in body:
        start, end = body.split("-", 1)
        if start.isdigit() and end.isdigit():
            start_aps = _unix_dow_to_apscheduler(int(start))
            end_aps = _unix_dow_to_apscheduler(int(end))
            if start_aps > end_aps:
                # Unix wrap-around (e.g. Sat→Tue = 6-2) → split into two
                # APS-valid ranges joined by a comma.
                tail = f"/{step}" if step else ""
                return f"{start_aps}-6{tail},0-{end_aps}{tail}"
            result = f"{start_aps}-{end_aps}"
        else:
            result = f"{start}-{end}"
    elif body.isdigit():
        result = str(_unix_dow_to_apscheduler(int(body)))
    else:
        result = body

    return f"{result}/{step}" if step else result


def _normalize_cron_day_of_week(cron: str) -> str:
    """Rewrite the day-of-week field of a 5-field Unix cron string into the
    numbering APScheduler expects. Leaves the field untouched if it contains
    no digits (``mon-fri``), is ``*``/``?``, or the cron is malformed."""
    parts = cron.split()
    if len(parts) != 5:
        return cron
    dow = parts[4]
    if dow in ("*", "?") or not _DOW_DIGIT_RE.search(dow):
        return cron
    parts[4] = ",".join(_convert_dow_token(t) for t in dow.split(","))
    return " ".join(parts)


def _build_trigger(
    *, cron: str | None, run_at: datetime | None, user_timezone: str
) -> Union[CronTrigger, DateTrigger]:
    if (cron is None) == (run_at is None):
        raise ValueError("Exactly one of `cron` or `run_at` must be provided")
    try:
        if cron is not None:
            return CronTrigger.from_crontab(
                _normalize_cron_day_of_week(cron), timezone=user_timezone
            )
        return DateTrigger(run_date=run_at, timezone=user_timezone)
    except ZoneInfoNotFoundError as e:
        # A KeyError, which the API would turn into a 500, not a 400.
        raise ValueError(f"Unknown timezone {user_timezone!r}") from e


def _updated_graph_job_args(
    info: GraphExecutionJobInfo,
    *,
    name: str | None,
    cron: str | None,
    input_data: GraphInput | None,
    input_credentials: dict[str, CredentialsMetaInput] | None,
) -> GraphExecutionJobArgs:
    """The stored job args with each provided field replaced. Identity fields
    (schedule id, owner, graph, tenancy, expert) are never editable."""
    current = info.model_dump(include=set(GraphExecutionJobArgs.model_fields))
    overrides = {
        "agent_name": normalize_schedule_name(name) if name else None,
        "cron": cron,
        "input_data": input_data,
        "input_credentials": input_credentials,
    }
    return GraphExecutionJobArgs.model_validate(
        current | {k: v for k, v in overrides.items() if v is not None}
    )


def _job_to_info(
    job: JobObj,
) -> Union[GraphExecutionJobInfo, CopilotTurnJobInfo, None]:
    """Materialize a polymorphic ``ScheduleInfo`` from an APScheduler job.

    Returns ``None`` if the job's kwargs don't deserialize as either kind
    (corrupted row or a schedule from a future code path).
    """
    job_kind = job.kwargs.get("kind", "graph")
    if job_kind == "graph":
        try:
            args = GraphExecutionJobArgs.model_validate(job.kwargs)
        except ValidationError:
            return None
        return GraphExecutionJobInfo.from_db(args, job)
    if job_kind == "copilot_turn":
        try:
            args = CopilotTurnJobArgs.model_validate(job.kwargs)
        except ValidationError:
            return None
        return CopilotTurnJobInfo.from_db(args, job)
    return None


def _index_entry(
    job_id: str, args: GraphExecutionJobArgs | CopilotTurnJobArgs
) -> ScheduleIndexEntry:
    """Build the index row for a schedule.

    *job_id* is passed separately because it must be the jobstore key
    (``job.id``): legacy rows can carry ``schedule_id=None`` in their kwargs
    even though the job itself is addressable.
    """
    return ScheduleIndexEntry(
        job_id=job_id,
        user_id=args.user_id,
        kind=args.kind,
        graph_id=args.graph_id if args.kind == "graph" else None,
        session_id=args.session_id if args.kind == "copilot_turn" else None,
        organization_id=args.organization_id or None,
        team_id=args.team_id,
        expert_id=args.expert_id,
    )


class Scheduler(AppService):
    scheduler: BackgroundScheduler
    _persistent_jobstores: dict[str, ResilientSQLAlchemyJobStore] = {}

    # Sidecar index for filtered schedule reads. ``None`` (not set up or
    # setup failed) or ``ready=False`` (backfill not yet complete) both
    # fall back to the full jobstore scan, so the index is never
    # load-bearing. Class-level defaults like the jobs cache below, so
    # every construction path has them.
    _schedule_index: ScheduleIndex | None = None
    _schedule_index_ready: bool = False
    _schedule_index_failure_version: int = 0
    _throwaway_dir: tempfile.TemporaryDirectory | None = None
    _block_load_error: Exception | None = None

    def __init__(
        self, register_system_tasks: bool = True, throwaway_jobstores: bool = False
    ):
        self.register_system_tasks = register_system_tasks
        # Test servers: jobs a real scheduler saved in the platform database
        # must not run in them, and their tests' schedules must not land there.
        self.throwaway_jobstores = throwaway_jobstores

    @classmethod
    def get_port(cls) -> int:
        return config.execution_scheduler_port

    @classmethod
    def db_pool_size(cls) -> int:
        return config.scheduler_db_pool_size

    async def health_check(self) -> str:
        # Thread-safe health check with proper initialization handling
        if not hasattr(self, "scheduler"):
            raise UnhealthyServiceError("Scheduler is still initializing")

        # Check if we're in the middle of cleanup
        if self._shutting_down:
            return await super().health_check()

        # Normal operation - check if scheduler is running
        if not self.scheduler.running:
            raise UnhealthyServiceError("Scheduler is not running")
        if self._block_load_error is not None:
            raise UnhealthyServiceError(
                f"Loading the block registry failed: {self._block_load_error}"
            )

        return await super().health_check()

    def run_service(self):
        load_dotenv()

        # Initialize the event loop for async jobs
        global _event_loop
        _event_loop = asyncio.new_event_loop()

        # Use daemon thread since it should die with the main service
        global _event_loop_thread
        _event_loop_thread = threading.Thread(
            target=_event_loop.run_forever, daemon=True, name="SchedulerEventLoop"
        )
        _event_loop_thread.start()

        db_schema, db_url = self._jobstore_database()
        # Configure executors to limit concurrency without skipping jobs
        from apscheduler.executors.pool import ThreadPoolExecutor

        # Shared between the EXECUTION jobstore and the schedule index so
        # both live in the same pool/database.
        execution_engine = create_engine(
            url=db_url,
            pool_size=self.db_pool_size(),
            max_overflow=0,
        )
        self._schedule_index = ScheduleIndex(execution_engine, schema=db_schema)
        try:
            self._schedule_index.ensure_table()
        except Exception:
            # Two replicas racing create_all(), or a transient DB error. Keep
            # the index so writes still go through once the table exists;
            # every reconcile retries the setup.
            logger.exception(
                "Schedule index setup failed; filtered schedule reads fall back "
                "to the full jobstore scan until the backfill retries it"
            )

        self._persistent_jobstores = {
            Jobstores.EXECUTION.value: ResilientSQLAlchemyJobStore(
                engine=execution_engine,
                metadata=MetaData(schema=db_schema),
                # this one is pre-existing so it keeps the
                # default table name.
                tablename="apscheduler_jobs",
            ),
            Jobstores.BATCHED_NOTIFICATIONS.value: ResilientSQLAlchemyJobStore(
                engine=create_engine(
                    url=db_url,
                    pool_size=self.db_pool_size(),
                    max_overflow=0,
                ),
                metadata=MetaData(schema=db_schema),
                tablename="apscheduler_jobs_batched_notifications",
            ),
        }
        # Named reference so ``_get_active_jobs_cached`` can query the table
        # directly with a server-side filter — see that method for why the
        # stock ``get_all_jobs()`` isn't enough. It must stay the same
        # instance as the dict entry above: the filtered read goes through
        # ``ResilientSQLAlchemyJobStore._get_jobs``, which parks an
        # unrestorable row rather than deleting it.
        self._execution_jobstore = self._persistent_jobstores[Jobstores.EXECUTION.value]

        self.scheduler = BackgroundScheduler(
            executors={
                "default": ThreadPoolExecutor(
                    max_workers=self.db_pool_size()
                ),  # Match DB pool size to prevent resource contention
            },
            job_defaults={
                "coalesce": True,  # Skip redundant missed jobs - just run the latest
                "max_instances": 1000,  # Effectively unlimited - never drop executions
                "misfire_grace_time": None,  # No time limit for missed jobs
            },
            jobstores={
                **self._persistent_jobstores,
                # These don't really need persistence
                Jobstores.WEEKLY_NOTIFICATIONS.value: MemoryJobStore(),
                Jobstores.MEMORY_CRON_RECONCILE.value: MemoryJobStore(),
            },
            logger=apscheduler_logger,
            timezone=ZoneInfo("UTC"),
        )

        if self.register_system_tasks:
            # ALERTS — empty the ten-minute debounce window. Runs every
            # minute so a condition raised at :01 goes out at :11, not at the
            # next quarter hour.
            self.scheduler.add_job(
                flush_matured_alerts,
                id="flush_matured_alerts",
                trigger="interval",
                replace_existing=True,
                seconds=60,
                # Belt and braces. This only stops APScheduler double-firing
                # the RPC; the RPC returns as soon as the pass is spawned, so
                # the real guards are the in-process one in NotificationManager
                # and the per-user claim the work itself takes.
                max_instances=1,
                jobstore=Jobstores.BATCHED_NOTIFICATIONS.value,
            )

            # BRIEFINGS — hourly, because "07:30 in the user's own timezone"
            # is a different UTC hour for each of them.
            self.scheduler.add_job(
                send_due_briefings,
                CronTrigger.from_crontab("30 * * * *"),
                id="send_due_briefings",
                kwargs={},
                replace_existing=True,
                max_instances=1,
                jobstore=Jobstores.WEEKLY_NOTIFICATIONS.value,
            )

            # Notification LATE EXECUTIONS ALERT
            self.scheduler.add_job(
                report_late_executions,
                id="report_late_executions",
                trigger="interval",
                replace_existing=True,
                seconds=config.execution_late_notification_threshold_secs,
                jobstore=Jobstores.EXECUTION.value,
            )

            # Block Error Rate Monitoring
            self.scheduler.add_job(
                report_block_error_rates,
                id="report_block_error_rates",
                trigger="interval",
                replace_existing=True,
                seconds=config.block_error_rate_check_interval_secs,
                jobstore=Jobstores.EXECUTION.value,
            )

            # Auth identity <-> platform User invariant. Heals any auth
            # identity that has no platform User row and pages when it had to.
            if config.auth_identity_orphan_sweep_enabled:
                self.scheduler.add_job(
                    report_orphaned_auth_identities,
                    id="report_orphaned_auth_identities",
                    trigger="interval",
                    replace_existing=True,
                    max_instances=1,
                    seconds=config.auth_identity_orphan_check_interval_secs,
                    jobstore=Jobstores.EXECUTION.value,
                )

            # Cloud Storage Cleanup - configurable interval
            self.scheduler.add_job(
                cleanup_expired_files,
                id="cleanup_expired_files",
                trigger="interval",
                replace_existing=True,
                seconds=config.cloud_storage_cleanup_interval_hours
                * 3600,  # Convert hours to seconds
                jobstore=Jobstores.EXECUTION.value,
            )

            # OAuth Token Cleanup - configurable interval
            self.scheduler.add_job(
                cleanup_oauth_tokens,
                id="cleanup_oauth_tokens",
                trigger="interval",
                replace_existing=True,
                seconds=config.oauth_token_cleanup_interval_hours
                * 3600,  # Convert hours to seconds
                jobstore=Jobstores.EXECUTION.value,
            )

            # Sign-in IP log retention - daily
            self.scheduler.add_job(
                cleanup_auth_ip_events,
                id="cleanup_auth_ip_events",
                trigger="interval",
                hours=24,
                replace_existing=True,
                max_instances=1,
                jobstore=Jobstores.EXECUTION.value,
            )

            # Failed Push Subscription Cleanup - configurable interval
            self.scheduler.add_job(
                cleanup_failed_push_subscriptions,
                id="cleanup_failed_push_subscriptions",
                trigger="interval",
                replace_existing=True,
                seconds=config.push_subscription_cleanup_interval_hours * 3600,
                jobstore=Jobstores.EXECUTION.value,
            )

            # Platform Link Token Cleanup - configurable interval
            self.scheduler.add_job(
                cleanup_platform_link_tokens,
                id="cleanup_platform_link_tokens",
                trigger="interval",
                replace_existing=True,
                seconds=config.platform_link_token_cleanup_interval_hours * 3600,
                jobstore=Jobstores.EXECUTION.value,
            )

            # Stripe Subscription Tier Reconciliation - configurable interval
            # Safety net for missed/lost Stripe webhooks: makes Stripe the
            # eventual source of truth for STRIPE-sourced tiers (incl. downgrades).
            self.scheduler.add_job(
                reconcile_stripe_tiers,
                id="reconcile_stripe_tiers",
                trigger="interval",
                replace_existing=True,
                max_instances=1,  # Prevent overlapping sweeps
                seconds=config.stripe_tier_reconcile_interval_hours * 3600,
                jobstore=Jobstores.EXECUTION.value,
            )

            # Execution Accuracy Monitoring - configurable interval
            self.scheduler.add_job(
                execution_accuracy_alerts,
                id="report_execution_accuracy_alerts",
                trigger="interval",
                replace_existing=True,
                seconds=config.execution_accuracy_check_interval_hours
                * 3600,  # Convert hours to seconds
                jobstore=Jobstores.EXECUTION.value,
            )

            # Embedding Coverage - Every 6 hours
            # Ensures all approved agents have embeddings for hybrid search
            # Critical: missing embeddings = agents invisible in search
            self.scheduler.add_job(
                ensure_embeddings_coverage,
                id="ensure_embeddings_coverage",
                trigger="interval",
                hours=6,
                # Due now rather than called inline below: run_service() is what
                # starts the event loop uvicorn binds the RPC port on.
                next_run_time=datetime.now(timezone.utc)
                + timedelta(
                    hours=0 if config.scheduler_startup_embedding_backfill else 6
                ),
                replace_existing=True,
                max_instances=1,  # Prevent overlapping runs
                misfire_grace_time=None,
                coalesce=True,
                jobstore=Jobstores.EXECUTION.value,
            )

            # Block Description Optimization - Every 24 hours
            # Generates concise LLM-optimized block descriptions for
            # agent generation. Only processes blocks missing descriptions.
            self.scheduler.add_job(
                optimize_block_descriptions,
                id="optimize_block_descriptions",
                trigger="interval",
                hours=24,
                replace_existing=True,
                max_instances=1,
                jobstore=Jobstores.EXECUTION.value,
            )

            # Dream passes: close those that outlived their lease, and delete
            # the closed ones past their retention.
            _register_dream_pass_jobs(self.scheduler)

            # Schedule Index Reconcile - Every 24 hours
            # Safety net for the best-effort index writes: re-adds rows a
            # failed write missed and clears rows for jobs APScheduler
            # auto-removed (fired one-shots) outside our delete path.
            self.scheduler.add_job(
                reconcile_schedule_index,
                id="reconcile_schedule_index",
                trigger="interval",
                hours=24,
                replace_existing=True,
                max_instances=1,
                jobstore=Jobstores.EXECUTION.value,
            )

        self.scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self.scheduler.add_listener(job_missed_listener, EVENT_JOB_MISSED)
        self.scheduler.add_listener(job_max_instances_listener, EVENT_JOB_MAX_INSTANCES)
        # Published before start: a system job due at start (such as the
        # schedule index reconcile) runs at once, on the scheduler's thread,
        # and reaches this instance only through _scheduler_instance.
        global _scheduler_instance
        _scheduler_instance = self
        self.scheduler.start()
        if self.register_system_tasks:
            # After start: until then get_job only sees pending jobs, not the
            # jobstore, so the "leave an unchanged job alone" check can't work.
            self._register_idle_credential_sweep()
            self._register_posthog_lifecycle_sweep()
            self._register_recording_purge()
            self._register_memory_cron_reconcile()
        self._report_parked_jobs()
        self._start_schedule_index_backfill()
        threading.Thread(
            target=self._load_blocks_once_serving,
            daemon=True,
            name="SchedulerBlockLoad",
        ).start()

        # Keep the service running since BackgroundScheduler doesn't block
        super().run_service()

    def _load_blocks_once_serving(self) -> None:
        """Load the block registry once the RPC server is up, then freeze it.

        The first add-schedule loads it inside graph validation otherwise. Not in
        ``run_service``: the server starts only after it, on the loop it hands over to.
        """
        while not (self.http_server and self.http_server.started):
            if not self.scheduler.running:
                return
            time.sleep(0.1)
        load_started = time.monotonic()
        try:
            block_count = len(get_blocks())
        except Exception as e:
            # Unhealthy, so the pod restarts as it did when this ran before serving.
            self._block_load_error = e
            logger.exception(f"[{self.service_name}] Loading the block registry failed")
            return
        logger.info(
            f"[{self.service_name}] Loaded {block_count} blocks in "
            f"{time.monotonic() - load_started:.1f}s"
        )
        # The serving lifespan froze the heap before the blocks existed.
        freeze_startup_heap()

    def _jobstore_database(self) -> tuple[str | None, str]:
        """Schema and URL of the jobstores and schedule index: the platform
        database, or a SQLite file of this process's own, removed at cleanup."""
        if not self.throwaway_jobstores:
            return _extract_schema_from_url(os.getenv("DIRECT_URL"))
        self._throwaway_dir = tempfile.TemporaryDirectory(
            prefix="scheduler-jobstores-", ignore_cleanup_errors=True
        )
        return None, f"sqlite:///{self._throwaway_dir.name}/jobs.sqlite"

    def _register_idle_credential_sweep(self) -> None:
        """Keep an existing daily job's overdue run or explicit pause intact."""
        job_id = "refresh_near_expiry_credentials"
        trigger = CronTrigger(hour=3, minute=0, timezone=timezone.utc)
        existing = self.scheduler.get_job(job_id, jobstore=Jobstores.EXECUTION.value)
        if (
            existing is not None
            and str(existing.trigger) == str(trigger)
            and _job_timezone_name(existing) == "UTC"
        ):
            return
        self.scheduler.add_job(
            refresh_near_expiry_credentials,
            id=job_id,
            trigger=trigger,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=None,
            jobstore=Jobstores.EXECUTION.value,
        )

    # Paused and fired-once rows are never deleted, so this scan is bounded
    # rather than left to grow with the backlog — startup precedes the RPC port.
    def _report_parked_jobs(self) -> None:
        """Parking is recoverable but silent — startup has to say it happened."""
        for alias, store in self._persistent_jobstores.items():
            try:
                parked = store.get_parked_job_ids(limit=_PARKED_SCAN_LIMIT)
            except Exception as e:
                logger.error(f"Could not check jobstore '{alias}' for parked jobs: {e}")
                continue
            if len(parked) == _PARKED_SCAN_LIMIT:
                logger.error(
                    f"at least {len(parked)} job(s) in jobstore '{alias}' are PARKED "
                    "and will not run until repaired; the startup scan stopped at "
                    f"its cap: {parked}"
                )
            elif parked:
                logger.error(
                    f"{len(parked)} job(s) in jobstore '{alias}' are PARKED and will "
                    f"not run until repaired: {parked}"
                )

    def cleanup(self):
        if self.scheduler:
            logger.info("⏳ Shutting down scheduler...")
            self.scheduler.shutdown(wait=True)

        global _event_loop
        if _event_loop:
            logger.info("⏳ Closing event loop...")
            _event_loop.call_soon_threadsafe(_event_loop.stop)

        global _event_loop_thread
        if _event_loop_thread:
            logger.info("⏳ Waiting for event loop thread to finish...")
            _event_loop_thread.join(timeout=SCHEDULER_OPERATION_TIMEOUT_SECONDS)

        # Explicitly: a forked service process ends in os._exit, which skips the
        # directory's own finalizer.
        if self._throwaway_dir is not None:
            self._throwaway_dir.cleanup()

        super().cleanup()

    def _persist_schedule(
        self,
        *,
        dispatch_func,
        job_args: GraphExecutionJobArgs | CopilotTurnJobArgs,
        trigger,
        name: str | None,
    ) -> JobObj:
        """Register *job_args* with APScheduler.

        Centralizes the ``add_job`` call shape so all scheduled jobs use the
        same persistence flags (jobstore, ``replace_existing``, JSON-mode
        kwargs serialization that round-trips datetimes correctly, and
        ``id=schedule_id`` so the row is addressable by our generated UUID).
        """
        assert job_args.schedule_id is not None
        job = self.scheduler.add_job(
            dispatch_func,
            kwargs=job_args.model_dump(mode="json"),
            name=name,
            trigger=trigger,
            jobstore=Jobstores.EXECUTION.value,
            replace_existing=True,
            id=job_args.schedule_id,
        )
        # Index AFTER add_job: a pre-add row would race the read path's
        # dangling-row cleanup (row visible, job not yet persisted → row
        # deleted → schedule invisible until reconcile).
        self._upsert_schedule_index_row(job.id, job_args)
        # Invalidate the read cache so the new job shows up on the next
        # ``get_execution_schedules`` call instead of waiting up to
        # ``_JOBS_CACHE_TTL_S`` seconds for the cached list to expire.
        self._invalidate_jobs_cache()
        return job

    @expose
    def add_graph_execution_schedule(
        self,
        user_id: str,
        graph_id: str,
        graph_version: int,
        cron: str,
        input_data: GraphInput,
        input_credentials: dict[str, CredentialsMetaInput],
        name: Optional[str] = None,
        user_timezone: str | None = None,
        organization_id: Optional[str] = None,
        team_id: Optional[str] = None,
        expert_id: Optional[str] = None,
    ) -> GraphExecutionJobInfo:
        name = normalize_schedule_name(name)

        if expert_id is not None:
            organization_id, team_id = run_async(
                experts_db().resolve_private_expert_tenancy(user_id, expert_id)
            )

        # Validate the graph before scheduling to prevent runtime failures
        # We don't need the return value, just want the validation to run
        run_async(
            execution_utils.validate_and_construct_node_execution_input(
                graph_id=graph_id,
                user_id=user_id,
                graph_inputs=input_data,
                graph_version=graph_version,
                graph_credentials_inputs=input_credentials,
            )
        )

        user_timezone = _resolve_timezone(user_timezone, user_id)
        job_args = GraphExecutionJobArgs(
            schedule_id=str(uuid.uuid4()),
            user_id=user_id,
            graph_id=graph_id,
            graph_version=graph_version,
            agent_name=name,
            cron=cron,
            input_data=input_data,
            input_credentials=input_credentials,
            organization_id=organization_id or "",
            team_id=team_id,
            expert_id=expert_id,
        )
        job = self._persist_schedule(
            dispatch_func=execute_graph,
            job_args=job_args,
            trigger=_build_trigger(cron=cron, run_at=None, user_timezone=user_timezone),
            name=name,
        )
        logger.info(
            f"Added job {job.id} with cron schedule '{cron}' in timezone "
            f"{user_timezone}"
        )
        _record_graph_schedule_created(
            job_args, job, title=name or "Scheduled agent run"
        )
        return GraphExecutionJobInfo.from_db(job_args, job)

    @expose
    def add_copilot_turn_schedule(
        self,
        user_id: str,
        message: str,
        session_id: str | None = None,
        cron: str | None = None,
        run_at: datetime | None = None,
        name: Optional[str] = None,
        user_timezone: str | None = None,
        cap_retry_count: int = 0,
        expert_lookup_retry_count: int = 0,
        plan_lookup_retry_count: int = 0,
        limits_retry_count: int = 0,
        organization_id: str | None = None,
        team_id: str | None = None,
        expert_id: str | None = None,
        routine_id: str | None = None,
        credential_pins: CredentialPins | None = None,
        routine_schedule_id: str | None = None,
        usage_wait_count: int = 0,
        purpose: Literal["followup", "resume", "send"] = "followup",
        schedule_id: str | None = None,
        created_at: datetime | None = None,
    ) -> CopilotTurnJobInfo:
        """Schedule a copilot turn at a future time.

        When *session_id* is ``None`` the executor creates a fresh chat
        at fire time in the persisted Otto or expert scope and routes
        the turn into it. Otherwise the turn resumes the named (existing)
        session with its full history, after re-validating that scope.

        The ``*_retry_count`` arguments are set internally to bound their
        respective transient retry paths; normal callers should leave them at 0.

        *purpose* ``"resume"`` / ``"send"`` marks a chat's turn at the budget
        reset; it comes with the chat's own *schedule_id*
        (``at_reset_schedule_id``), so adding it again replaces the earlier
        job. *created_at* is passed only when putting a job back, to keep when
        it was first set; a job put back after the user set a newer one
        under the same id while it ran doesn't overwrite that one (see
        ``_put_back_onto``).
        """
        # Mirror add_graph_execution_schedule: validate the expert scope at
        # creation (active, owned, PRIVATE) and pin the schedule to the
        # owner's personal tenancy, instead of persisting a job that can only
        # ever skip at fire time.
        if expert_id is not None:
            organization_id, team_id = run_async(
                experts_db().resolve_private_expert_tenancy(user_id, expert_id)
            )

        if purpose != "followup" and schedule_id and created_at is not None:
            newer = self._newer_at_reset_job(schedule_id, created_at)
            if newer is not None and purpose == "resume":
                # Set again while this one ran: the newer resume stands.
                return newer
            if newer is not None:
                message, created_at = _put_back_onto(newer, message)

        user_timezone = _resolve_timezone(user_timezone, user_id)
        trigger = _build_trigger(cron=cron, run_at=run_at, user_timezone=user_timezone)
        job_args = CopilotTurnJobArgs(
            schedule_id=schedule_id or str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            message=message,
            cron=cron,
            run_at=run_at,
            cap_retry_count=cap_retry_count,
            expert_lookup_retry_count=expert_lookup_retry_count,
            plan_lookup_retry_count=plan_lookup_retry_count,
            limits_retry_count=limits_retry_count,
            usage_wait_count=usage_wait_count,
            user_timezone=user_timezone,
            organization_id=organization_id,
            team_id=team_id,
            expert_id=expert_id,
            routine_id=routine_id,
            credential_pins=credential_pins or {},
            routine_schedule_id=routine_schedule_id,
            purpose=purpose,
            created_at=created_at or datetime.now(timezone.utc),
        )
        if purpose != "followup":
            default_name = AT_RESET_JOB_NAMES[purpose]
        elif session_id:
            default_name = f"copilot turn (session {session_id[:8]})"
        else:
            default_name = "copilot turn (new chat)"
        job = self._persist_schedule(
            dispatch_func=execute_copilot_turn,
            job_args=job_args,
            trigger=trigger,
            name=name or default_name,
        )
        session_label = session_id[:12] if session_id else "<new>"
        logger.info(
            f"Added copilot-turn job {job.id} ({trigger.__class__.__name__}) "
            f"for session {session_label} in timezone {user_timezone}"
        )
        # An at-reset turn is put back under its own id every time it can't
        # fire yet, so the feed and the schedule metrics would count it again
        # each time; only follow-ups are recorded.
        if purpose == "followup":
            _record_copilot_turn_schedule_created(
                job_args, job, title=name or message[:80] or "Follow-up"
            )
        return CopilotTurnJobInfo.from_db(job_args, job)

    def _newer_at_reset_job(
        self, schedule_id: str, than: datetime
    ) -> CopilotTurnJobInfo | None:
        """The job under *schedule_id*, if it was set after *than*."""
        job = self.scheduler.get_job(schedule_id, jobstore=Jobstores.EXECUTION.value)
        info = _job_to_info(job) if job else None
        if not isinstance(info, CopilotTurnJobInfo) or info.created_at is None:
            return None
        return info if info.created_at > than else None

    @expose
    def cancel_at_reset_turn(
        self, user_id: str, session_id: str, kind: AtResetKind
    ) -> bool:
        """Remove a chat's resume or send at the budget reset, if it has one.

        Idempotent and quiet when there is nothing to remove: the chat route
        calls this whenever the user sends a message, so a missing job is the
        usual case rather than an error. Returns whether a job was removed.
        """
        job_id = at_reset_schedule_id(kind, session_id)
        job = self.scheduler.get_job(job_id, jobstore=Jobstores.EXECUTION.value)
        info = _job_to_info(job) if job else None
        if job is not None and (info is None or info.user_id != user_id):
            return False
        # Remembered even with no job to remove: it may be running right
        # now, about to fire or to put itself back.
        _remember_at_reset_cancel(job_id)
        if job is None:
            return False
        job.remove()
        # Its index row stays for the next listing to drop. The chat's next
        # turn at reset reuses this id, and that drop puts back the row of a
        # job set again meanwhile, which a delete here would not.
        self._invalidate_jobs_cache()
        logger.info(f"Removed {kind}-at-reset job for session {session_id[:12]}")
        return True

    @expose
    def delete_graph_execution_schedule(
        self, schedule_id: str, user_id: str
    ) -> Union[GraphExecutionJobInfo, CopilotTurnJobInfo]:
        """Delete a schedule by id, regardless of kind.

        Endpoint name kept for wire back-compat; the client binding is
        ``SchedulerClient.delete_schedule`` and accepts both graph and
        copilot-turn schedules.
        """
        job, info = self._authorized_job(schedule_id, user_id, action="delete")
        logger.info(f"Deleting job {schedule_id} (kind={info.kind})")
        job.remove()
        self._delete_schedule_index_row(schedule_id)
        # Invalidate the read cache so the deletion shows up immediately
        # on the next ``get_execution_schedules`` call.
        self._invalidate_jobs_cache()
        return info

    @expose
    def update_graph_execution_schedule(
        self,
        schedule_id: str,
        user_id: str,
        name: Optional[str] = None,
        cron: Optional[str] = None,
        user_timezone: Optional[str] = None,
        input_data: Optional[GraphInput] = None,
        input_credentials: Optional[dict[str, CredentialsMetaInput]] = None,
    ) -> GraphExecutionJobInfo:
        """Change an agent schedule's name, timing or inputs in place.

        The job keeps its id, owner, graph and tenancy, so every reference to
        it (Library links, the schedule index, run attribution) stays valid.
        Trigger, kwargs and name land in one ``modify_job`` call, which the
        jobstore writes as a single row update: a concurrent reader sees the
        old schedule or the new one, never a mix. Omitted fields keep their
        current value, and a paused schedule stays paused.
        """
        job, info = self._authorized_job(schedule_id, user_id, action="update")
        if not isinstance(info, GraphExecutionJobInfo):
            raise NotFoundError(f"Workflow schedule #{schedule_id} not found.")

        job_args = _updated_graph_job_args(
            info,
            name=name,
            cron=cron,
            input_data=input_data,
            input_credentials=input_credentials,
        )
        if input_data is not None or input_credentials is not None:
            run_async(
                execution_utils.validate_and_construct_node_execution_input(
                    graph_id=job_args.graph_id,
                    user_id=user_id,
                    graph_inputs=job_args.input_data,
                    graph_version=job_args.graph_version,
                    graph_credentials_inputs=job_args.input_credentials,
                )
            )

        changes: dict[str, Any] = {"kwargs": job_args.model_dump(mode="json")}
        if name:
            changes["name"] = job_args.agent_name
        if cron is not None or user_timezone:
            trigger = _build_trigger(
                cron=job_args.cron,
                run_at=None,
                user_timezone=user_timezone or info.timezone,
            )
            changes["trigger"] = trigger
            if job.next_run_time is not None:
                changes["next_run_time"] = trigger.get_next_fire_time(
                    None, datetime.now(timezone.utc)
                )
        updated = self.scheduler.modify_job(
            schedule_id, jobstore=Jobstores.EXECUTION.value, **changes
        )
        self._invalidate_jobs_cache()
        logger.info(
            f"Updated job {schedule_id}: cron '{job_args.cron}' in timezone "
            f"{_timezone_from_job(updated)}"
        )
        return GraphExecutionJobInfo.from_db(job_args, updated)

    def _authorized_job(
        self, schedule_id: str, user_id: str, *, action: str
    ) -> tuple[JobObj, Union[GraphExecutionJobInfo, CopilotTurnJobInfo]]:
        """Load a schedule and check *user_id* owns it.

        Shared by delete/pause/resume so all three authorize identically:
        kwargs that parse as neither kind carry no `user_id` field to
        authorize against, so acting on them would let any caller who can
        guess a schedule_id mutate corrupted rows. Surface 404 rather than
        a distinct error so the shape can't be probed by timing either.
        """
        job = self.scheduler.get_job(schedule_id, jobstore=Jobstores.EXECUTION.value)
        if not job:
            raise NotFoundError(f"Job #{schedule_id} not found.")
        info = _job_to_info(job)
        if info is None:
            logger.warning(
                f"Refusing {action} for job {schedule_id} with unrecognized "
                f"kwargs shape (no parseable user_id to authorize against)"
            )
            raise NotFoundError(f"Job #{schedule_id} has invalid schedule data.")
        if info.user_id != user_id:
            raise NotAuthorizedError("User ID does not match the job's user ID")
        return job, info

    @expose
    def pause_execution_schedule(self, schedule_id: str, user_id: str) -> bool:
        """Suspend a schedule without discarding it.

        APScheduler persists a pause as ``next_run_time = NULL``, which
        ``get_execution_schedules`` already skips — so a paused schedule
        disappears from every read path while keeping its trigger, inputs
        and credentials intact for ``resume_execution_schedule``. Returns
        False when it was already paused, so callers don't double-log.
        """
        job, info = self._authorized_job(schedule_id, user_id, action="pause")
        if job.next_run_time is None:
            return False
        logger.info(f"Pausing job {schedule_id} (kind={info.kind})")
        self.scheduler.pause_job(schedule_id, jobstore=Jobstores.EXECUTION.value)
        self._invalidate_jobs_cache()
        return True

    @expose
    def resume_execution_schedule(self, schedule_id: str, user_id: str) -> bool:
        """Reverse of ``pause_execution_schedule``.

        APScheduler recomputes ``next_run_time`` from the trigger, so fires
        missed while paused are NOT backfilled — a re-hired expert picks up
        at her next cadence instead of replaying the whole archived window.
        Returns False when the schedule was not paused.
        """
        job, info = self._authorized_job(schedule_id, user_id, action="resume")
        if job.next_run_time is not None:
            return False
        logger.info(f"Resuming job {schedule_id} (kind={info.kind})")
        self.scheduler.resume_job(schedule_id, jobstore=Jobstores.EXECUTION.value)
        self._invalidate_jobs_cache()
        return True

    @expose
    def update_copilot_turn_cron(
        self, schedule_id: str, user_id: str, cron: str
    ) -> CopilotTurnJobInfo:
        """Give a repeating follow-up a new cadence, in place.

        The job keeps its id, which is what its failure streak, its notes and
        Resume all name it by. A paused one stays paused: changing how often it
        runs is not the same thing as starting it again. A routine's jobs are
        not changed here; its row owns their crons.
        """
        job, info = self._authorized_job(schedule_id, user_id, action="reschedule")
        if not isinstance(info, CopilotTurnJobInfo) or info.cron is None:
            raise NotFoundError(f"Job #{schedule_id} is not a repeating follow-up.")
        if info.routine_id is not None:
            raise NotFoundError(f"Job #{schedule_id} belongs to a routine.")
        # The zone the dialog showed its times in: a follow-up saved before
        # user_timezone existed keeps its zone on the trigger alone.
        trigger = _build_trigger(
            cron=cron, run_at=None, user_timezone=info.user_timezone or info.timezone
        )
        args = CopilotTurnJobArgs.model_validate(job.kwargs).model_copy(
            update={"cron": cron}
        )
        changes: dict[str, object] = {
            "trigger": trigger,
            "kwargs": args.model_dump(mode="json"),
        }
        if job.next_run_time is not None:
            changes["next_run_time"] = trigger.get_next_fire_time(
                None, datetime.now(timezone.utc)
            )
        logger.info(f"Changing the cadence of job {schedule_id}")
        updated = self.scheduler.modify_job(
            schedule_id, jobstore=Jobstores.EXECUTION.value, **changes
        )
        self._invalidate_jobs_cache()
        return CopilotTurnJobInfo.from_db(args, updated)

    @expose
    def get_graph_execution_schedules(
        self,
        graph_id: str | None = None,
        user_id: str | None = None,
        organization_id: str | None = None,
        team_ids: list[str] | None = None,
        include_paused: bool = False,
    ) -> list[GraphExecutionJobInfo]:
        """Return graph-kind schedules only (typed for legacy callers).

        Thin wrapper over :meth:`get_execution_schedules` with
        ``kind="graph"``. We cast the result list because the polymorphic
        path returns ``GraphExecutionJobInfo | CopilotTurnJobInfo`` but
        the ``kind`` filter guarantees only the graph branch lands here.
        """
        return [
            info
            for info in self.get_execution_schedules(
                graph_id=graph_id,
                user_id=user_id,
                kind="graph",
                organization_id=organization_id,
                team_ids=team_ids,
                include_paused=include_paused,
            )
            if isinstance(info, GraphExecutionJobInfo)
        ]

    # Two cached reads of the EXECUTION jobstore (``_read_job_list``):
    # ``_get_jobs_cached`` unpickles every row, paused and fired-once included,
    # for the ``include_paused=True`` lookups; ``_get_active_jobs_cached`` filters
    # ``next_run_time IS NOT NULL`` in SQL, since the table keeps paused and
    # fired rows forever. Mutations call ``_invalidate_jobs_cache``.
    _JOBS_CACHE_TTL_S = 5.0
    # Process-wide; guards both lists and the version. A threading lock because
    # APScheduler's thread and the Pyro RPC workers both read.
    _jobs_cache_lock = threading.Lock()
    # Bumped by every invalidation. A read that started before one must not
    # write its result back: it may predate the mutation.
    _jobs_cache_version: int = 0
    _job_lists: "dict[str, _JobList] | None" = None

    def _get_jobs_cached(self) -> list[JobObj]:
        return self._read_job_list(
            "all", lambda: self.scheduler.get_jobs(jobstore=Jobstores.EXECUTION.value)
        )

    def _get_active_jobs_cached(self) -> list[JobObj]:
        return self._read_job_list(
            "active",
            lambda: self._execution_jobstore._get_jobs(
                self._execution_jobstore.jobs_t.c.next_run_time.isnot(None)
            ),
            # The one scheduler metric with an alert on it. Only an accepted
            # read sets it, so an invalidated one cannot overwrite a newer count.
            on_accept=lambda jobs: SCHEDULER_JOBS.labels(
                job_type="execution", status="scheduled"
            ).set(len(jobs)),
        )

    def _read_job_list(
        self,
        key: Literal["all", "active"],
        fetch: Callable[[], list[JobObj]],
        on_accept: Callable[[list[JobObj]], None] | None = None,
    ) -> list[JobObj]:
        """The cached list for *key*, with at most one fetch in flight.

        Callers of a cold or invalidated list wait for that one fetch. An
        expired list is returned as it is while a background fetch replaces
        it, so no caller waits a scan that an older list can answer.
        """
        with self._jobs_cache_lock:
            entry = self._job_list(key)
            if entry.jobs is not None and time.monotonic() < entry.expires_at:
                return entry.jobs
            stale = entry.jobs
            version = self._jobs_cache_version
            pending = entry.refresh
            starts_refresh = pending is None
            if pending is None:
                pending = entry.refresh = Future()
        if starts_refresh:
            args = (entry, pending, version, fetch, on_accept)
            if stale is None:
                self._refresh_job_list(*args)
            else:
                pending.add_done_callback(_log_refresh_failure)
                threading.Thread(
                    target=self._refresh_job_list,
                    args=args,
                    daemon=True,
                    name="ScheduleListRefresh",
                ).start()
        return stale if stale is not None else pending.result()

    def _refresh_job_list(
        self,
        entry: "_JobList",
        pending: "Future[list[JobObj]]",
        version: int,
        fetch: Callable[[], list[JobObj]],
        on_accept: Callable[[list[JobObj]], None] | None,
    ) -> None:
        try:
            jobs = fetch()
        except BaseException as e:
            with self._jobs_cache_lock:
                if entry.refresh is pending:
                    entry.refresh = None
            pending.set_exception(e)
            return
        try:
            with self._jobs_cache_lock:
                if entry.refresh is pending:
                    entry.refresh = None
                if self._jobs_cache_version == version:
                    entry.jobs = jobs
                    entry.expires_at = time.monotonic() + self._JOBS_CACHE_TTL_S
                    if on_accept is not None:
                        on_accept(jobs)
        finally:
            # Waiters block on this; nothing after the fetch may strand them.
            pending.set_result(jobs)

    def _job_list(self, key: Literal["all", "active"]) -> "_JobList":
        """The caller holds ``_jobs_cache_lock``."""
        if self._job_lists is None:
            self._job_lists = {"all": _JobList(), "active": _JobList()}
        return self._job_lists[key]

    def _invalidate_jobs_cache(self) -> None:
        with self._jobs_cache_lock:
            # Dropping the in-flight fetch too sends the next caller to a fresh
            # read rather than to one that began before this write.
            for entry in (self._job_lists or {}).values():
                entry.jobs = None
                entry.expires_at = 0.0
                entry.refresh = None
            self._jobs_cache_version += 1

    # --- schedule index (see backend/executor/schedule_index.py) ---
    #
    # The index narrows filtered reads to a handful of point lookups
    # instead of unpickling the whole jobstore. It is candidates-only:
    # every job it nominates still flows through the same predicate as
    # the full-scan path, so index staleness can hide a schedule at
    # worst (healed by backfill/reconcile), never leak or corrupt one.

    def _get_schedule_jobs(
        self,
        *,
        user_id: str | None,
        graph_id: str | None,
        session_id: str | None,
        kind: str | None,
        organization_id: str | None,
        include_paused: bool = False,
    ) -> list[JobObj]:
        """Jobs to run the ``get_execution_schedules`` predicate over.

        Index-narrowed when a usable index exists and an identity filter
        was given; the full (cached) jobstore scan otherwise.
        """
        full_scan = (
            self._get_jobs_cached if include_paused else self._get_active_jobs_cached
        )
        index = self._schedule_index
        with self._jobs_cache_lock:
            ready = self._schedule_index_ready
            failure_version = self._schedule_index_failure_version
        if index is None or not ready:
            return full_scan()
        try:
            job_ids = index.candidate_job_ids(
                user_id=user_id,
                graph_id=graph_id,
                session_id=session_id,
                kind=kind,
                organization_id=organization_id,
            )
        except Exception:
            logger.warning(
                "Schedule index query failed; falling back to full scan",
                exc_info=True,
            )
            return full_scan()
        if job_ids is None:  # no identity filter — trusted-global listing
            return full_scan()

        jobs: list[JobObj] = []
        dangling: list[str] = []
        for job_id in job_ids:
            try:
                job = self.scheduler.get_job(job_id, jobstore=Jobstores.EXECUTION.value)
            except Exception:
                logger.warning(
                    "Schedule %s could not be loaded; falling back to full scan",
                    job_id,
                    exc_info=True,
                )
                return full_scan()
            if job is None:
                dangling.append(job_id)
                continue
            if job.next_run_time is None and not include_paused:
                continue
            jobs.append(job)
        with self._jobs_cache_lock:
            ready = (
                self._schedule_index_ready
                and self._schedule_index_failure_version == failure_version
            )
        if not ready:
            return full_scan()
        if self.scheduler.running:
            self._delete_schedule_index_rows(dangling)
            self._restore_reused_index_rows(dangling)
        return jobs

    def _upsert_schedule_index_row(
        self,
        job_id: str,
        job_args: GraphExecutionJobArgs | CopilotTurnJobArgs,
    ) -> None:
        """Keep successful mutations visible through full scans after an index failure."""
        if self._schedule_index is None:
            return
        try:
            self._schedule_index.upsert(_index_entry(job_id, job_args))
        except Exception:
            with self._jobs_cache_lock:
                self._schedule_index_ready = False
                self._schedule_index_failure_version += 1
            logger.exception(
                f"Failed to index schedule {job_id}; filtered listings will "
                "use the full jobstore scan until a successful reconcile"
            )

    def _delete_schedule_index_row(self, job_id: str) -> None:
        self._delete_schedule_index_rows([job_id])

    def _delete_schedule_index_rows(self, job_ids: list[str]) -> None:
        if self._schedule_index is None or not job_ids:
            return
        try:
            self._schedule_index.delete_many(job_ids)
        except Exception:
            logger.warning("Failed to remove stale schedule index rows", exc_info=True)

    def _restore_reused_index_rows(self, job_ids: list[str]) -> None:
        """Put back the row of any job set again under one of *job_ids*, the
        rows just dropped for having no job.

        A chat's turn at the budget reset reuses its id, so a new one can be
        set between a listing finding the old job gone and the drop, and the
        drop would hide it from the chat's listings until the next reconcile.
        Every schedule is added before its row is written, so reading the job
        after the drop catches each such row.
        """
        for job_id in job_ids:
            try:
                job = self.scheduler.get_job(job_id, jobstore=Jobstores.EXECUTION.value)
            except Exception:
                logger.warning(
                    "Could not load job %s to check its dropped index row",
                    job_id,
                    exc_info=True,
                )
                continue
            info = _job_to_info(job) if job else None
            if info is not None:
                self._upsert_schedule_index_row(job_id, info)

    def _mark_schedule_index_ready(self, failure_version: int) -> None:
        with self._jobs_cache_lock:
            if self._schedule_index_failure_version == failure_version:
                self._schedule_index_ready = True

    def _reconcile_schedule_index(self) -> None:
        """Rebuild the index from a full jobstore scan (startup + daily)."""
        index = self._schedule_index
        if index is None:
            return
        if not self.scheduler.running:
            # BaseScheduler.shutdown() flips the state outside the jobstore
            # lock, after which get_jobs()/get_job() answer "nothing" for
            # everything. Reconciling on that view would wipe the shared
            # index for every replica.
            logger.warning("Scheduler is not running; skipping index reconcile")
            return
        # Idempotent; repairs a table whose creation failed at startup.
        index.ensure_table()
        with self._jobs_cache_lock:
            failure_version = self._schedule_index_failure_version
        jobs = self.scheduler.get_jobs(jobstore=Jobstores.EXECUTION.value)
        if not jobs and failure_version:
            return
        entries = []
        for job in jobs:
            info = _job_to_info(job)
            if info is None:
                # System/maintenance jobs and corrupted rows aren't
                # user-listable, so they don't get index rows either.
                continue
            entries.append(_index_entry(job.id, info))
        index.upsert_many(entries)

        # Clear rows whose job is gone — but verify each against the live
        # jobstore first: a schedule added while this scan ran is in the
        # index and not in our snapshot, and must survive.
        stale = index.all_job_ids() - {e.job_id for e in entries}
        if not self.scheduler.running or (not jobs and stale):
            # A shutdown raced the scan, or the raw scan came back empty while
            # the index still has rows: neither is evidence the rows are
            # stale, so keep them for the next reconcile. (Test the raw scan,
            # not `entries`: a jobstore holding only system jobs is a
            # legitimate empty listing whose stale rows must still go.)
            logger.warning(
                "Skipping stale-row sweep: scan returned %d jobs against %d "
                "index rows (running=%s)",
                len(jobs),
                len(stale),
                self.scheduler.running,
            )
            self._mark_schedule_index_ready(failure_version)
            return
        confirmed_gone = []
        for job_id in stale:
            try:
                if (
                    self.scheduler.get_job(job_id, jobstore=Jobstores.EXECUTION.value)
                    is None
                ):
                    confirmed_gone.append(job_id)
            except Exception:
                # A row that cannot be reconstituted is not proof the job is
                # gone; leave it for the full scan's own cleanup.
                logger.warning(
                    "Could not load job %s while reconciling the index",
                    job_id,
                    exc_info=True,
                )
        if not self.scheduler.running:
            return
        index.delete_many(confirmed_gone)
        self._restore_reused_index_rows(confirmed_gone)

        self._mark_schedule_index_ready(failure_version)
        logger.info(
            f"Schedule index reconciled: {len(entries)} rows, "
            f"{len(confirmed_gone)} stale rows removed"
        )

    def _start_schedule_index_backfill(self) -> None:
        if self._schedule_index is None:
            return

        def _run():
            try:
                time.sleep(_SCHEDULE_INDEX_SETTLE_S)
                self._reconcile_schedule_index()
            except Exception:
                logger.exception(
                    "Schedule index backfill failed; filtered schedule reads "
                    "will use the full jobstore scan until the next reconcile"
                )

        threading.Thread(target=_run, daemon=True, name="ScheduleIndexBackfill").start()

    @expose
    def execute_reconcile_schedule_index(self):
        """Manually trigger a schedule index rebuild."""
        return reconcile_schedule_index()

    @expose
    def get_execution_schedules(
        self,
        graph_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        kind: str | None = None,
        organization_id: str | None = None,
        team_ids: list[str] | None = None,
        include_paused: bool = False,
    ) -> list[Union[GraphExecutionJobInfo, CopilotTurnJobInfo]]:
        """Return schedules of both kinds, filtered by the given fields.

        *kind* may be ``"graph"``, ``"copilot_turn"``, or ``None`` for all.
        Graph-only filters (*graph_id*) silently skip copilot-turn rows;
        copilot-only filters (*session_id*) silently skip graph rows.

        With *organization_id* (from a membership-verified RequestContext)
        the org/team visibility rules apply instead of strict ownership:
        own schedules + org-home schedules + schedules of teams in
        *team_ids* (resolved by the caller, who has async DB access). Expert
        schedules remain owner-only for scoped calls. Trusted global callers
        that provide neither *user_id* nor *organization_id* receive all jobs.

        Paused jobs (``next_run_time is None``) are hidden by default —
        excluded at the SQL level via ``_get_active_jobs_cached`` rather than
        filtered out afterwards — which is what keeps a fired expert's
        suspended schedules out of every user-facing listing. *include_paused*
        is for the lifecycle callers that have to find them again to resume
        them; that path reads the unfiltered ``_get_jobs_cached`` instead.
        Fired one-shot jobs share the same null marker, so they resurface too
        — filter by kind/id if that matters to the caller.
        """
        jobs: list[JobObj] = self._get_schedule_jobs(
            user_id=user_id,
            graph_id=graph_id,
            session_id=session_id,
            kind=kind,
            organization_id=organization_id,
            include_paused=include_paused,
        )
        results: list[Union[GraphExecutionJobInfo, CopilotTurnJobInfo]] = []
        for job in jobs:
            info = _job_to_info(job)
            if info is None:
                continue
            if kind is not None and info.kind != kind:
                continue
            if info.expert_id is not None:
                if (
                    user_id is not None or organization_id is not None
                ) and info.user_id != user_id:
                    continue
            elif organization_id is not None:
                # GraphExecutionJobArgs defaults organization_id to "" —
                # normalise so untagged rows never match an org clause.
                info_org = info.organization_id or None
                owned = user_id is not None and info.user_id == user_id
                in_org = info_org == organization_id and (
                    info.team_id is None or info.team_id in (team_ids or [])
                )
                if not (owned or in_org):
                    continue
            elif user_id is not None and info.user_id != user_id:
                continue
            if graph_id is not None and (
                not isinstance(info, GraphExecutionJobInfo) or info.graph_id != graph_id
            ):
                continue
            if session_id is not None and (
                not isinstance(info, CopilotTurnJobInfo)
                or info.session_id != session_id
            ):
                continue
            results.append(info)
        return results

    @expose
    def execute_flush_matured_alerts(self):
        flush_matured_alerts()

    @expose
    def execute_send_due_briefings(self):
        send_due_briefings()

    @expose
    def get_parked_jobs(
        self, limit: int | None = _PARKED_SCAN_LIMIT
    ) -> dict[str, list[str]]:
        """Job ids the scheduler could not restore, per jobstore.

        Bounded by default: paused and fired-once rows accumulate forever and
        each one read here is deserialized. Pass ``limit=None`` for the whole
        set, accepting a scan proportional to that backlog.
        """
        return {
            alias: store.get_parked_job_ids(limit=limit)
            for alias, store in self._persistent_jobstores.items()
        }

    @expose
    def reconcile_parked_jobs(self) -> dict[str, list[str]]:
        """Make repaired rows resumable again, per jobstore.

        For a repair made without ``jobstore_backfill``, which already does
        this for the rows it rewrites.
        """
        return {
            alias: store.reconcile_repaired_jobs()
            for alias, store in self._persistent_jobstores.items()
        }

    @expose
    def execute_report_late_executions(self):
        return report_late_executions()

    @expose
    def execute_report_block_error_rates(self):
        return report_block_error_rates()

    @expose
    def execute_report_orphaned_auth_identities(self):
        """Manually trigger the auth-identity invariant check and heal."""
        return report_orphaned_auth_identities()

    @expose
    def execute_cleanup_expired_files(self):
        """Manually trigger cleanup of expired cloud storage files."""
        return cleanup_expired_files()

    @expose
    def execute_cleanup_oauth_tokens(self):
        """Manually trigger cleanup of expired OAuth tokens."""
        return cleanup_oauth_tokens()

    @expose
    def execute_report_execution_accuracy_alerts(self):
        """Manually trigger execution accuracy alert checking."""
        return execution_accuracy_alerts()

    @expose
    def execute_ensure_embeddings_coverage(self):
        """Manually trigger embedding backfill for approved store agents."""
        return ensure_embeddings_coverage()

    # --- Graphiti community detection (P-1.7) ---
    #
    # Rebuilds run weekly at owner-local 04:00 on Mondays
    # (``0 4 * * 0``: APScheduler counts weekdays from 0 = Monday) to avoid
    # the Leiden cost spike during active hours and to stagger from the
    # nightly dream at 03:00.

    @expose
    def add_community_rebuild_schedule(
        self,
        user_id: str,
        user_timezone: str = "UTC",
    ) -> dict:
        """The account's weekly community rebuild: the scope-keyed method
        for ``MemoryScope.for_user(user_id)``, whose job id is unchanged."""
        return self.add_scope_community_rebuild_schedule(
            MemoryScope.for_user(user_id), user_timezone
        )

    @expose
    def add_scope_community_rebuild_schedule(
        self,
        scope: MemoryScope,
        user_timezone: str = "UTC",
    ) -> dict:
        """Register the weekly community rebuild of one memory scope.

        Only adds the job: the registry row that
        lets it fire is ``copilot/dream/registry.py``'s to write.
        """
        user_timezone = user_timezone or "UTC"
        job = self._add_memory_job(
            *_MEMORY_CRONS[COMMUNITY_REBUILD_JOB_PREFIX],
            dream_system_job(COMMUNITY_REBUILD_JOB_PREFIX),
            scope,
            user_timezone,
        )
        return memory_job_result(scope, job.id, job.next_run_time, user_timezone)

    @expose
    def delete_community_rebuild_schedule(self, user_id: str) -> bool:
        """Remove the account's weekly community rebuild."""
        return self._delete_account_memory_job(
            user_id, dream_system_job(COMMUNITY_REBUILD_JOB_PREFIX)
        )

    @expose
    def remove_scope_memory_jobs(self, scope: MemoryScope) -> list[str]:
        """Remove every dream-system cron of one memory scope (``jobs_for``:
        an expert's has no skill-learning cron); returns the job ids removed.
        The ids derive from the scope, so a caller can only ever reach that
        scope's crons. The registry calls this when it pauses or wipes a
        scope."""
        jobs = [
            self.scheduler.get_job(
                job.job_id(scope), jobstore=Jobstores.EXECUTION.value
            )
            for job in jobs_for(scope)
        ]
        removed = [job.id for job in jobs if job is not None and _removed(job)]
        if removed:
            logger.info("Removed the memory crons of %s", scope_label(scope))
        return removed

    def _reconcile_memory_crons(self) -> None:
        """One reconcile pass (``reconcile_memory_scope_crons``), its listing
        of the dream-system crons included, under
        ``MEMORY_CRON_RECONCILE_TIMEOUT_SECONDS``."""
        try:
            report = run_async(
                reconcile(_SchedulerCrons(self)),
                timeout=MEMORY_CRON_RECONCILE_TIMEOUT_SECONDS,
                cancel_on_timeout=True,
            )
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Memory cron reconcile cut off; the next pass carries on after "
                "the last scope it took"
            )
            return
        if report.restored or report.dropped or report.removed or report.failed:
            logger.info(
                "Memory cron reconcile: restored %d, took back %d, removed the "
                "crons of %d scopes, %d failed",
                len(report.restored),
                len(report.dropped),
                len(report.removed),
                len(report.failed),
            )

    def _memory_crons(self) -> list[ListedCron]:
        """Every dream-system cron that names a scope, with its timezone."""
        return [
            cron
            for job in self._memory_jobs()
            if (cron := listed_cron(job)) is not None
        ]

    def _memory_jobs(self) -> list[JobObj]:
        """Every dream-system cron in the EXECUTION jobstore, in one read
        filtered by job-id prefix and stopped by the database after
        ``MEMORY_CRON_LISTING_TIMEOUT_SECONDS``."""
        ids = self._execution_jobstore.jobs_t.c.id
        return self._execution_jobstore._get_jobs(
            or_(
                *(
                    ids.startswith(f"{job.job_id_prefix}_", autoescape=True)
                    for job in DREAM_SYSTEM_JOBS
                )
            ),
            statement_timeout=MEMORY_CRON_LISTING_TIMEOUT_SECONDS,
        )

    def _add_memory_job(
        self,
        func: Callable[..., object],
        crontab: str,
        job: DreamSystemJob,
        scope: MemoryScope,
        user_timezone: str,
        reconciled: bool = False,
    ) -> JobObj:
        """Add, or replace, one dream-system cron of ``scope``; tagged as the
        reconcile's when it registers it (``RECONCILED_SUFFIX``)."""
        tag = RECONCILED_SUFFIX if reconciled else ""
        kwargs: dict[str, object] = {**memory_job_kwargs(scope)}
        if scope.expert_id is not None:
            kwargs["func"] = func
            func = execute_expert_memory_cron
        added = self.scheduler.add_job(
            func,
            kwargs=kwargs,
            trigger=CronTrigger.from_crontab(crontab, timezone=user_timezone),
            id=job.job_id(scope),
            name=f"{job.name} for {scope_label(scope)}{tag}",
            jobstore=Jobstores.EXECUTION.value,
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=_MEMORY_CRON_MISFIRE_GRACE.get(job.job_id_prefix),
        )
        logger.info(
            "Registered %s job %s for %s in tz %s",
            job.name,
            added.id,
            scope_label(scope),
            user_timezone,
        )
        return added

    def _delete_account_memory_job(self, user_id: str, job: DreamSystemJob) -> bool:
        """Remove one of the account's crons, the registry forgetting it
        first: the reconcile restores a cron a row records, and takes back
        only one it registered itself once the row stops recording it, so the
        record goes before the job. A forget that fails keeps the job, and
        says so. The next ``ensure_scope_scheduled`` registers it again."""
        scope = MemoryScope.for_user(user_id)
        existing = self.scheduler.get_job(
            job.job_id(scope), jobstore=Jobstores.EXECUTION.value
        )
        if not existing:
            return False
        if not _forget_dream_registration(scope, job):
            logger.warning(
                "Kept %s job for user %s: the registry could not forget it",
                job.name,
                user_id[:12],
            )
            return False
        _removed(existing)
        logger.info("Removed %s job for user %s", job.name, user_id[:12])
        return True

    def _drop_memory_job(self, job: DreamSystemJob, scope: MemoryScope) -> None:
        """Remove one dream-system cron of ``scope`` if it is there."""
        try:
            self.scheduler.remove_job(
                job.job_id(scope), jobstore=Jobstores.EXECUTION.value
            )
        except JobLookupError:
            pass

    @expose
    def execute_community_rebuild_pass(self, user_id: str, force: bool = False) -> dict:
        """Manually trigger a community rebuild for one user (bypasses cron).

        Set ``force=True`` to bypass the activity gate inside
        ``rebuild_communities_for_user`` — useful for admin debugging
        when you want to rebuild even on an unchanged graph.
        """
        return run_async(
            rebuild_communities_for_user(user_id, force=force),
            timeout=SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS,
        )

    def _register_posthog_lifecycle_sweep(self) -> None:
        """Daily PostHog lifecycle sweep: the safety net behind the webhook and
        signup hooks, and what moves a trial that ran out on the clock on.

        Cron rather than an interval, because replace_existing re-arms an
        interval from every restart. An unchanged job isn't re-registered
        either: replacing it recomputes ``next_run_time``, so a restart after
        04:15 but before the overdue run fires would skip that day's sweep.
        Left in place, the overdue run fires on start (``coalesce``, no
        misfire limit).
        """
        trigger = CronTrigger.from_crontab("15 4 * * *")
        existing = self.scheduler.get_job(
            _POSTHOG_LIFECYCLE_JOB_ID, jobstore=Jobstores.EXECUTION.value
        )
        if existing is not None and str(existing.trigger) == str(trigger):
            return
        self.scheduler.add_job(
            sync_posthog_lifecycles,
            trigger,
            id=_POSTHOG_LIFECYCLE_JOB_ID,
            replace_existing=True,
            max_instances=1,
            jobstore=Jobstores.EXECUTION.value,
        )

    def _register_recording_purge(self) -> None:
        """The brain-dump recording sweep, which keeps "deleted within 48
        hours" by running every ``RECORDING_PURGE_INTERVAL``.

        Registered like the PostHog sweep: replacing an interval job re-arms
        it at start + interval, so restarts closer together than the
        interval would postpone the sweep indefinitely. An unchanged job is
        left with the next run it has, and an overdue run fires on start.
        """
        trigger = IntervalTrigger(
            seconds=int(RECORDING_PURGE_INTERVAL.total_seconds()),
            timezone=timezone.utc,
        )
        existing = self.scheduler.get_job(
            RECORDING_PURGE_JOB_ID, jobstore=Jobstores.EXECUTION.value
        )
        if existing is not None and str(existing.trigger) == str(trigger):
            return
        self.scheduler.add_job(
            purge_expired_recordings_job,
            trigger,
            id=RECORDING_PURGE_JOB_ID,
            replace_existing=True,
            max_instances=1,
            jobstore=Jobstores.EXECUTION.value,
        )

    def _register_memory_cron_reconcile(self) -> None:
        """The memory cron reconcile (``reconcile_memory_scope_crons``), the
        level-triggered backstop of the registry's pause, resume and ensure
        (``copilot/dream/scope_reconcile.py``), every
        ``RECONCILE_INTERVAL_MINUTES``.

        Each process starts with an immediate pass. Keep the job in memory
        so an old scheduler cannot park it during a rolling deploy.
        Overlapping passes replay the same bounded, idempotent work.
        """
        trigger = IntervalTrigger(minutes=RECONCILE_INTERVAL_MINUTES)
        self.scheduler.add_job(
            reconcile_memory_scope_crons,
            trigger,
            id=_MEMORY_CRON_RECONCILE_JOB_ID,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            next_run_time=datetime.now(ZoneInfo("UTC")),
            jobstore=Jobstores.MEMORY_CRON_RECONCILE.value,
        )

    # --- Morning briefing ---
    #
    # Daily per-user cron at user-local 09:00. The job body's flag gate and
    # idempotency (per local calendar date) live inside
    # ``generate_and_deliver_briefing`` itself, so — unlike community
    # rebuild — there's no registration-time flag check here.

    @expose
    def add_morning_briefing_schedule(
        self,
        user_id: str,
        user_timezone: str = "UTC",
    ) -> dict:
        """Register a daily morning briefing for one user.

        Re-registration with an unchanged timezone is a no-op rather than a
        ``replace_existing`` write: APScheduler recomputes ``next_run_time``
        on the replace path, so replacing a job whose 09:00 fire is already
        overdue (e.g. after a scheduler restart) would push it to tomorrow
        and lose that day's briefing entirely.
        """
        if not user_timezone:
            user_timezone = "UTC"

        job_id = f"morning_briefing_{user_id}"
        existing = self.scheduler.get_job(job_id, jobstore=Jobstores.EXECUTION.value)
        if existing is not None and _job_timezone_name(existing) == user_timezone:
            return {
                "id": existing.id,
                "user_id": user_id,
                "user_timezone": user_timezone,
                "next_run_time": (
                    existing.next_run_time.isoformat()
                    if existing.next_run_time
                    else None
                ),
                "skipped": True,
                "reason": "already_registered",
            }

        job = self.scheduler.add_job(
            execute_morning_briefing,
            kwargs={"user_id": user_id},
            trigger=CronTrigger.from_crontab(
                _morning_briefing_crontab(user_id), timezone=user_timezone
            ),
            id=job_id,
            name=f"Morning briefing for {user_id[:12]}",
            jobstore=Jobstores.EXECUTION.value,
            replace_existing=True,
            max_instances=1,
        )
        logger.info(
            "Registered morning briefing job %s for user %s in tz %s",
            job.id,
            user_id[:12],
            user_timezone,
        )
        return {
            "id": job.id,
            "user_id": user_id,
            "user_timezone": user_timezone,
            "next_run_time": (
                job.next_run_time.isoformat() if job.next_run_time else None
            ),
        }

    @expose
    def remove_morning_briefing_schedule(self, user_id: str) -> dict:
        """Delete one user's morning-briefing cron.

        Deliberately not routed through ``delete_graph_execution_schedule``:
        that path authorizes via ``_job_to_info``, which parses only graph and
        copilot-turn kwargs and would reject this job's ``{"user_id": ...}``
        shape as corrupt. The job id embeds the user id, so it is self-scoping.
        """
        job_id = f"morning_briefing_{user_id}"
        job = self.scheduler.get_job(job_id, jobstore=Jobstores.EXECUTION.value)
        if job is None:
            return {"id": job_id, "user_id": user_id, "removed": False}
        job.remove()
        self._invalidate_jobs_cache()
        logger.info(f"Removed morning briefing job {job_id} for user {user_id[:12]}")
        return {"id": job_id, "user_id": user_id, "removed": True}

    # --- Dream nightly batch (P-0.2 + P-0.4 consolidated) ---
    #
    # ONE per-user cron at user-local 03:00 fans out all batch-family
    # work (dream pass, ratification supersession sweep, plus future
    # P2 / P3 / P4 / P11 submitters). The separate dream-pass and
    # ratification-pass crons that previously existed are gone — both
    # are now submitters inside this nightly cron's body.
    #
    # In real-batch mode this cron returns in seconds (each submitter
    # just enqueues to the provider's batch API; results land hours
    # later via the ``copilot_batch_executor`` poller). In
    # sync_baseline mode (today, until P0.1 adapter stubs are filled
    # in) each submitter inlines its work.
    #
    # ``execute_ratification_pass_now`` remains as an admin-callable
    # sync endpoint because ratification is Cypher-only and finishes in
    # seconds (no batch SLA to amortise). The dream pass + nightly fan-
    # out moved to fire-and-forget + JobStatus polling — see
    # ``schedule_immediate_*`` below — so their old sync ``_now``
    # methods were removed in Step 9 of the providers rollout.

    @expose
    def add_nightly_batch_schedule(
        self,
        user_id: str,
        user_timezone: str = "UTC",
    ) -> dict:
        """The account's nightly batch cron: the scope-keyed method for
        ``MemoryScope.for_user(user_id)``, whose job id is unchanged."""
        return self.add_scope_nightly_batch_schedule(
            MemoryScope.for_user(user_id), user_timezone
        )

    @expose
    def add_scope_nightly_batch_schedule(
        self,
        scope: MemoryScope,
        user_timezone: str = "UTC",
    ) -> dict:
        """Register the nightly batch-family fan-out cron of one memory scope.

        Gated by ``Flag.DREAM_PASS_ENABLED`` for the scope's owner. When
        the flag is off the call is a no-op — returns a structured
        "skipped" dict matching the shape of a successful registration.
        Defense-in-depth: the registry in ``copilot/dream/registry.py``
        already gates this, but direct callers (admin endpoint, ad-hoc
        scripts) bypass it. Only adds the job: the registry row that lets
        it fire is the registry's to write.
        """
        from backend.util.feature_flag import Flag, is_feature_enabled

        user_timezone = user_timezone or "UTC"
        if not run_async(
            is_feature_enabled(Flag.DREAM_PASS_ENABLED, scope.owner_user_id)
        ):
            logger.info(
                "Nightly batch registration skipped for %s — "
                "DREAM_PASS_ENABLED flag is off.",
                scope_label(scope),
            )
            return memory_job_skipped(scope, user_timezone, "dream_pass_disabled")
        job = self._add_memory_job(
            *_MEMORY_CRONS[NIGHTLY_BATCH_JOB_PREFIX],
            dream_system_job(NIGHTLY_BATCH_JOB_PREFIX),
            scope,
            user_timezone,
        )
        return memory_job_result(scope, job.id, job.next_run_time, user_timezone)

    @expose
    def delete_nightly_batch_schedule(self, user_id: str) -> bool:
        """Remove the account's nightly batch cron."""
        return self._delete_account_memory_job(
            user_id, dream_system_job(NIGHTLY_BATCH_JOB_PREFIX)
        )

    @expose
    def add_skill_learning_schedule(
        self,
        user_id: str,
        user_timezone: str = "UTC",
    ) -> dict:
        """The account's nightly skill-learning cron: the scope-keyed method
        for ``MemoryScope.for_user(user_id)``, whose job id is unchanged."""
        return self.add_scope_skill_learning_schedule(
            MemoryScope.for_user(user_id), user_timezone
        )

    @expose
    def add_scope_skill_learning_schedule(
        self,
        scope: MemoryScope,
        user_timezone: str = "UTC",
    ) -> dict:
        """Register the nightly skill-learning cron of an account's scope.

        Gated by ``Flag.DREAM_SKILL_LEARNING_ENABLED`` for the owner,
        independent of the dream pass gate. An expert's scope has no such
        cron (one pass reviews the account's skills and its experts'), so it
        is refused. Fires at 03:00 owner-local. A cron already registered in
        that timezone is kept as it is, its next run included, so an overdue
        run is not pushed back a day, and reported as registered. A missed
        run is caught up only the same morning
        (``_MEMORY_CRON_MISFIRE_GRACE``): at most one run, and the pass
        re-checks every source's eligibility, so a stale proposal is never
        applied by a catch-up run. Only adds the job: the registry row that
        lets it fire is the registry's to write.
        """
        from backend.util.feature_flag import Flag, is_feature_enabled

        user_timezone = user_timezone or "UTC"
        if scope.is_expert:
            return memory_job_skipped(scope, user_timezone, "account_only")
        if not run_async(
            is_feature_enabled(Flag.DREAM_SKILL_LEARNING_ENABLED, scope.owner_user_id)
        ):
            logger.info(
                "Skill learning registration skipped for %s — "
                "DREAM_SKILL_LEARNING_ENABLED flag is off.",
                scope_label(scope),
            )
            return memory_job_skipped(scope, user_timezone, "skill_learning_disabled")
        job = dream_system_job(SKILL_LEARNING_JOB_PREFIX)
        existing = self.scheduler.get_job(
            job.job_id(scope), jobstore=Jobstores.EXECUTION.value
        )
        if existing is not None and _job_timezone_name(existing) == user_timezone:
            return memory_job_result(
                scope, existing.id, existing.next_run_time, user_timezone
            )
        added = self._add_memory_job(
            *_MEMORY_CRONS[SKILL_LEARNING_JOB_PREFIX], job, scope, user_timezone
        )
        return memory_job_result(scope, added.id, added.next_run_time, user_timezone)

    @expose
    def delete_skill_learning_schedule(self, user_id: str) -> bool:
        """Remove the account's nightly skill-learning cron."""
        return self._delete_account_memory_job(
            user_id, dream_system_job(SKILL_LEARNING_JOB_PREFIX)
        )

    @expose
    def execute_skill_learning_now(self, user_id: str) -> dict:
        """Run one skill-learning pass synchronously (owner/admin trigger)."""
        from backend.copilot.learning.nightly import run_skill_learning_pass

        result = run_async(
            run_skill_learning_pass(user_id, trigger="admin"),
            timeout=SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS,
        )
        return result.model_dump(mode="json")

    @expose
    def schedule_held_wake(
        self, user_id: str, session_id: str, delay_seconds: float
    ) -> None:
        """Wake the chat's answered cards in *delay_seconds*. Keyed by chat, so
        each later answer moves the one job, and stored, so it runs after a
        restart however late."""
        self.scheduler.add_job(
            execute_held_wake,
            kwargs={"user_id": user_id, "session_id": session_id},
            trigger="date",
            run_date=datetime.now(timezone.utc) + timedelta(seconds=delay_seconds),
            id=f"held_wake_{session_id}",
            jobstore=Jobstores.EXECUTION.value,
            replace_existing=True,
        )

    # ---- Fire-and-forget admin triggers (JobStatus-aware) -------------------
    #
    # The ``schedule_immediate_*`` methods schedule the matching
    # ``*_with_status`` wrapper to run on the APScheduler thread pool
    # at the current time, then return immediately. The admin route
    # writes the initial ``state="queued"`` row before calling these;
    # the wrapper writes ``running`` / phase transitions / ``complete``
    # / ``errored``. The admin frontend polls a GET endpoint to render
    # progress without holding an HTTP connection open for minutes.

    @expose
    def schedule_immediate_nightly_batch(self, user_id: str, job_id: str) -> dict:
        """Schedule a one-shot nightly batch run keyed by ``job_id``.

        The admin endpoint that calls this has already written an
        initial ``state="queued"`` JobStatus row, so this method just
        adds the APScheduler job and returns immediately. The wrapper
        body is responsible for advancing the status row.
        """
        import datetime as _dt

        scheduler_job_id = f"adhoc_nightly_{job_id}"
        self.scheduler.add_job(
            execute_nightly_batch_with_status,
            kwargs={"user_id": user_id, "job_id": job_id},
            trigger="date",
            run_date=_dt.datetime.now(_dt.timezone.utc),
            id=scheduler_job_id,
            jobstore=Jobstores.EXECUTION.value,
            replace_existing=False,
            max_instances=1,
        )
        logger.info(
            "Scheduled immediate nightly batch %s for user %s",
            job_id[:12],
            user_id[:12],
        )
        return {"scheduled": True, "job_id": job_id, "kind": "nightly"}

    @expose
    def schedule_immediate_dream_pass(
        self, user_id: str, job_id: str, force: bool = False
    ) -> dict[str, Any]:
        """Schedule a one-shot dream pass run keyed by ``job_id``; ``force``
        goes to the pass's guard (see ``execute_dream_pass_with_status``)."""
        import datetime as _dt

        scheduler_job_id = f"adhoc_dream_{job_id}"
        self.scheduler.add_job(
            execute_dream_pass_with_status,
            kwargs={"user_id": user_id, "job_id": job_id, "force": force},
            trigger="date",
            run_date=_dt.datetime.now(_dt.timezone.utc),
            id=scheduler_job_id,
            jobstore=Jobstores.EXECUTION.value,
            replace_existing=False,
            max_instances=1,
        )
        logger.info(
            "Scheduled immediate dream pass %s for user %s",
            job_id[:12],
            user_id[:12],
        )
        return {"scheduled": True, "job_id": job_id, "kind": "dream_pass"}

    @expose
    def schedule_immediate_community_rebuild(self, user_id: str, job_id: str) -> dict:
        """Schedule a one-shot community rebuild run keyed by ``job_id``."""
        import datetime as _dt

        scheduler_job_id = f"adhoc_rebuild_{job_id}"
        self.scheduler.add_job(
            execute_community_rebuild_with_status,
            kwargs={"user_id": user_id, "job_id": job_id},
            trigger="date",
            run_date=_dt.datetime.now(_dt.timezone.utc),
            id=scheduler_job_id,
            jobstore=Jobstores.EXECUTION.value,
            replace_existing=False,
            max_instances=1,
        )
        logger.info(
            "Scheduled immediate community rebuild %s for user %s",
            job_id[:12],
            user_id[:12],
        )
        return {"scheduled": True, "job_id": job_id, "kind": "rebuild"}

    @expose
    def execute_ratification_pass_now(self, user_id: str) -> dict:
        """Run the ratification sweep in isolation (admin debug endpoint).

        Bypasses the nightly batch cron and runs only the ratification
        supersession sweep. Useful for testing ratification logic
        without waiting for / running the full nightly fan-out.
        """
        from backend.copilot.dream.ratification import run_ratification_pass

        result = run_async(run_ratification_pass(user_id))
        return result.model_dump(mode="json")


class _JobList:
    """One cached job list, when it expires, and the fetch replacing it."""

    def __init__(self) -> None:
        self.jobs: list[JobObj] | None = None
        self.expires_at = 0.0
        self.refresh: Future[list[JobObj]] | None = None


def _log_refresh_failure(refresh: "Future[list[JobObj]]") -> None:
    if (error := refresh.exception()) is not None:
        logger.warning(
            "Background schedule list refresh failed; serving the expired list",
            exc_info=error,
        )


class SchedulerClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return Scheduler

    add_execution_schedule = endpoint_to_async(Scheduler.add_graph_execution_schedule)
    add_copilot_turn_schedule = endpoint_to_async(Scheduler.add_copilot_turn_schedule)
    cancel_at_reset_turn = endpoint_to_async(Scheduler.cancel_at_reset_turn)
    delete_schedule = endpoint_to_async(Scheduler.delete_graph_execution_schedule)
    update_execution_schedule = endpoint_to_async(
        Scheduler.update_graph_execution_schedule
    )
    pause_schedule = endpoint_to_async(Scheduler.pause_execution_schedule)
    resume_schedule = endpoint_to_async(Scheduler.resume_execution_schedule)
    update_copilot_turn_cron = endpoint_to_async(Scheduler.update_copilot_turn_cron)
    # Graph-only typed list — for legacy callers that need GraphExecutionJobInfo.
    get_graph_execution_schedules = endpoint_to_async(
        Scheduler.get_graph_execution_schedules
    )
    # Polymorphic list — preferred for new callers; returns both kinds.
    get_execution_schedules = endpoint_to_async(Scheduler.get_execution_schedules)

    get_parked_jobs = endpoint_to_async(Scheduler.get_parked_jobs)
    reconcile_parked_jobs = endpoint_to_async(Scheduler.reconcile_parked_jobs)

    add_community_rebuild_schedule = endpoint_to_async(
        Scheduler.add_community_rebuild_schedule
    )
    add_scope_community_rebuild_schedule = endpoint_to_async(
        Scheduler.add_scope_community_rebuild_schedule
    )
    delete_community_rebuild_schedule = endpoint_to_async(
        Scheduler.delete_community_rebuild_schedule
    )
    remove_scope_memory_jobs = endpoint_to_async(Scheduler.remove_scope_memory_jobs)
    execute_community_rebuild_pass = endpoint_to_async(
        Scheduler.execute_community_rebuild_pass
    )

    add_morning_briefing_schedule = endpoint_to_async(
        Scheduler.add_morning_briefing_schedule
    )
    remove_morning_briefing_schedule = endpoint_to_async(
        Scheduler.remove_morning_briefing_schedule
    )

    add_nightly_batch_schedule = endpoint_to_async(Scheduler.add_nightly_batch_schedule)
    add_scope_nightly_batch_schedule = endpoint_to_async(
        Scheduler.add_scope_nightly_batch_schedule
    )
    delete_nightly_batch_schedule = endpoint_to_async(
        Scheduler.delete_nightly_batch_schedule
    )
    add_skill_learning_schedule = endpoint_to_async(
        Scheduler.add_skill_learning_schedule
    )
    add_scope_skill_learning_schedule = endpoint_to_async(
        Scheduler.add_scope_skill_learning_schedule
    )
    delete_skill_learning_schedule = endpoint_to_async(
        Scheduler.delete_skill_learning_schedule
    )
    execute_skill_learning_now = endpoint_to_async(Scheduler.execute_skill_learning_now)

    # Ratification stays synchronous — Cypher-only, finishes in seconds.
    # The admin viz "Ratification" button still calls this directly.
    execute_ratification_pass_now = endpoint_to_async(
        Scheduler.execute_ratification_pass_now
    )

    # Fire-and-forget admin triggers — the admin route writes an
    # initial JobStatus row, calls these, and returns 202 immediately.
    # Frontend polls the JobStatus GET endpoint to render progress.
    schedule_immediate_nightly_batch = endpoint_to_async(
        Scheduler.schedule_immediate_nightly_batch
    )
    schedule_immediate_dream_pass = endpoint_to_async(
        Scheduler.schedule_immediate_dream_pass
    )
    schedule_immediate_community_rebuild = endpoint_to_async(
        Scheduler.schedule_immediate_community_rebuild
    )
    schedule_held_wake = endpoint_to_async(Scheduler.schedule_held_wake)
