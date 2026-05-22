import asyncio
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Literal, Optional, Union
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MAX_INSTANCES,
    EVENT_JOB_MISSED,
)
from apscheduler.job import Job as JobObj
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.util import ZoneInfo
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import MetaData, create_engine

from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.executor.utils import schedule_turn
from backend.copilot.model import get_chat_session
from backend.copilot.optimize_blocks import optimize_block_descriptions
from backend.data.execution import GraphExecutionWithNodes
from backend.data.model import CredentialsMetaInput, GraphInput
from backend.executor import utils as execution_utils
from backend.monitoring import (
    NotificationJobArgs,
    process_existing_batches,
    process_weekly_summary,
    report_block_error_rates,
    report_execution_accuracy_alerts,
    report_late_executions,
)
from backend.util.clients import (
    get_database_manager_async_client,
    get_database_manager_client,
    get_scheduler_client,
)
from backend.util.cloud_storage import cleanup_expired_files_async
from backend.util.exceptions import (
    GraphNotFoundError,
    GraphNotInLibraryError,
    GraphValidationError,
    NotAuthorizedError,
    NotFoundError,
)
from backend.util.logging import PrefixFilter
from backend.util.retry import func_retry
from backend.util.service import (
    AppService,
    AppServiceClient,
    UnhealthyServiceError,
    endpoint_to_async,
    expose,
)
from backend.util.settings import Config


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


@func_retry
def get_event_loop():
    """Get the shared event loop."""
    if _event_loop is None:
        raise RuntimeError("Event loop not initialized. Scheduler not started.")
    return _event_loop


def run_async(coro, timeout: float = SCHEDULER_OPERATION_TIMEOUT_SECONDS):
    """Run a coroutine in the shared event loop and wait for completion."""
    loop = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
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
        )
        await db.increment_onboarding_runs(args.user_id)
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
    except Exception as e:
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.error(
            f"Error executing graph {args.graph_id} after {elapsed:.2f}s: "
            f"{type(e).__name__}: {e}"
        )


def execute_copilot_turn(**kwargs):
    """Enqueue a copilot turn on the executor queue and wait for completion.

    APScheduler dispatches scheduled copilot-turn jobs here. ``kwargs``
    is the serialized form of :class:`CopilotTurnJobArgs`.
    """
    run_async(_execute_copilot_turn(**kwargs))


async def _execute_copilot_turn(**kwargs):
    args = CopilotTurnJobArgs(**kwargs)
    start_time = asyncio.get_event_loop().time()
    try:
        # Verify the session still exists — if the user deleted their
        # conversation between scheduling and now, dispatching would create
        # an orphan turn that no UI ever surfaces. Self-clean the dead
        # schedule instead.
        session = await get_chat_session(args.session_id, args.user_id)
        if session is None:
            logger.info(
                f"Copilot turn schedule {args.schedule_id} skipped — session "
                f"{args.session_id[:12]} no longer exists; removing schedule"
            )
            await _self_delete_copilot_turn_schedule(args)
            return

        # `schedule_turn` (not raw `enqueue_copilot_turn`) is the right entry
        # point: it acquires a per-user concurrency slot AND registers the
        # session in the stream registry before queue-publishing, so the
        # executor's streamed output reaches a known consumer instead of
        # being orphaned.
        await schedule_turn(
            session_id=args.session_id,
            user_id=args.user_id,
            turn_id=str(uuid.uuid4()),
            message=args.message,
            tool_call_id="scheduled_followup",
            tool_name="schedule_followup",
        )
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(
            f"Dispatched scheduled copilot turn for session "
            f"{args.session_id[:12]} (took {elapsed:.2f}s)"
        )
    except ConcurrentTurnLimitError as e:
        # User is at their per-user concurrency cap. For cron schedules the
        # next tick retries automatically; for one-shot (run_at) schedules
        # APScheduler removes the job after fire regardless of our error,
        # so re-schedule a copy 5 min out to give the slot time to free.
        logger.warning(
            f"Scheduled copilot turn for session {args.session_id[:12]} "
            f"hit concurrency cap; cron={args.cron is not None}: {e}"
        )
        if args.run_at is not None:
            await _reschedule_one_shot_after_cap(args)
    except Exception as e:
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.error(
            f"Error dispatching copilot turn for session "
            f"{args.session_id[:12]} after {elapsed:.2f}s: "
            f"{type(e).__name__}: {e}"
        )


# One-shot schedules that hit the per-user concurrency cap are pushed out
# this many seconds and retried at most _MAX_CAP_RETRIES times. We don't
# loop indefinitely because the original use case ("check CI in 20 min")
# is time-sensitive — a multi-hour delay defeats the purpose.
_CONCURRENCY_RETRY_DELAY_SECONDS = 300
_MAX_CAP_RETRIES = 1


async def _reschedule_one_shot_after_cap(args: "CopilotTurnJobArgs") -> None:
    """Re-create a one-shot copilot-turn schedule a few minutes in the future.

    Best-effort: failures are logged. Schedules that have already been
    retried ``_MAX_CAP_RETRIES`` times are dropped to avoid loops. The
    retry depth is tracked in ``CopilotTurnJobArgs.cap_retry_count``
    (which round-trips through APScheduler's persisted kwargs).
    """
    if args.cap_retry_count >= _MAX_CAP_RETRIES:
        logger.error(
            f"Dropping one-shot copilot turn for session "
            f"{args.session_id[:12]} — exhausted {_MAX_CAP_RETRIES} "
            f"concurrency-cap retry/retries"
        )
        return
    try:
        new_run_at = datetime.now(tz=timezone.utc) + timedelta(
            seconds=_CONCURRENCY_RETRY_DELAY_SECONDS
        )
        await get_scheduler_client().add_copilot_turn_schedule(
            user_id=args.user_id,
            session_id=args.session_id,
            message=args.message,
            run_at=new_run_at,
            name=f"{args.schedule_id or 'copilot'}-cap-retry",
            cap_retry_count=args.cap_retry_count + 1,
        )
        logger.info(
            f"Rescheduled one-shot copilot turn for session "
            f"{args.session_id[:12]} to {new_run_at.isoformat()} after "
            f"concurrency cap (retry {args.cap_retry_count + 1}/"
            f"{_MAX_CAP_RETRIES})"
        )
    except Exception:
        logger.warning(
            f"Failed to reschedule capped one-shot copilot turn for "
            f"session {args.session_id[:12]}",
            exc_info=True,
        )


async def _best_effort_unschedule(
    schedule_id: str | None, user_id: str, *, reason: str
) -> None:
    """Self-delete a schedule whose firing condition is no longer satisfiable
    (graph deleted, session deleted, validation failure, etc.).

    Best-effort: failures are logged and swallowed. For recurring schedules
    the next cron tick will re-attempt the cleanup; for one-shot schedules
    APScheduler removes the job after fire anyway, so a missed delete
    here doesn't accumulate orphans indefinitely.
    """
    if not schedule_id:
        logger.warning(
            f"Cannot unschedule (reason: {reason}) — no schedule_id "
            f"available; this is an old job, remove manually"
        )
        return
    try:
        await get_scheduler_client().delete_schedule(
            schedule_id=schedule_id, user_id=user_id
        )
        logger.info(f"Unscheduled job {schedule_id} (reason: {reason})")
    except Exception:
        logger.warning(
            f"Failed to unschedule job {schedule_id} (reason: {reason})",
            exc_info=True,
        )


async def _self_delete_copilot_turn_schedule(args: "CopilotTurnJobArgs") -> None:
    """Convenience wrapper for copilot-turn schedules whose target session is gone."""
    await _best_effort_unschedule(
        args.schedule_id, args.user_id, reason="session deleted"
    )


async def _handle_graph_validation_error(args: "GraphExecutionJobArgs") -> None:
    logger.warning(
        f"Scheduled Graph {args.graph_id} failed validation. Unscheduling graph"
    )
    await _best_effort_unschedule(
        args.schedule_id,
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


def cleanup_expired_files():
    """Clean up expired files from cloud storage."""
    # Wait for completion
    run_async(cleanup_expired_files_async())


def cleanup_oauth_tokens():
    """Clean up expired OAuth tokens from the database."""

    # Wait for completion
    async def _cleanup():
        db = get_database_manager_async_client()
        return await db.cleanup_expired_oauth_tokens()

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


def execution_accuracy_alerts():
    """Check execution accuracy and send alerts if drops are detected."""
    return report_execution_accuracy_alerts()


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


class Jobstores(Enum):
    EXECUTION = "execution"
    BATCHED_NOTIFICATIONS = "batched_notifications"
    WEEKLY_NOTIFICATIONS = "weekly_notifications"


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


class CopilotTurnJobArgs(BaseModel):
    kind: Literal["copilot_turn"] = "copilot_turn"
    schedule_id: str | None = None
    user_id: str
    session_id: str
    message: str
    cron: str | None = None
    run_at: datetime | None = None
    # Set by ``_reschedule_one_shot_after_cap`` when re-creating a one-shot
    # schedule after a concurrency-cap miss. Bounds the retry depth so a
    # persistently-capped user can't loop forever.
    cap_retry_count: int = 0


def _timezone_from_job(job_obj: JobObj) -> str:
    if hasattr(job_obj.trigger, "timezone"):
        return str(job_obj.trigger.timezone)
    return "UTC"


def _next_run_time_iso(job_obj: JobObj) -> str:
    """Render APScheduler's next_run_time. Returns "" for jobs already fired
    (one-shot DateTrigger jobs have ``next_run_time=None`` post-fire)."""
    return job_obj.next_run_time.isoformat() if job_obj.next_run_time else ""


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


def _build_trigger(
    *, cron: str | None, run_at: datetime | None, user_timezone: str
) -> Union[CronTrigger, DateTrigger]:
    if (cron is None) == (run_at is None):
        raise ValueError("Exactly one of `cron` or `run_at` must be provided")
    if cron is not None:
        return CronTrigger.from_crontab(cron, timezone=user_timezone)
    return DateTrigger(run_date=run_at, timezone=user_timezone)


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


class NotificationJobInfo(NotificationJobArgs):
    id: str
    name: str
    next_run_time: str

    @staticmethod
    def from_db(
        job_args: NotificationJobArgs, job_obj: JobObj
    ) -> "NotificationJobInfo":
        return NotificationJobInfo(
            id=job_obj.id,
            name=job_obj.name,
            next_run_time=job_obj.next_run_time.isoformat(),
            **job_args.model_dump(),
        )


class Scheduler(AppService):
    scheduler: BackgroundScheduler

    def __init__(self, register_system_tasks: bool = True):
        self.register_system_tasks = register_system_tasks

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

        db_schema, db_url = _extract_schema_from_url(os.getenv("DIRECT_URL"))
        # Configure executors to limit concurrency without skipping jobs
        from apscheduler.executors.pool import ThreadPoolExecutor

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
                Jobstores.EXECUTION.value: SQLAlchemyJobStore(
                    engine=create_engine(
                        url=db_url,
                        pool_size=self.db_pool_size(),
                        max_overflow=0,
                    ),
                    metadata=MetaData(schema=db_schema),
                    # this one is pre-existing so it keeps the
                    # default table name.
                    tablename="apscheduler_jobs",
                ),
                Jobstores.BATCHED_NOTIFICATIONS.value: SQLAlchemyJobStore(
                    engine=create_engine(
                        url=db_url,
                        pool_size=self.db_pool_size(),
                        max_overflow=0,
                    ),
                    metadata=MetaData(schema=db_schema),
                    tablename="apscheduler_jobs_batched_notifications",
                ),
                # These don't really need persistence
                Jobstores.WEEKLY_NOTIFICATIONS.value: MemoryJobStore(),
            },
            logger=apscheduler_logger,
            timezone=ZoneInfo("UTC"),
        )

        if self.register_system_tasks:
            # Notification PROCESS WEEKLY SUMMARY
            # Runs every Monday at 9 AM UTC
            self.scheduler.add_job(
                process_weekly_summary,
                CronTrigger.from_crontab("0 9 * * 1"),
                id="process_weekly_summary",
                kwargs={},
                replace_existing=True,
                jobstore=Jobstores.WEEKLY_NOTIFICATIONS.value,
            )

            # Notification PROCESS EXISTING BATCHES
            # self.scheduler.add_job(
            #     process_existing_batches,
            #     id="process_existing_batches",
            #     CronTrigger.from_crontab("0 12 * * 5"),
            #     replace_existing=True,
            #     jobstore=Jobstores.BATCHED_NOTIFICATIONS.value,
            # )

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
                replace_existing=True,
                max_instances=1,  # Prevent overlapping runs
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

        self.scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self.scheduler.add_listener(job_missed_listener, EVENT_JOB_MISSED)
        self.scheduler.add_listener(job_max_instances_listener, EVENT_JOB_MAX_INSTANCES)
        self.scheduler.start()

        # Run embedding backfill immediately on startup
        # This ensures blocks/docs are searchable right away, not after 6 hours
        # Safe to run on multiple pods - uses upserts and checks for existing embeddings
        if self.register_system_tasks:
            logger.info("Running embedding backfill on startup...")
            try:
                result = ensure_embeddings_coverage()
                logger.info(f"Startup embedding backfill complete: {result}")
            except Exception as e:
                logger.error(f"Startup embedding backfill failed: {e}")
                # Don't fail startup - the scheduled job will retry later

        # Keep the service running since BackgroundScheduler doesn't block
        super().run_service()

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

        super().cleanup()

    def _persist_schedule(
        self,
        *,
        dispatch_func,
        job_args: Union[GraphExecutionJobArgs, CopilotTurnJobArgs],
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
        return self.scheduler.add_job(
            dispatch_func,
            kwargs=job_args.model_dump(mode="json"),
            name=name,
            trigger=trigger,
            jobstore=Jobstores.EXECUTION.value,
            replace_existing=True,
            id=job_args.schedule_id,
        )

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
    ) -> GraphExecutionJobInfo:
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
        return GraphExecutionJobInfo.from_db(job_args, job)

    @expose
    def add_copilot_turn_schedule(
        self,
        user_id: str,
        session_id: str,
        message: str,
        cron: str | None = None,
        run_at: datetime | None = None,
        name: Optional[str] = None,
        user_timezone: str | None = None,
        cap_retry_count: int = 0,
    ) -> CopilotTurnJobInfo:
        """Schedule a copilot turn to resume *session_id* at a future time.

        At fire time, the scheduler enqueues a copilot turn carrying
        *message* against the existing session, so the copilot resumes
        with the conversation history intact.

        *cap_retry_count* is set internally by
        ``_reschedule_one_shot_after_cap`` to bound the retry depth on
        concurrency-cap misses; normal callers should leave it at 0.
        """
        user_timezone = _resolve_timezone(user_timezone, user_id)
        trigger = _build_trigger(cron=cron, run_at=run_at, user_timezone=user_timezone)
        job_args = CopilotTurnJobArgs(
            schedule_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            message=message,
            cron=cron,
            run_at=run_at,
            cap_retry_count=cap_retry_count,
        )
        job = self._persist_schedule(
            dispatch_func=execute_copilot_turn,
            job_args=job_args,
            trigger=trigger,
            name=name or f"copilot turn (session {session_id[:8]})",
        )
        logger.info(
            f"Added copilot-turn job {job.id} ({trigger.__class__.__name__}) "
            f"for session {session_id[:12]} in timezone {user_timezone}"
        )
        return CopilotTurnJobInfo.from_db(job_args, job)

    @expose
    def delete_graph_execution_schedule(
        self, schedule_id: str, user_id: str
    ) -> Union[GraphExecutionJobInfo, CopilotTurnJobInfo]:
        """Delete a schedule by id, regardless of kind.

        Endpoint name kept for wire back-compat; the client binding is
        ``SchedulerClient.delete_schedule`` and accepts both graph and
        copilot-turn schedules.
        """
        job = self.scheduler.get_job(schedule_id, jobstore=Jobstores.EXECUTION.value)
        if not job:
            raise NotFoundError(f"Job #{schedule_id} not found.")

        info = _job_to_info(job)
        if info is None:
            # kwargs parse as neither graph nor copilot-turn — we have no
            # `user_id` field to authorize against, so refuse the delete.
            # Removing without an ownership check would let any caller who
            # can guess a schedule_id wipe corrupted rows. Surface 404 so
            # the caller can't probe for shape via timing either.
            logger.warning(
                f"Refusing delete for job {schedule_id} with unrecognized "
                f"kwargs shape (no parseable user_id to authorize against)"
            )
            raise NotFoundError(f"Job #{schedule_id} has invalid schedule data.")

        if info.user_id != user_id:
            raise NotAuthorizedError("User ID does not match the job's user ID")

        logger.info(f"Deleting job {schedule_id} (kind={info.kind})")
        job.remove()
        return info

    @expose
    def get_graph_execution_schedules(
        self, graph_id: str | None = None, user_id: str | None = None
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
                graph_id=graph_id, user_id=user_id, kind="graph"
            )
            if isinstance(info, GraphExecutionJobInfo)
        ]

    @expose
    def get_execution_schedules(
        self,
        graph_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        kind: str | None = None,
    ) -> list[Union[GraphExecutionJobInfo, CopilotTurnJobInfo]]:
        """Return schedules of both kinds, filtered by the given fields.

        *kind* may be ``"graph"``, ``"copilot_turn"``, or ``None`` for all.
        Graph-only filters (*graph_id*) silently skip copilot-turn rows;
        copilot-only filters (*session_id*) silently skip graph rows.
        """
        jobs: list[JobObj] = self.scheduler.get_jobs(jobstore=Jobstores.EXECUTION.value)
        results: list[Union[GraphExecutionJobInfo, CopilotTurnJobInfo]] = []
        for job in jobs:
            info = _job_to_info(job) if job.next_run_time is not None else None
            if info is None:
                continue
            if kind is not None and info.kind != kind:
                continue
            if user_id is not None and info.user_id != user_id:
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
    def execute_process_existing_batches(self, kwargs: dict):
        process_existing_batches(**kwargs)

    @expose
    def execute_process_weekly_summary(self):
        process_weekly_summary()

    @expose
    def execute_report_late_executions(self):
        return report_late_executions()

    @expose
    def execute_report_block_error_rates(self):
        return report_block_error_rates()

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


class SchedulerClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return Scheduler

    add_execution_schedule = endpoint_to_async(Scheduler.add_graph_execution_schedule)
    add_copilot_turn_schedule = endpoint_to_async(Scheduler.add_copilot_turn_schedule)
    delete_schedule = endpoint_to_async(Scheduler.delete_graph_execution_schedule)
    # Graph-only typed list — for legacy callers that need GraphExecutionJobInfo.
    get_graph_execution_schedules = endpoint_to_async(
        Scheduler.get_graph_execution_schedules
    )
    # Polymorphic list — preferred for new callers; returns both kinds.
    get_execution_schedules = endpoint_to_async(Scheduler.get_execution_schedules)
