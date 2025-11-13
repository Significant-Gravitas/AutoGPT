import asyncio
import logging
import os
import threading
from enum import Enum
from typing import Optional
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
from apscheduler.util import ZoneInfo
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import MetaData, create_engine

from backend.data.block import BlockInput
from backend.data.execution import GraphExecutionWithNodes
from backend.data.model import CredentialsMetaInput
from backend.executor import utils as execution_utils
from backend.monitoring import (
    NotificationJobArgs,
    process_existing_batches,
    process_weekly_summary,
    report_block_error_rates,
    report_late_executions,
)
from backend.util.clients import get_scheduler_client
from backend.util.cloud_storage import cleanup_expired_files_async
from backend.util.exceptions import (
    GraphNotInLibraryError,
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
        logger.error(
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
        logger.error(f"Async operation failed: {type(e).__name__}: {e}")
        raise


def execute_graph(**kwargs):
    """Execute graph in the shared event loop and wait for completion."""
    # Wait for completion to ensure job doesn't exit prematurely
    run_async(_execute_graph(**kwargs))


async def _execute_graph(**kwargs):
    args = GraphExecutionJobArgs(**kwargs)
    start_time = asyncio.get_event_loop().time()
    try:
        logger.info(f"Executing recurring job for graph #{args.graph_id}")
        graph_exec: GraphExecutionWithNodes = await execution_utils.add_graph_execution(
            user_id=args.user_id,
            graph_id=args.graph_id,
            graph_version=args.graph_version,
            inputs=args.input_data,
            graph_credentials_inputs=args.input_credentials,
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
    except GraphNotInLibraryError as e:
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.warning(
            f"Scheduled execution blocked for deleted/archived graph {args.graph_id} "
            f"(user {args.user_id}) after {elapsed:.2f}s: {e}"
        )
        # Clean up orphaned schedules for this graph
        await _cleanup_orphaned_schedules_for_graph(args.graph_id, args.user_id)
    except Exception as e:
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.error(
            f"Error executing graph {args.graph_id} after {elapsed:.2f}s: "
            f"{type(e).__name__}: {e}"
        )


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


# Monitoring functions are now imported from monitoring module


class Jobstores(Enum):
    EXECUTION = "execution"
    BATCHED_NOTIFICATIONS = "batched_notifications"
    WEEKLY_NOTIFICATIONS = "weekly_notifications"


class GraphExecutionJobArgs(BaseModel):
    user_id: str
    graph_id: str
    graph_version: int
    cron: str
    input_data: BlockInput
    input_credentials: dict[str, CredentialsMetaInput] = Field(default_factory=dict)


class GraphExecutionJobInfo(GraphExecutionJobArgs):
    id: str
    name: str
    next_run_time: str
    timezone: str = Field(default="UTC", description="Timezone used for scheduling")

    @staticmethod
    def from_db(
        job_args: GraphExecutionJobArgs, job_obj: JobObj
    ) -> "GraphExecutionJobInfo":
        # Extract timezone from the trigger if it's a CronTrigger
        timezone_str = "UTC"
        if hasattr(job_obj.trigger, "timezone"):
            timezone_str = str(job_obj.trigger.timezone)

        return GraphExecutionJobInfo(
            id=job_obj.id,
            name=job_obj.name,
            next_run_time=job_obj.next_run_time.isoformat(),
            timezone=timezone_str,
            **job_args.model_dump(),
        )


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

        self.scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self.scheduler.add_listener(job_missed_listener, EVENT_JOB_MISSED)
        self.scheduler.add_listener(job_max_instances_listener, EVENT_JOB_MAX_INSTANCES)
        self.scheduler.start()

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

    @expose
    def add_graph_execution_schedule(
        self,
        user_id: str,
        graph_id: str,
        graph_version: int,
        cron: str,
        input_data: BlockInput,
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

        # Use provided timezone or default to UTC
        # Note: Timezone should be passed from the client to avoid database lookups
        if not user_timezone:
            user_timezone = "UTC"
            logger.warning(
                f"No timezone provided for user {user_id}, using UTC for scheduling. "
                f"Client should pass user's timezone for correct scheduling."
            )

        logger.info(
            f"Scheduling job for user {user_id} with timezone {user_timezone} (cron: {cron})"
        )

        job_args = GraphExecutionJobArgs(
            user_id=user_id,
            graph_id=graph_id,
            graph_version=graph_version,
            cron=cron,
            input_data=input_data,
            input_credentials=input_credentials,
        )
        job = self.scheduler.add_job(
            execute_graph,
            kwargs=job_args.model_dump(),
            name=name,
            trigger=CronTrigger.from_crontab(cron, timezone=user_timezone),
            jobstore=Jobstores.EXECUTION.value,
            replace_existing=True,
        )
        logger.info(
            f"Added job {job.id} with cron schedule '{cron}' in timezone {user_timezone}, input data: {input_data}"
        )
        return GraphExecutionJobInfo.from_db(job_args, job)

    @expose
    def delete_graph_execution_schedule(
        self, schedule_id: str, user_id: str
    ) -> GraphExecutionJobInfo:
        job = self.scheduler.get_job(schedule_id, jobstore=Jobstores.EXECUTION.value)
        if not job:
            raise NotFoundError(f"Job #{schedule_id} not found.")

        job_args = GraphExecutionJobArgs(**job.kwargs)
        if job_args.user_id != user_id:
            raise NotAuthorizedError("User ID does not match the job's user ID")

        logger.info(f"Deleting job {schedule_id}")
        job.remove()

        return GraphExecutionJobInfo.from_db(job_args, job)

    @expose
    def get_graph_execution_schedules(
        self, graph_id: str | None = None, user_id: str | None = None
    ) -> list[GraphExecutionJobInfo]:
        jobs: list[JobObj] = self.scheduler.get_jobs(jobstore=Jobstores.EXECUTION.value)
        schedules = []
        for job in jobs:
            logger.debug(
                f"Found job {job.id} with cron schedule {job.trigger} and args {job.kwargs}"
            )
            try:
                job_args = GraphExecutionJobArgs.model_validate(job.kwargs)
            except ValidationError:
                continue
            if (
                job.next_run_time is not None
                and (graph_id is None or job_args.graph_id == graph_id)
                and (user_id is None or job_args.user_id == user_id)
            ):
                schedules.append(GraphExecutionJobInfo.from_db(job_args, job))
        return schedules

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


class SchedulerClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return Scheduler

    add_execution_schedule = endpoint_to_async(Scheduler.add_graph_execution_schedule)
    delete_schedule = endpoint_to_async(Scheduler.delete_graph_execution_schedule)
    get_execution_schedules = endpoint_to_async(Scheduler.get_graph_execution_schedules)
