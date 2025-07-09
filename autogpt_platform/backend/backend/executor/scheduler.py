import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.job import Job as JobObj
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from autogpt_libs.utils.cache import thread_cached
from dotenv import load_dotenv
from prisma.enums import NotificationType
from pydantic import BaseModel, ValidationError
from sqlalchemy import MetaData, create_engine

from backend.data.block import BlockInput
from backend.data.execution import ExecutionStatus
from backend.executor import utils as execution_utils
from backend.notifications.notifications import NotificationManagerClient
from backend.util.metrics import sentry_capture_error
from backend.util.service import (
    AppService,
    AppServiceClient,
    endpoint_to_async,
    expose,
    get_service_client,
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
config = Config()


def log(msg, **kwargs):
    logger.info("[Scheduler] " + msg, **kwargs)


def job_listener(event):
    """Logs job execution outcomes for better monitoring."""
    if event.exception:
        log(f"Job {event.job_id} failed.")
    else:
        log(f"Job {event.job_id} completed successfully.")


@thread_cached
def get_notification_client():
    return get_service_client(NotificationManagerClient)


@thread_cached
def get_event_loop():
    return asyncio.new_event_loop()


def execute_graph(**kwargs):
    get_event_loop().run_until_complete(_execute_graph(**kwargs))


async def _execute_graph(**kwargs):
    args = GraphExecutionJobArgs(**kwargs)
    try:
        log(f"Executing recurring job for graph #{args.graph_id}")
        await execution_utils.add_graph_execution(
            graph_id=args.graph_id,
            inputs=args.input_data,
            user_id=args.user_id,
            graph_version=args.graph_version,
            use_db_query=False,
        )
    except Exception as e:
        logger.exception(f"Error executing graph {args.graph_id}: {e}")


class LateExecutionException(Exception):
    pass


def report_late_executions() -> str:
    late_executions = execution_utils.get_db_client().get_graph_executions(
        statuses=[ExecutionStatus.QUEUED],
        created_time_gte=datetime.now(timezone.utc)
        - timedelta(seconds=config.execution_late_notification_checkrange_secs),
        created_time_lte=datetime.now(timezone.utc)
        - timedelta(seconds=config.execution_late_notification_threshold_secs),
        limit=1000,
    )

    if not late_executions:
        return "No late executions detected."

    num_late_executions = len(late_executions)
    num_users = len(set([r.user_id for r in late_executions]))

    late_execution_details = [
        f"* `Execution ID: {exec.id}, Graph ID: {exec.graph_id}v{exec.graph_version}, User ID: {exec.user_id}, Created At: {exec.started_at.isoformat()}`"
        for exec in late_executions
    ]

    error = LateExecutionException(
        f"Late executions detected: {num_late_executions} late executions from {num_users} users "
        f"in the last {config.execution_late_notification_checkrange_secs} seconds. "
        f"Graph has been queued for more than {config.execution_late_notification_threshold_secs} seconds. "
        "Please check the executor status. Details:\n"
        + "\n".join(late_execution_details)
    )
    msg = str(error)
    sentry_capture_error(error)
    get_notification_client().discord_system_alert(msg)
    return msg


def process_existing_batches(**kwargs):
    args = NotificationJobArgs(**kwargs)
    try:
        log(
            f"Processing existing batches for notification type {args.notification_types}"
        )
        get_notification_client().process_existing_batches(args.notification_types)
    except Exception as e:
        logger.exception(f"Error processing existing batches: {e}")


def process_weekly_summary(**kwargs):
    try:
        log("Processing weekly summary")
        get_notification_client().queue_weekly_summary()
    except Exception as e:
        logger.exception(f"Error processing weekly summary: {e}")


def report_block_error_rates(**kwargs):
    """Check block error rates and send Discord alerts if thresholds are exceeded."""
    try:
        log("Checking block error rates")

        # Get executions from the last 24 hours
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=24)

        executions = execution_utils.get_db_client().get_node_executions(
            created_time_gte=start_time, created_time_lte=end_time
        )

        # Calculate error rates by block and collect error samples
        block_stats = {}
        for execution in executions:
            block_name = (
                execution.agentNode.agentBlock.name
                if execution.agentNode and execution.agentNode.agentBlock
                else "Unknown"
            )

            if block_name not in block_stats:
                block_stats[block_name] = {"total": 0, "failed": 0, "error_samples": []}

            block_stats[block_name]["total"] += 1
            if execution.executionStatus == "FAILED":
                block_stats[block_name]["failed"] += 1

                # Collect error samples (limit to 5 per block)
                if len(block_stats[block_name]["error_samples"]) < 5:
                    error_message = _extract_error_message(execution)
                    if error_message:
                        masked_error = _mask_sensitive_data(error_message)
                        block_stats[block_name]["error_samples"].append(masked_error)

        # Check thresholds and send alerts
        threshold = config.block_error_rate_threshold
        alerts = []

        for block_name, stats in block_stats.items():
            if stats["total"] >= 10:  # Only check blocks with at least 10 executions
                error_rate = stats["failed"] / stats["total"]
                if error_rate >= threshold:
                    error_percentage = error_rate * 100

                    # Group similar errors
                    error_groups = _group_similar_errors(stats["error_samples"])

                    alert_msg = (
                        f"🚨 Block '{block_name}' has {error_percentage:.1f}% error rate "
                        f"({stats['failed']}/{stats['total']}) in the last 24 hours"
                    )

                    if error_groups:
                        alert_msg += "\n\n📊 Error Types:"
                        for error_pattern, count in error_groups.items():
                            alert_msg += f"\n• {error_pattern} ({count}x)"

                    alerts.append(alert_msg)

        if alerts:
            msg = "Block Error Rate Alert:\n\n" + "\n\n".join(alerts)
            get_notification_client().discord_system_alert(msg)
            log(f"Sent block error rate alert for {len(alerts)} blocks")
            return f"Alert sent for {len(alerts)} blocks with high error rates"
        else:
            log("No blocks exceeded error rate threshold")
            return "No blocks exceeded error rate threshold"

    except Exception as e:
        logger.exception(f"Error checking block error rates: {e}")
        error = LateExecutionException(f"Error checking block error rates: {e}")
        msg = str(error)
        sentry_capture_error(error)
        get_notification_client().discord_system_alert(msg)
        return msg


def _extract_error_message(execution):
    """Extract error message from execution stats."""
    try:
        if hasattr(execution, "stats") and execution.stats:
            stats = execution.stats
            if isinstance(stats, dict):
                # Look for error message in various common locations
                error_msg = (
                    stats.get("error_message")
                    or stats.get("error")
                    or stats.get("exception")
                    or str(stats.get("output", ""))
                )
                return error_msg if error_msg else None
            elif isinstance(stats, str):
                return stats
        return None
    except Exception:
        return None


def _mask_sensitive_data(error_message):
    """Mask sensitive data in error messages to enable grouping."""
    import re

    if not error_message:
        return ""

    # Convert to string if not already
    error_str = str(error_message)

    # Mask numbers (replace with X)
    error_str = re.sub(r"\d+", "X", error_str)

    # Mask all caps words (likely constants/IDs)
    error_str = re.sub(r"\b[A-Z_]{3,}\b", "MASKED", error_str)

    # Mask words with underscores (likely internal variables)
    error_str = re.sub(r"\b\w*_\w*\b", "MASKED", error_str)

    # Mask UUIDs and long alphanumeric strings
    error_str = re.sub(
        r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
        "UUID",
        error_str,
    )
    error_str = re.sub(r"\b[a-f0-9]{20,}\b", "HASH", error_str)

    # Mask file paths
    error_str = re.sub(r"(/[^/\s]+)+", "/MASKED/path", error_str)

    # Mask URLs
    error_str = re.sub(r"https?://[^\s]+", "URL", error_str)

    # Mask email addresses
    error_str = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "EMAIL", error_str
    )

    # Truncate if too long
    if len(error_str) > 100:
        error_str = error_str[:97] + "..."

    return error_str.strip()


def _group_similar_errors(error_samples):
    """Group similar error messages and return counts."""
    if not error_samples:
        return {}

    error_groups = {}
    for error in error_samples:
        if error in error_groups:
            error_groups[error] += 1
        else:
            error_groups[error] = 1

    # Sort by frequency, most common first
    return dict(sorted(error_groups.items(), key=lambda x: x[1], reverse=True))


class Jobstores(Enum):
    EXECUTION = "execution"
    BATCHED_NOTIFICATIONS = "batched_notifications"
    WEEKLY_NOTIFICATIONS = "weekly_notifications"


class GraphExecutionJobArgs(BaseModel):
    graph_id: str
    input_data: BlockInput
    user_id: str
    graph_version: int
    cron: str


class GraphExecutionJobInfo(GraphExecutionJobArgs):
    id: str
    name: str
    next_run_time: str

    @staticmethod
    def from_db(
        job_args: GraphExecutionJobArgs, job_obj: JobObj
    ) -> "GraphExecutionJobInfo":
        return GraphExecutionJobInfo(
            id=job_obj.id,
            name=job_obj.name,
            next_run_time=job_obj.next_run_time.isoformat(),
            **job_args.model_dump(),
        )


class NotificationJobArgs(BaseModel):
    notification_types: list[NotificationType]
    cron: str


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
    scheduler: BlockingScheduler

    def __init__(self, register_system_tasks: bool = True):
        self.register_system_tasks = register_system_tasks

    @classmethod
    def get_port(cls) -> int:
        return config.execution_scheduler_port

    @classmethod
    def db_pool_size(cls) -> int:
        return config.scheduler_db_pool_size

    def run_service(self):
        load_dotenv()
        db_schema, db_url = _extract_schema_from_url(os.getenv("DIRECT_URL"))
        self.scheduler = BlockingScheduler(
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
            }
        )

        if self.register_system_tasks:
            # Notification PROCESS WEEKLY SUMMARY
            self.scheduler.add_job(
                process_weekly_summary,
                CronTrigger.from_crontab("0 * * * *"),
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

        self.scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self.scheduler.start()

    def cleanup(self):
        super().cleanup()
        logger.info(f"[{self.service_name}] ⏳ Shutting down scheduler...")
        if self.scheduler:
            self.scheduler.shutdown(wait=False)

    @expose
    def add_graph_execution_schedule(
        self,
        graph_id: str,
        graph_version: int,
        cron: str,
        input_data: BlockInput,
        user_id: str,
    ) -> GraphExecutionJobInfo:
        job_args = GraphExecutionJobArgs(
            graph_id=graph_id,
            input_data=input_data,
            user_id=user_id,
            graph_version=graph_version,
            cron=cron,
        )
        job = self.scheduler.add_job(
            execute_graph,
            CronTrigger.from_crontab(cron),
            kwargs=job_args.model_dump(),
            replace_existing=True,
            jobstore=Jobstores.EXECUTION.value,
        )
        log(f"Added job {job.id} with cron schedule '{cron}' input data: {input_data}")
        return GraphExecutionJobInfo.from_db(job_args, job)

    @expose
    def delete_graph_execution_schedule(
        self, schedule_id: str, user_id: str
    ) -> GraphExecutionJobInfo:
        job = self.scheduler.get_job(schedule_id, jobstore=Jobstores.EXECUTION.value)
        if not job:
            log(f"Job {schedule_id} not found.")
            raise ValueError(f"Job #{schedule_id} not found.")

        job_args = GraphExecutionJobArgs(**job.kwargs)
        if job_args.user_id != user_id:
            raise ValueError("User ID does not match the job's user ID.")

        log(f"Deleting job {schedule_id}")
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


class SchedulerClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return Scheduler

    add_execution_schedule = endpoint_to_async(Scheduler.add_graph_execution_schedule)
    delete_schedule = endpoint_to_async(Scheduler.delete_graph_execution_schedule)
    get_execution_schedules = endpoint_to_async(Scheduler.get_graph_execution_schedules)
