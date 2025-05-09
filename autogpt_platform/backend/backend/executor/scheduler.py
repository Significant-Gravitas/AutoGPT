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
from backend.util.metrics import discord_send_alert, sentry_capture_error
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


def execute_graph(**kwargs):
    args = GraphExecutionJobArgs(**kwargs)
    try:
        log(f"Executing recurring job for graph #{args.graph_id}")
        execution_utils.add_graph_execution(
            graph_id=args.graph_id,
            inputs=args.input_data,
            user_id=args.user_id,
            graph_version=args.graph_version,
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
    error = LateExecutionException(
        f"Late executions detected: {num_late_executions} late executions from {num_users} users "
        f"in the last {config.execution_late_notification_checkrange_secs} seconds. "
        f"Graph has been queued for more than {config.execution_late_notification_threshold_secs} seconds. "
        "Please check the executor status."
    )
    msg = str(error)
    sentry_capture_error(error)
    discord_send_alert(msg)
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


class SchedulerClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return Scheduler

    add_execution_schedule = endpoint_to_async(Scheduler.add_graph_execution_schedule)
    delete_schedule = endpoint_to_async(Scheduler.delete_graph_execution_schedule)
    get_execution_schedules = endpoint_to_async(Scheduler.get_graph_execution_schedules)
