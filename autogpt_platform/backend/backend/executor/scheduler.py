import logging
import os
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.job import Job as JobObj
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from autogpt_libs.utils.cache import thread_cached
from dotenv import load_dotenv
from pydantic import BaseModel
from sqlalchemy import MetaData, create_engine

from backend.data.block import BlockInput
from backend.executor.manager import ExecutionManager
from backend.util.service import AppService, expose, get_service_client
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
    logger.info("[ExecutionScheduler] " + msg, **kwargs)


def job_listener(event):
    """Logs job execution outcomes for better monitoring."""
    if event.exception:
        log(f"Job {event.job_id} failed.")
    else:
        log(f"Job {event.job_id} completed successfully.")


@thread_cached
def get_execution_client() -> ExecutionManager:
    return get_service_client(ExecutionManager)


def execute_graph(**kwargs):
    args = JobArgs(**kwargs)
    try:
        log(f"Executing recurring job for graph #{args.graph_id}")
        get_execution_client().add_execution(
            graph_id=args.graph_id,
            data=args.input_data,
            user_id=args.user_id,
            graph_version=args.graph_version,
        )
    except Exception as e:
        logger.exception(f"Error executing graph {args.graph_id}: {e}")


class JobArgs(BaseModel):
    graph_id: str
    input_data: BlockInput
    user_id: str
    graph_version: int
    cron: str


class JobInfo(JobArgs):
    id: str
    name: str
    next_run_time: str

    @staticmethod
    def from_db(job_args: JobArgs, job_obj: JobObj) -> "JobInfo":
        return JobInfo(
            id=job_obj.id,
            name=job_obj.name,
            next_run_time=job_obj.next_run_time.isoformat(),
            **job_args.model_dump(),
        )


class ExecutionScheduler(AppService):
    scheduler: BlockingScheduler

    @classmethod
    def get_port(cls) -> int:
        return config.execution_scheduler_port

    @classmethod
    def db_pool_size(cls) -> int:
        return config.scheduler_db_pool_size

    @property
    @thread_cached
    def execution_client(self) -> ExecutionManager:
        return get_service_client(ExecutionManager)

    def run_service(self):
        load_dotenv()
        db_schema, db_url = _extract_schema_from_url(os.getenv("DATABASE_URL"))
        self.scheduler = BlockingScheduler(
            jobstores={
                "default": SQLAlchemyJobStore(
                    engine=create_engine(
                        url=db_url,
                        pool_size=self.db_pool_size(),
                        max_overflow=0,
                    ),
                    metadata=MetaData(schema=db_schema),
                )
            }
        )
        self.scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self.scheduler.start()

    @expose
    def add_execution_schedule(
        self,
        graph_id: str,
        graph_version: int,
        cron: str,
        input_data: BlockInput,
        user_id: str,
    ) -> JobInfo:
        job_args = JobArgs(
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
        )
        log(f"Added job {job.id} with cron schedule '{cron}' input data: {input_data}")
        return JobInfo.from_db(job_args, job)

    @expose
    def delete_schedule(self, schedule_id: str, user_id: str) -> JobInfo:
        job = self.scheduler.get_job(schedule_id)
        if not job:
            log(f"Job {schedule_id} not found.")
            raise ValueError(f"Job #{schedule_id} not found.")

        job_args = JobArgs(**job.kwargs)
        if job_args.user_id != user_id:
            raise ValueError("User ID does not match the job's user ID.")

        log(f"Deleting job {schedule_id}")
        job.remove()

        return JobInfo.from_db(job_args, job)

    @expose
    def get_execution_schedules(
        self, graph_id: str | None = None, user_id: str | None = None
    ) -> list[JobInfo]:
        schedules = []
        for job in self.scheduler.get_jobs():
            job_args = JobArgs(**job.kwargs)
            if (
                job.next_run_time is not None
                and (graph_id is None or job_args.graph_id == graph_id)
                and (user_id is None or job_args.user_id == user_id)
            ):
                schedules.append(JobInfo.from_db(job_args, job))
        return schedules
