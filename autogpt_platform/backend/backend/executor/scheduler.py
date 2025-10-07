import asyncio
import logging
import os
import threading
from enum import Enum
from typing import Optional, Awaitable
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
from apscheduler.executors.pool import ThreadPoolExecutor

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
from backend.util.cloud_storage import cleanup_expired_files_async
from backend.util.exceptions import NotAuthorizedError, NotFoundError
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
    parsed_url = urlparse(database_url)
    query_params = parse_qs(parsed_url.query)
    schema_list = query_params.pop("schema", None)
    schema = schema_list[0] if schema_list else "public"
    new_query = urlencode(query_params, doseq=True)
    new_parsed_url = parsed_url._replace(query=new_query)
    database_url_clean = str(urlunparse(new_parsed_url))
    return schema, database_url_clean

logger = logging.getLogger(__name__)
logger.addFilter(PrefixFilter("[Scheduler]"))
apscheduler_logger = logger.getChild("apscheduler")
apscheduler_logger.addFilter(PrefixFilter("[Scheduler] [APScheduler]"))
config = Config()

SCHEDULER_OPERATION_TIMEOUT_SECONDS = 300

class EventLoopPool:
    def __init__(self, size: int = 4):
        self.size = max(1, size)
        self._loops: list[asyncio.AbstractEventLoop] = []
        self._threads: list[threading.Thread] = []
        self._counter = 0
        self._lock = threading.Lock()

    def start(self):
        if self._loops:
            return
        for i in range(self.size):
            loop = asyncio.new_event_loop()
            t = threading.Thread(target=loop.run_forever, daemon=True, name=f"SchedulerLoop-{i}")
            t.start()
            self._loops.append(loop)
            self._threads.append(t)
        logger.info(f"Initialized EventLoopPool with {self.size} loops")

    def stop(self):
        for loop in self._loops:
            loop.call_soon_threadsafe(loop.stop)
        for t in self._threads:
            t.join(timeout=SCHEDULER_OPERATION_TIMEOUT_SECONDS)
        self._loops.clear()
        self._threads.clear()

    def _pick_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            loop = self._loops[self._counter % len(self._loops)]
            self._counter += 1
            return loop

    def run(self, coro: Awaitable, timeout: float | None = SCHEDULER_OPERATION_TIMEOUT_SECONDS):
        loop = self._pick_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout=timeout)

_loop_pool: EventLoopPool | None = None

@func_retry
def require_pool() -> EventLoopPool:
    if _loop_pool is None:
        raise RuntimeError("Event loop pool not initialized. Scheduler not started.")
    return _loop_pool

def run_async(coro: Awaitable, timeout: float | None = SCHEDULER_OPERATION_TIMEOUT_SECONDS):
    return require_pool().run(coro, timeout)


def execute_graph(**kwargs):
    run_async(_execute_graph(**kwargs))

async def _execute_graph(**kwargs):
    args = GraphExecutionJobArgs(**kwargs)
    start = asyncio.get_event_loop().time()
    try:
        logger.info(f"Executing recurring job for graph #{args.graph_id}")
        graph_exec: GraphExecutionWithNodes = await execution_utils.add_graph_execution(
            user_id=args.user_id,
            graph_id=args.graph_id,
            graph_version=args.graph_version,
            inputs=args.input_data,
            graph_credentials_inputs=args.input_credentials,
        )
        elapsed = asyncio.get_event_loop().time() - start
        logger.info(
            f"Graph execution started id={graph_exec.id} graph={args.graph_id} (created in {elapsed:.2f}s)"
        )
        if elapsed > 10:
            logger.warning(
                f"Graph execution {graph_exec.id} enqueue slow {elapsed:.2f}s - possible contention"
            )
    except Exception as e:
        elapsed = asyncio.get_event_loop().time() - start
        logger.error(f"Error executing graph {args.graph_id} after {elapsed:.2f}s: {type(e).__name__}: {e}")
        raise

def cleanup_expired_files():
    run_async(cleanup_expired_files_async())


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

class GraphExecutionJobInfo(BaseModel):
    id: str
    name: str
    next_run_time: str
    timezone: str = Field(default="UTC")
    user_id: str
    graph_id: str
    graph_version: int
    cron: str
    input_data: BlockInput
    input_credentials: dict[str, CredentialsMetaInput] = Field(default_factory=dict)

    @staticmethod
    def from_db(job_args: GraphExecutionJobArgs, job_obj: JobObj) -> "GraphExecutionJobInfo":
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
    def from_db(job_args: NotificationJobArgs, job_obj: JobObj) -> "NotificationJobInfo":
        return NotificationJobInfo(
            id=job_obj.id,
            name=job_obj.name,
            next_run_time=job_obj.next_run_time.isoformat(),
            **job_args.model_dump(),
        )


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = CircuitState.CLOSED
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._opened_at and (asyncio.get_event_loop().time() - self._opened_at) > self.reset_timeout:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            return True

    def on_success(self):
        with self._lock:
            self.failures = 0
            self.state = CircuitState.CLOSED
            self._opened_at = None

    def on_failure(self):
        with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self._opened_at = asyncio.get_event_loop().time()
                logger.warning("Scheduler circuit opened due to repeated failures")

_circuit = CircuitBreaker(
    failure_threshold=max(3, getattr(config, "scheduler_circuit_failure_threshold", 5)),
    reset_timeout=getattr(config, "scheduler_circuit_reset_timeout", 30),
)


def job_listener(event):
    if event.exception:
        logger.error(f"Job {event.job_id} failed: {type(event.exception).__name__}: {event.exception}")
    else:
        logger.info(f"Job {event.job_id} completed successfully.")

def job_missed_listener(event):
    logger.warning(
        f"Job {event.job_id} was missed at scheduled time {event.scheduled_run_time}. "
        f"Scheduler may be overloaded or previous executions still running."
    )

def job_max_instances_listener(event):
    logger.warning(
        f"Job {event.job_id} skipped - max instances limit reached. "
        f"Consider increasing max_instances or investigate long run times."
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
        if not hasattr(self, "scheduler"):
            raise UnhealthyServiceError("Scheduler is still initializing")
        if self.cleaned_up:
            return await super().health_check()
        if not self.scheduler.running:
            raise UnhealthyServiceError("Scheduler is not running")
        return await super().health_check()

    def _build_engine(self, db_url: str):
        return create_engine(
            url=db_url,
            pool_size=self.db_pool_size(),
            max_overflow=0,
            pool_pre_ping=True,
            pool_recycle=1800,
        )

    def run_service(self):
        load_dotenv()

        global _loop_pool
        _loop_pool = EventLoopPool(size=self.db_pool_size())
        _loop_pool.start()

        db_schema, db_url = _extract_schema_from_url(os.getenv("DIRECT_URL"))

        self.scheduler = BackgroundScheduler(
            executors={
                "default": ThreadPoolExecutor(max_workers=self.db_pool_size()),
            },
            job_defaults={
                "coalesce": True,
                "max_instances": 1000,
                "misfire_grace_time": None,
            },
            jobstores={
                Jobstores.EXECUTION.value: SQLAlchemyJobStore(
                    engine=self._build_engine(db_url),
                    metadata=MetaData(schema=db_schema),
                    tablename="apscheduler_jobs",
                ),
                Jobstores.BATCHED_NOTIFICATIONS.value: SQLAlchemyJobStore(
                    engine=self._build_engine(db_url),
                    metadata=MetaData(schema=db_schema),
                    tablename="apscheduler_jobs_batched_notifications",
                ),
                Jobstores.WEEKLY_NOTIFICATIONS.value: MemoryJobStore(),
            },
            logger=apscheduler_logger,
            timezone=ZoneInfo("UTC"),
        )

        if self.register_system_tasks:
            self.scheduler.add_job(
                process_weekly_summary,
                CronTrigger.from_crontab("0 9 * * 1"),
                id="process_weekly_summary",
                kwargs={},
                replace_existing=True,
                jobstore=Jobstores.WEEKLY_NOTIFICATIONS.value,
            )
            self.scheduler.add_job(
                report_late_executions,
                id="report_late_executions",
                trigger="interval",
                replace_existing=True,
                seconds=config.execution_late_notification_threshold_secs,
                jobstore=Jobstores.EXECUTION.value,
            )
            self.scheduler.add_job(
                report_block_error_rates,
                id="report_block_error_rates",
                trigger="interval",
                replace_existing=True,
                seconds=config.block_error_rate_check_interval_secs,
                jobstore=Jobstores.EXECUTION.value,
            )
            self.scheduler.add_job(
                cleanup_expired_files,
                id="cleanup_expired_files",
                trigger="interval",
                replace_existing=True,
                seconds=config.cloud_storage_cleanup_interval_hours * 3600,
                jobstore=Jobstores.EXECUTION.value,
            )

        self.scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self.scheduler.add_listener(job_missed_listener, EVENT_JOB_MISSED)
        self.scheduler.add_listener(job_max_instances_listener, EVENT_JOB_MAX_INSTANCES)
        self.scheduler.start()

        super().run_service()

    def cleanup(self):
        super().cleanup()
        if getattr(self, "scheduler", None):
            logger.info("⏳ Shutting down scheduler...")
            self.scheduler.shutdown(wait=True)
        global _loop_pool
        if _loop_pool:
            logger.info("⏳ Shutting down event loop pool...")
            _loop_pool.stop()
            _loop_pool = None
        logger.info("Scheduler cleanup complete.")

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
        if not _circuit.allow():
            raise UnhealthyServiceError("Scheduler temporarily paused due to recent failures")
        run_async(
            execution_utils.validate_and_construct_node_execution_input(
                graph_id=graph_id,
                user_id=user_id,
                graph_inputs=input_data,
                graph_version=graph_version,
                graph_credentials_inputs=input_credentials,
            )
        )
        if not user_timezone:
            user_timezone = "UTC"
            logger.warning(
                f"No timezone provided for user {user_id}, defaulting to UTC. Client should send timezone."
            )
        logger.info(
            f"Scheduling job for user {user_id} TZ={user_timezone} cron={cron}"
        )
        job_args = GraphExecutionJobArgs(
            user_id=user_id,
            graph_id=graph_id,
            graph_version=graph_version,
            cron=cron,
            input_data=input_data,
            input_credentials=input_credentials,
        )
        try:
            job = self.scheduler.add_job(
                execute_graph,
                kwargs=job_args.model_dump(),
                name=name,
                trigger=CronTrigger.from_crontab(cron, timezone=user_timezone),
                jobstore=Jobstores.EXECUTION.value,
                replace_existing=True,
            )
        except Exception:
            _circuit.on_failure()
            raise
        else:
            _circuit.on_success()
        logger.info(
            f"Added job {job.id} cron='{cron}' tz={user_timezone}"
        )
        return GraphExecutionJobInfo.from_db(job_args, job)

    @expose
    def delete_graph_execution_schedule(self, schedule_id: str, user_id: str) -> GraphExecutionJobInfo:
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
    def get_graph_execution_schedules(self, graph_id: str | None = None, user_id: str | None = None) -> list[GraphExecutionJobInfo]:
        jobs: list[JobObj] = self.scheduler.get_jobs(jobstore=Jobstores.EXECUTION.value)
        schedules: list[GraphExecutionJobInfo] = []
        for job in jobs
