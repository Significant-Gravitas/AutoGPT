import logging
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime

from autogpt_server.data import schedule as model
from autogpt_server.util.service import AppService, expose, get_service_client
from autogpt_server.executor.manager import ExecutionManager

logger = logging.getLogger(__name__)


def log(msg, **kwargs):
    logger.warning("[ExecutionScheduler] " + msg, **kwargs)


class ExecutionScheduler(AppService):

    def __init__(self, refresh_interval=10):
        self.last_check = datetime.min
        self.refresh_interval = refresh_interval

    @property
    def execution_manager_client(self):
        return get_service_client(ExecutionManager)

    def run_service(self):
        scheduler = BackgroundScheduler()
        scheduler.start()
        while True:
            self.__refresh_jobs_from_db(scheduler)
            time.sleep(self.refresh_interval)

    def __refresh_jobs_from_db(self, scheduler: BackgroundScheduler):
        schedules = self.run_and_wait(model.get_active_schedules(self.last_check))
        for schedule in schedules:
            self.last_check = max(self.last_check, schedule.last_updated)

            if not schedule.is_enabled:
                log(f"Removing recurring job {schedule.id}: {schedule.schedule}")
                scheduler.remove_job(schedule.id)
                continue

            log(f"Adding recurring job {schedule.id}: {schedule.schedule}")
            scheduler.add_job(
                self.__execute_graph,
                CronTrigger.from_crontab(schedule.schedule),
                id=schedule.id,
                args=[schedule.graph_id, schedule.input_data],
                replace_existing=True,
            )

    def __execute_graph(self, graph_id: str, input_data: dict):
        try:
            log(f"Executing recurring job for graph #{graph_id}")
            execution_manager = self.execution_manager_client
            execution_manager.add_execution(graph_id, input_data)
        except Exception as e:
            logger.exception(f"Error executing graph {graph_id}: {e}")

    @expose
    def update_schedule(self, schedule_id: str, is_enabled: bool) -> str:
        self.run_and_wait(model.update_schedule(schedule_id, is_enabled))
        return schedule_id

    @expose
    def add_execution_schedule(self, graph_id: str, cron: str, input_data: dict) -> str:
        schedule = model.ExecutionSchedule(
            graph_id=graph_id,
            schedule=cron,
            input_data=input_data,
        )
        return self.run_and_wait(model.add_schedule(schedule)).id

    @expose
    def get_execution_schedules(self, graph_id: str) -> dict[str, str]:
        query = model.get_schedules(graph_id)
        schedules: list[model.ExecutionSchedule] = self.run_and_wait(query)
        return {v.id: v.schedule for v in schedules}
