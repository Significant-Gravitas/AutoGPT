import time

from autogpt_server.util.service import AppService, expose


class ExecutionScheduler(AppService):

    def run_service(self):
        while True:
            time.sleep(1)  # This will be replaced with apscheduler executor.

    @expose
    def add_execution_schedule(self, agent_id: str, cron: str, input_data: dict) -> str:
        print(
            f"Adding execution schedule for agent {agent_id} with cron {cron} and "
            f"input data {input_data}"
        )
        return "dummy_schedule_id"

    @expose
    def get_execution_schedules(self, agent_id: str) -> list[dict]:
        print(f"Getting execution schedules for agent {agent_id}")
        return [{"cron": "dummy_cron", "input_data": {"dummy_input": "dummy_value"}}]
