import time

from autogpt_server.data import block, db
from autogpt_server.data.execution import ExecutionStatus
from autogpt_server.executor import ExecutionManager
from autogpt_server.server import AgentServer
from autogpt_server.util.service import PyroNameServer


class SpinTestServer:
    def __init__(self):
        self.name_server = PyroNameServer()
        self.exec_manager = ExecutionManager(1)
        self.agent_server = AgentServer()

    async def __aenter__(self):
        self.name_server.__enter__()
        self.agent_server.__enter__()
        self.exec_manager.__enter__()

        await db.connect()
        await block.initialize_blocks()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await db.disconnect()

        self.name_server.__exit__(exc_type, exc_val, exc_tb)
        self.agent_server.__exit__(exc_type, exc_val, exc_tb)
        self.exec_manager.__exit__(exc_type, exc_val, exc_tb)


async def wait_execution(
        exec_manager: ExecutionManager,
        graph_id: str,
        graph_exec_id: str,
        num_execs: int,
        timeout: int = 20,
) -> list:
    async def is_execution_completed():
        execs = await AgentServer().get_run_execution_results(graph_id, graph_exec_id)
        return exec_manager.queue.empty() and len(execs) == num_execs and all(
            v.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
            for v in execs
        )

    # Wait for the executions to complete
    for i in range(timeout):
        if await is_execution_completed():
            return await AgentServer().get_run_execution_results(
                graph_id, graph_exec_id
            )
        time.sleep(1)

    assert False, "Execution did not complete in time."
