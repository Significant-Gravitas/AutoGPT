import asyncio
import time

from autogpt_server.data import db
from autogpt_server.data.block import Block, initialize_blocks
from autogpt_server.data.execution import ExecutionResult, ExecutionStatus
from autogpt_server.data.queue import AsyncEventQueue
from autogpt_server.executor import ExecutionManager, ExecutionScheduler
from autogpt_server.server import AgentServer
from autogpt_server.server.rest_api import get_user_id
from autogpt_server.util.service import PyroNameServer

log = print


class InMemoryAsyncEventQueue(AsyncEventQueue):
    def __init__(self):
        self.queue = asyncio.Queue()
        self.connected = False
        self.closed = False

    async def connect(self):
        if not self.connected:
            self.connected = True
        return

    async def close(self):
        self.closed = True
        self.connected = False
        return

    async def put(self, execution_result: ExecutionResult):
        if not self.connected:
            raise RuntimeError("Queue is not connected")
        await self.queue.put(execution_result)

    async def get(self):
        if self.closed:
            return None
        if not self.connected:
            raise RuntimeError("Queue is not connected")
        try:
            item = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            return item
        except asyncio.TimeoutError:
            return None


class SpinTestServer:
    def __init__(self):
        self.name_server = PyroNameServer()
        self.exec_manager = ExecutionManager()
        self.in_memory_queue = InMemoryAsyncEventQueue()
        self.agent_server = AgentServer(event_queue=self.in_memory_queue)
        self.scheduler = ExecutionScheduler()

    @staticmethod
    def test_get_user_id():
        return "3e53486c-cf57-477e-ba2a-cb02dc828e1a"

    async def __aenter__(self):
        self.name_server.__enter__()
        self.setup_dependency_overrides()
        self.agent_server.__enter__()
        self.exec_manager.__enter__()
        self.scheduler.__enter__()

        await db.connect()
        await initialize_blocks()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await db.disconnect()

        self.name_server.__exit__(exc_type, exc_val, exc_tb)
        self.agent_server.__exit__(exc_type, exc_val, exc_tb)
        self.exec_manager.__exit__(exc_type, exc_val, exc_tb)
        self.scheduler.__exit__(exc_type, exc_val, exc_tb)

    def setup_dependency_overrides(self):
        # Override get_user_id for testing
        self.agent_server.set_test_dependency_overrides(
            {get_user_id: self.test_get_user_id}
        )


async def wait_execution(
    exec_manager: ExecutionManager,
    user_id: str,
    graph_id: str,
    graph_exec_id: str,
    num_execs: int,
    timeout: int = 20,
) -> list:
    async def is_execution_completed():
        execs = await AgentServer().get_graph_run_node_execution_results(
            graph_id, graph_exec_id, user_id
        )
        return (
            exec_manager.queue.empty()
            and len(execs) == num_execs
            and all(
                v.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
                for v in execs
            )
        )

    # Wait for the executions to complete
    for i in range(timeout):
        if await is_execution_completed():
            return await AgentServer().get_graph_run_node_execution_results(
                graph_id, graph_exec_id, user_id
            )
        time.sleep(1)

    assert False, "Execution did not complete in time."


def execute_block_test(block: Block):
    prefix = f"[Test-{block.name}]"

    if not block.test_input or not block.test_output:
        log(f"{prefix} No test data provided")
        return
    if not isinstance(block.test_input, list):
        block.test_input = [block.test_input]
    if not isinstance(block.test_output, list):
        block.test_output = [block.test_output]

    output_index = 0
    log(f"{prefix} Executing {len(block.test_input)} tests...")
    prefix = " " * 4 + prefix

    for mock_name, mock_obj in (block.test_mock or {}).items():
        log(f"{prefix} mocking {mock_name}...")
        if hasattr(block, mock_name):
            setattr(block, mock_name, mock_obj)
        else:
            log(f"{prefix} mock {mock_name} not found in block")

    for input_data in block.test_input:
        log(f"{prefix} in: {input_data}")

        for output_name, output_data in block.execute(input_data):
            if output_index >= len(block.test_output):
                raise ValueError(f"{prefix} produced output more than expected")
            ex_output_name, ex_output_data = block.test_output[output_index]

            def compare(data, expected_data):
                if data == expected_data:
                    is_matching = True
                elif isinstance(expected_data, type):
                    is_matching = isinstance(data, expected_data)
                elif callable(expected_data):
                    is_matching = expected_data(data)
                else:
                    is_matching = False

                mark = "✅" if is_matching else "❌"
                log(f"{prefix} {mark} comparing `{data}` vs `{expected_data}`")
                if not is_matching:
                    raise ValueError(
                        f"{prefix}: wrong output {data} vs {expected_data}"
                    )

            compare(output_data, ex_output_data)
            compare(output_name, ex_output_name)
            output_index += 1

    if output_index < len(block.test_output):
        raise ValueError(
            f"{prefix} produced output less than expected. output_index={output_index}, len(block.test_output)={len(block.test_output)}"
        )
