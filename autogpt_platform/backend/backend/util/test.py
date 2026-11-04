import asyncio
import inspect
import logging
import time
import uuid
from contextlib import ExitStack
from typing import Sequence, cast

from autogpt_libs.auth import get_user_id

from backend.api.rest_api import AgentServer
from backend.blocks._base import Block, BlockSchema
from backend.data import db, redis_client
from backend.data.block import initialize_blocks
from backend.data.db_manager import DatabaseManager
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    NodeExecutionResult,
    get_graph_execution,
)
from backend.data.model import _BaseCredentials
from backend.data.user import create_default_user
from backend.executor import ExecutionManager, Scheduler
from backend.notifications.notifications import NotificationManager
from backend.util.process import AppProcess

log = logging.getLogger(__name__)


class SpinTestServer:
    def __init__(self):
        self.db_api = DatabaseManager()
        self.exec_manager = ExecutionManager()
        # Only for its test_* helpers: tests call the REST API in process, so
        # its server process is never started.
        self.agent_server = AgentServer()
        self.scheduler = Scheduler(
            register_system_tasks=False, throwaway_jobstores=True
        )
        self.notif_manager = NotificationManager()

    @staticmethod
    def test_get_user_id():
        return "3e53486c-cf57-477e-ba2a-cb02dc828e1a"

    async def __aenter__(self):
        self.setup_dependency_overrides()
        try:
            self.db_api.__enter__()
            self.exec_manager.__enter__()
            self.scheduler.__enter__()
            self.notif_manager.__enter__()

            await db.connect()
            await initialize_blocks()
            await create_default_user()
            self._require_running_services()
        except BaseException as e:
            # A failed __aenter__ gets no __aexit__, and a service left running
            # would hang the session at exit, which joins every child process.
            # The start-up error stays the one raised.
            try:
                await self.__aexit__(type(e), e, e.__traceback__)
            except Exception:
                log.exception("Cleanup after a failed start-up also failed")
            raise

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        died: list[str] = []
        try:
            try:
                try:
                    await redis_client.disconnect_async()
                finally:
                    await db.disconnect()
            finally:
                # Read just before stopping them, which is what makes them
                # exit. Without a preload (CI) a service can still be starting
                # when start-up finishes and die after it; this names it
                # instead of passing the run.
                if exc_type is None:
                    died = self._exited_services()
                self._stop_services()
                # Give services time to fully shut down
                #  This prevents event loop issues where services haven't fully
                # cleaned up before the next test starts
                await asyncio.sleep(0.5)
        except Exception as cleanup_error:
            # A failing cleanup must not hide a service that died: report it,
            # with the cleanup error as its cause.
            if died:
                raise _services_died(died) from cleanup_error
            raise

        if died:
            raise _services_died(died)

    def _require_running_services(self) -> None:
        """Fail on a service process that has already exited. One that dies at
        start-up only logs why, and the tests that need it then fail with
        errors that don't name it, or hang retrying its calls."""
        exited = self._exited_services()
        if exited:
            raise RuntimeError(
                "Test service processes exited during start-up: "
                f"{', '.join(exited)}. Their own log lines say why."
            )

    def _stop_services(self) -> None:
        """Stop every service, even when stopping one fails: one left running
        would hang the session at exit. Stopped in this order, as before."""
        with ExitStack() as stack:
            for service in (
                self.notif_manager,
                self.db_api,
                self.exec_manager,
                self.scheduler,
            ):
                # Callbacks run last-registered first.
                stack.callback(service.stop)

    def _exited_services(self) -> list[str]:
        services: list[AppProcess] = [
            self.db_api,
            self.exec_manager,
            self.scheduler,
            self.notif_manager,
        ]
        return [
            f"{service.service_name} (exit code {process.exitcode})"
            for service in services
            if (process := service.process) is not None and not process.is_alive()
        ]

    def setup_dependency_overrides(self):
        # Override get_user_id for testing
        self.agent_server.set_test_dependency_overrides(
            {get_user_id: self.test_get_user_id}
        )


def _services_died(died: list[str]) -> RuntimeError:
    return RuntimeError(
        "Test service processes exited during the session: "
        f"{', '.join(died)}. Their own log lines say why."
    )


async def wait_execution(
    user_id: str,
    graph_exec_id: str,
    timeout: int = 30,
) -> Sequence[NodeExecutionResult]:
    async def is_execution_completed():
        status = await AgentServer().test_get_graph_run_status(graph_exec_id, user_id)
        log.info(f"Execution status: {status}")
        if status == ExecutionStatus.FAILED:
            log.info("Execution failed")
            raise Exception("Execution failed")
        if status == ExecutionStatus.TERMINATED:
            log.info("Execution terminated")
            raise Exception("Execution terminated")
        return status == ExecutionStatus.COMPLETED

    # Wait for the executions to complete
    for i in range(timeout):
        if await is_execution_completed():
            graph_exec = await get_graph_execution(
                user_id=user_id,
                execution_id=graph_exec_id,
                include_node_executions=True,
            )
            assert graph_exec, f"Graph execution #{graph_exec_id} not found"
            return graph_exec.node_executions
        time.sleep(1)

    assert False, "Execution did not complete in time."


async def execute_block_test(block: Block):
    prefix = f"[Test-{block.name}]"

    if not block.test_input or not block.test_output:
        log.info(f"{prefix} No test data provided")
        return
    if not isinstance(block.test_input, list):
        block.test_input = [block.test_input]
    if not isinstance(block.test_output, list):
        block.test_output = [block.test_output]

    output_index = 0
    log.info(f"{prefix} Executing {len(block.test_input)} tests...")
    prefix = " " * 4 + prefix

    for mock_name, mock_obj in (block.test_mock or {}).items():
        log.info(f"{prefix} mocking {mock_name}...")
        # check whether the field mock_name is an async function or not
        if not hasattr(block, mock_name):
            log.info(f"{prefix} mock {mock_name} not found in block")
            continue

        fun = getattr(block, mock_name)
        is_async = inspect.iscoroutinefunction(fun) or inspect.isasyncgenfunction(fun)

        if is_async:

            async def async_mock(
                *args, _mock_name=mock_name, _mock_obj=mock_obj, **kwargs
            ):
                return _mock_obj(*args, **kwargs)

            setattr(block, mock_name, async_mock)

        else:
            setattr(block, mock_name, mock_obj)

    # Populate credentials argument(s)
    # Generate IDs for execution context
    graph_id = str(uuid.uuid4())
    node_id = str(uuid.uuid4())
    graph_exec_id = str(uuid.uuid4())
    node_exec_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    graph_version = 1  # Default version for tests

    extra_exec_kwargs: dict = {
        "graph_id": graph_id,
        "node_id": node_id,
        "graph_exec_id": graph_exec_id,
        "node_exec_id": node_exec_id,
        "user_id": user_id,
        "graph_version": graph_version,
        "execution_context": ExecutionContext(
            user_id=user_id,
            graph_id=graph_id,
            graph_exec_id=graph_exec_id,
            graph_version=graph_version,
            node_id=node_id,
            node_exec_id=node_exec_id,
        ),
    }
    input_model = cast(type[BlockSchema], block.input_schema)

    # Handle regular credentials fields
    credentials_input_fields = input_model.get_credentials_fields()
    if len(credentials_input_fields) == 1 and isinstance(
        block.test_credentials, _BaseCredentials
    ):
        field_name = next(iter(credentials_input_fields))
        extra_exec_kwargs[field_name] = block.test_credentials
    elif credentials_input_fields and block.test_credentials:
        if not isinstance(block.test_credentials, dict):
            raise TypeError(f"Block {block.name} has no usable test credentials")
        else:
            for field_name in credentials_input_fields:
                if field_name in block.test_credentials:
                    extra_exec_kwargs[field_name] = block.test_credentials[field_name]

    # Handle auto-generated credentials (e.g., from GoogleDriveFileInput)
    auto_creds_fields = input_model.get_auto_credentials_fields()
    if auto_creds_fields and block.test_credentials:
        if isinstance(block.test_credentials, _BaseCredentials):
            # Single credentials object - use for all auto_credentials kwargs
            for kwarg_name in auto_creds_fields.keys():
                extra_exec_kwargs[kwarg_name] = block.test_credentials
        elif isinstance(block.test_credentials, dict):
            for kwarg_name in auto_creds_fields.keys():
                if kwarg_name in block.test_credentials:
                    extra_exec_kwargs[kwarg_name] = block.test_credentials[kwarg_name]

    for input_data in block.test_input:
        log.info(f"{prefix} in: {input_data}")

        async for output_name, output_data in block.execute(
            input_data, **extra_exec_kwargs
        ):
            if output_index >= len(block.test_output):
                raise ValueError(
                    f"{prefix} produced output more than expected {output_index} >= {len(block.test_output)}:\nOutput Expected:\t\t{block.test_output}\nFailed Output Produced:\t('{output_name}', {output_data})\nNote that this may not be the one that was unexpected, but it is the first that triggered the extra output warning"
                )
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
                log.info(f"{prefix} {mark} comparing `{data}` vs `{expected_data}`")
                if not is_matching:
                    raise ValueError(
                        f"{prefix}: wrong output {data} vs {expected_data}\n"
                        f"Output Expected:\t\t{block.test_output}\n"
                        f"Failed Output Produced:\t('{output_name}', {output_data})"
                    )

            compare(output_data, ex_output_data)
            compare(output_name, ex_output_name)
            output_index += 1

    if output_index < len(block.test_output):
        raise ValueError(
            f"{prefix} produced output less than expected. output_index={output_index}, len(block.test_output)={len(block.test_output)}"
        )
