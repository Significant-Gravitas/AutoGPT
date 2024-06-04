import pytest
from fastapi.testclient import TestClient

from autogpt_server.server import start_server
from autogpt_server.executor import start_executor_manager
from autogpt_server.data import ExecutionQueue


@pytest.fixture
def client():
    execution_queue = ExecutionQueue()
    start_executor_manager(5, execution_queue)
    return TestClient(start_server(execution_queue, use_uvicorn=False))


def test_execute_agent(client: TestClient):
    # Assert API is working
    response = client.post("/agents/dummy_agent_1/execute")
    assert response.status_code == 200

    # Assert response is correct
    data = response.json()
    exec_id = data["execution_id"]
    agent_id = data["agent_id"]
    assert agent_id == "dummy_agent_1"
    assert isinstance(exec_id, str)
    assert len(exec_id) == 36

    # TODO: Add assertion that the executor is executed after some time
    # Add this when db integration is done.
