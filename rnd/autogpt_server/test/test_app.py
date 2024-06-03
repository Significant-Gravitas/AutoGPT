import pytest

from autogpt_server.data import ExecutionQueue
from autogpt_server.agent_api import start_server
from autogpt_server.agent_executor import start_executors
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    execution_queue = ExecutionQueue()
    start_executors(5, execution_queue)
    return TestClient(start_server(execution_queue, use_uvicorn=False))


def test_execute_agent(client):
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
