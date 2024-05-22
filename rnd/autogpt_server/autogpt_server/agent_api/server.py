import uvicorn
from fastapi import FastAPI

from autogpt_server.data import ExecutionQueue

app = FastAPI(
    title="AutoGPT Agent Server",
    description=(
        "This server is used to execute agents that are created by the AutoGPT system."
    ),
    summary="AutoGPT Agent Server",
    version="0.1",
)

execution_queue: ExecutionQueue = None


@app.post("/agents/{agent_id}/execute")
def execute_agent(agent_id: str):
    execution_id = execution_queue.add(agent_id)
    return {"execution_id": execution_id, "agent_id": agent_id}


def start_server(queue: ExecutionQueue):
    global execution_queue
    execution_queue = queue
    uvicorn.run(app)
