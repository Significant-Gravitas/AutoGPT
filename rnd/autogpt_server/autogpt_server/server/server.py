import uvicorn
from fastapi import APIRouter, FastAPI

from autogpt_server.data import ExecutionQueue


class AgentServer:
    def __init__(self, queue: ExecutionQueue):
        self.app = FastAPI(
            title="AutoGPT Agent Server",
            description=(
                "This server is used to execute agents that are created by the "
                "AutoGPT system."
            ),
            summary="AutoGPT Agent Server",
            version="0.1",
        )
        self.execution_queue = queue

        # Define the API routes
        self.router = APIRouter()
        self.router.add_api_route(
            path="/agents/{agent_id}/execute",
            endpoint=self.execute_agent,
            methods=["POST"],
        )
        self.app.include_router(self.router)

    def execute_agent(self, agent_id: str):
        execution_id = self.execution_queue.add(agent_id)
        return {"execution_id": execution_id, "agent_id": agent_id}


def start_server(queue: ExecutionQueue, use_uvicorn: bool = True):
    app = AgentServer(queue).app
    if use_uvicorn:
        uvicorn.run(app)
    return app
