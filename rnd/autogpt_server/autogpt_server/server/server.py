import asyncio
import uuid
import uvicorn

from fastapi import APIRouter, FastAPI, HTTPException

from autogpt_server.data import db, execution, graph


class AgentServer:
    def __init__(self, queue: execution.ExecutionQueue):
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
        self.app.on_event("startup")(db.connect)
        self.app.on_event("shutdown")(db.disconnect)

    async def execute_agent(self, agent_id: str, node_input: dict[str, str]):
        agent = await graph.get_graph(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found.")

        graph_exec_id = str(uuid.uuid4())
        tasks = []

        # Currently, there is no constraint on the number of root nodes in the graph.
        for node in agent.starting_nodes:
            provided = set(node_input.keys())
            expected = set(node.input_schema.keys()) - set(node.input_default.keys())
            if not expected.issubset(provided):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Input data does not match the expected input schema: "
                        f"expected {expected}, only {provided} is provided."
                    ),
                )

            task = execution.add_execution(
                execution.Execution(
                    graph_exec_id=graph_exec_id, node_id=node.id, data=node_input
                ),
                self.execution_queue,
            )

            tasks.append(task)

        return await asyncio.gather(*tasks)


def start_server(queue: execution.ExecutionQueue):
    agent_server = AgentServer(queue)
    uvicorn.run(agent_server.app)
