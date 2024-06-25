import uuid
import uvicorn

from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI, HTTPException

from autogpt_server.data import db, execution, graph, block
from autogpt_server.executor import ExecutionManager, ExecutionScheduler
from autogpt_server.util.process import AppProcess
from autogpt_server.util.service import get_service_client


class AgentServer(AppProcess):

    @asynccontextmanager
    async def lifespan(self, _: FastAPI):
        await db.connect()
        yield
        await db.disconnect()

    def run(self):
        app = FastAPI(
            title="AutoGPT Agent Server",
            description=(
                "This server is used to execute agents that are created by the "
                "AutoGPT system."
            ),
            summary="AutoGPT Agent Server",
            version="0.1",
            lifespan=self.lifespan,
        )

        # Define the API routes
        router = APIRouter()
        router.add_api_route(
            path="/blocks",
            endpoint=self.get_agent_blocks,
            methods=["GET"],
        )
        router.add_api_route(
            path="/agents",
            endpoint=self.get_agents,
            methods=["GET"],
        )
        router.add_api_route(
            path="/agents/{agent_id}",
            endpoint=self.get_agent,
            methods=["GET"],
        )
        router.add_api_route(
            path="/agents",
            endpoint=self.create_agent,
            methods=["POST"],
        )
        router.add_api_route(
            path="/agents/{agent_id}/execute",
            endpoint=self.execute_agent,
            methods=["POST"],
        )
        router.add_api_route(
            path="/agents/{agent_id}/executions/{run_id}",
            endpoint=self.get_executions,
            methods=["GET"],
        )
        router.add_api_route(
            path="/agents/{agent_id}/schedules",
            endpoint=self.schedule_agent,
            methods=["POST"],
        )
        router.add_api_route(
            path="/agents/{agent_id}/schedules",
            endpoint=self.get_execution_schedules,
            methods=["GET"],
        )
        router.add_api_route(
            path="/agents/schedules/{schedule_id}",
            endpoint=self.update_schedule,
            methods=["PUT"],
        )

        app.include_router(router)
        uvicorn.run(app, host="0.0.0.0", port=8000)

    @property
    def execution_manager_client(self) -> ExecutionManager:
        return get_service_client(ExecutionManager)

    @property
    def execution_scheduler_client(self) -> ExecutionScheduler:
        return get_service_client(ExecutionScheduler)

    async def get_agent_blocks(self) -> list[dict]:
        return [v.to_dict() for v in await block.get_blocks()]

    async def get_agents(self) -> list[str]:
        return await graph.get_graph_ids()

    async def get_agent(self, agent_id: str) -> graph.Graph:
        agent = await graph.get_graph(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found.")

        return agent

    async def create_agent(self, agent: graph.Graph) -> graph.Graph:
        agent.id = str(uuid.uuid4())

        id_map = {node.id: str(uuid.uuid4()) for node in agent.nodes}
        for node in agent.nodes:
            node.id = id_map[node.id]
            node.input_nodes = {k: id_map[v] for k, v in node.input_nodes.items()}
            node.output_nodes = {k: id_map[v] for k, v in node.output_nodes.items()}

        return await graph.create_graph(agent)

    async def execute_agent(self, agent_id: str, node_input: dict) -> dict:
        try:
            return self.execution_manager_client.add_execution(agent_id, node_input)
        except Exception as e:
            msg = e.__str__().encode().decode('unicode_escape')
            raise HTTPException(status_code=400, detail=msg)

    async def get_executions(
            self, agent_id: str, run_id: str) -> list[execution.ExecutionResult]:
        agent = await graph.get_graph(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found.")

        return await execution.get_executions(run_id)

    async def schedule_agent(self, agent_id: str, cron: str, input_data: dict) -> dict:
        agent = await graph.get_graph(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found.")
        execution_scheduler = self.execution_scheduler_client
        return {
            "id": execution_scheduler.add_execution_schedule(agent_id, cron, input_data)
        }

    def update_schedule(self, schedule_id: str, input_data: dict) -> dict:
        execution_scheduler = self.execution_scheduler_client
        is_enabled = input_data.get("is_enabled", False)
        execution_scheduler.update_schedule(schedule_id, is_enabled)
        return {"id": schedule_id}

    def get_execution_schedules(self, agent_id: str) -> dict[str, str]:
        execution_scheduler = self.execution_scheduler_client
        return execution_scheduler.get_execution_schedules(agent_id)
