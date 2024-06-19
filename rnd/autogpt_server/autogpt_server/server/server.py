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
            endpoint=AgentServer.get_agent_blocks,
            methods=["GET"],
        )
        router.add_api_route(
            path="/agents",
            endpoint=AgentServer.get_agents,
            methods=["GET"],
        )
        router.add_api_route(
            path="/agents/{agent_id}",
            endpoint=AgentServer.get_agent,
            methods=["GET"],
        )
        router.add_api_route(
            path="/agents",
            endpoint=AgentServer.create_agent,
            methods=["POST"],
        )
        router.add_api_route(
            path="/agents/{agent_id}/execute",
            endpoint=AgentServer.execute_agent,
            methods=["POST"],
        )
        router.add_api_route(
            path="/agents/{agent_id}/executions/{run_id}",
            endpoint=AgentServer.get_executions,
            methods=["GET"],
        )
        router.add_api_route(
            path="/agents/{agent_id}/schedules",
            endpoint=AgentServer.schedule_agent,
            methods=["POST"],
        )
        router.add_api_route(
            path="/agents/{agent_id}/schedules",
            endpoint=AgentServer.get_execution_schedules,
            methods=["GET"],
        )

        app.include_router(router)
        uvicorn.run(app, host="0.0.0.0", port=8000)

    @staticmethod
    async def get_agent_blocks() -> list[dict]:
        return [v.to_dict() for v in await block.get_blocks()]

    @staticmethod
    async def get_agents() -> list[str]:
        return await graph.get_graph_ids()

    @staticmethod
    async def get_agent(agent_id: str) -> graph.Graph:
        agent = await graph.get_graph(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found.")

        return agent

    @staticmethod
    async def create_agent(agent: graph.Graph) -> graph.Graph:
        agent.id = str(uuid.uuid4())

        id_map = {node.id: str(uuid.uuid4()) for node in agent.nodes}
        for node in agent.nodes:
            node.id = id_map[node.id]
            node.input_nodes = {k: id_map[v] for k, v in node.input_nodes.items()}
            node.output_nodes = {k: id_map[v] for k, v in node.output_nodes.items()}

        return await graph.create_graph(agent)

    @staticmethod
    async def execute_agent(agent_id: str, node_input: dict) -> dict:
        agent = await graph.get_graph(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found.")

        run_id = str(uuid.uuid4())
        executions = []
        execution_manager = get_service_client(ExecutionManager)

        # Currently, there is no constraint on the number of root nodes in the graph.
        for node in agent.starting_nodes:
            node_block = await block.get_block(node.block_id)
            if not node_block:
                raise HTTPException(
                    status_code=404,
                    detail=f"Block #{node.block_id} not found.",
                )
            if error := node_block.input_schema.validate_data(node_input):
                raise HTTPException(
                    status_code=400,
                    detail=f"Input data doesn't match {node_block.name} input: {error}",
                )

            exec_id = execution_manager.add_execution(
                run_id=run_id, node_id=node.id, data=node_input
            )
            executions.append({
                "exec_id": exec_id,
                "node_id": node.id,
            })

        return {
            "run_id": run_id,
            "executions": executions,
        }

    @staticmethod
    async def get_executions(
            agent_id: str,
            run_id: str
    ) -> list[execution.ExecutionResult]:
        agent = await graph.get_graph(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found.")

        return await execution.get_executions(run_id)

    @staticmethod
    def schedule_agent(agent_id: str, cron: str, input_data: dict) -> dict:
        execution_scheduler = get_service_client(ExecutionScheduler)
        return {
            "id": execution_scheduler.add_execution_schedule(agent_id, cron, input_data)
        }

    @staticmethod
    def get_execution_schedules(agent_id: str) -> list[dict]:
        execution_scheduler = get_service_client(ExecutionScheduler)
        return execution_scheduler.get_execution_schedules(agent_id)
