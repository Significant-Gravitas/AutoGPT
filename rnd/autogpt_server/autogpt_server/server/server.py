import uuid

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException

from autogpt_server.data import db, execution, graph
from autogpt_server.executor import ExecutionManager, ExecutionScheduler
from autogpt_server.util.process import AppProcess
from autogpt_server.util.service import get_service_client


class AgentServer(AppProcess):
    def run(self):
        app = FastAPI(
            title="AutoGPT Agent Server",
            description=(
                "This server is used to execute agents that are created by the "
                "AutoGPT system."
            ),
            summary="AutoGPT Agent Server",
            version="0.1",
        )

        # Define the API routes
        router = APIRouter()
        router.add_api_route(
            path="/agents/{agent_id}",
            endpoint=AgentServer.get_agent,
            methods=["GET"],
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
        app.on_event("startup")(db.connect)
        app.on_event("shutdown")(db.disconnect)
        uvicorn.run(app)

    @staticmethod
    def get_agent(agent_id: str) -> graph.Graph:
        agent = graph.get_graph(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found.")

        return agent

    @staticmethod
    def execute_agent(agent_id: str, node_input: dict) -> dict:
        agent = graph.get_graph(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found.")

        run_id = str(uuid.uuid4())
        executions = []

        # Currently, there is no constraint on the number of root nodes in the graph.
        for node in agent.starting_nodes:
            block = node.block
            if error := block.input_schema.validate_data(node_input):
                raise HTTPException(
                    status_code=400,
                    detail=f"Input data doesn't match {block.name} input: {error}",
                )

            executor_manager = get_service_client(ExecutionManager)
            obj = executor_manager.add_execution(
                run_id=run_id, node_id=node.id, data=node_input
            )
            executions.append(obj)

        return {
            "run_id": run_id,
            "executions": executions,
        }

    @staticmethod
    def get_executions(agent_id: str, run_id: str) -> list[execution.ExecutionResult]:
        agent = graph.get_graph(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found.")

        return execution.get_executions(run_id)

    @staticmethod
    def schedule_agent(agent_id: str, cron: str, input_data: dict) -> dict:
        executor_manager = get_service_client(ExecutionScheduler)
        return {
            "id": executor_manager.add_execution_schedule(agent_id, cron, input_data)
        }

    @staticmethod
    def get_execution_schedules(agent_id: str) -> list[dict]:
        executor_manager = get_service_client(ExecutionScheduler)
        return executor_manager.get_execution_schedules(agent_id)
