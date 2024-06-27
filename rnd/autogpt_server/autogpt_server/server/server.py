from typing import Annotated, Any, Dict
import uuid
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from contextlib import asynccontextmanager
from fastapi import APIRouter, Body, FastAPI, HTTPException

from autogpt_server.data import db, execution, block
from autogpt_server.data.graph import (
    create_graph,
    get_graph,
    get_graph_ids,
    Graph,
    Link,
)
from autogpt_server.executor import ExecutionManager, ExecutionScheduler
from autogpt_server.util.data import get_frontend_path
from autogpt_server.util.process import AppProcess
from autogpt_server.util.service import get_service_client
from autogpt_server.util.settings import Settings


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
            endpoint=self.get_graph_blocks,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs",
            endpoint=self.get_graphs,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}",
            endpoint=self.get_graph,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs",
            endpoint=self.create_new_graph,
            methods=["POST"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}/execute",
            endpoint=self.execute_graph,
            methods=["POST"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}/executions/{run_id}",
            endpoint=self.get_executions,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}/schedules",
            endpoint=self.create_schedule,
            methods=["POST"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}/schedules",
            endpoint=self.get_execution_schedules,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs/schedules/{schedule_id}",
            endpoint=self.update_schedule,
            methods=["PUT"],
        )
        router.add_api_route(
            path="/settings",
            endpoint=self.update_configuration,
            methods=["POST"],
        )

        app.mount(
            path="/frontend",
            app=StaticFiles(directory=get_frontend_path(), html=True),
            name="example_files",
        )

        app.include_router(router)
        uvicorn.run(app, host="0.0.0.0", port=8000)

    @property
    def execution_manager_client(self) -> ExecutionManager:
        return get_service_client(ExecutionManager)

    @property
    def execution_scheduler_client(self) -> ExecutionScheduler:
        return get_service_client(ExecutionScheduler)

    async def get_graph_blocks(self) -> list[dict]:
        return [v.to_dict() for v in await block.get_blocks()]

    async def get_graphs(self) -> list[str]:
        return await get_graph_ids()

    async def get_graph(self, graph_id: str) -> Graph:
        graph = await get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
        return graph

    async def create_new_graph(self, graph: Graph) -> Graph:
        # TODO: replace uuid generation here to DB generated uuids.
        graph.id = str(uuid.uuid4())
        id_map = {node.id: str(uuid.uuid4()) for node in graph.nodes}
        for node in graph.nodes:
            node.id = id_map[node.id]
            node.input_nodes = [Link(k, id_map[v]) for k, v in node.input_nodes]
            node.output_nodes = [Link(k, id_map[v]) for k, v in node.output_nodes]

        return await create_graph(graph)

    async def execute_graph(self, graph_id: str, node_input: dict) -> dict:
        try:
            return self.execution_manager_client.add_execution(graph_id, node_input)
        except Exception as e:
            msg = e.__str__().encode().decode("unicode_escape")
            raise HTTPException(status_code=400, detail=msg)

    async def get_executions(
        self, graph_id: str, run_id: str
    ) -> list[execution.ExecutionResult]:
        graph = await get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Agent #{graph_id} not found.")

        return await execution.get_executions(run_id)

    async def create_schedule(self, graph_id: str, cron: str, input_data: dict) -> dict:
        graph = await get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
        execution_scheduler = self.execution_scheduler_client
        return {
            "id": execution_scheduler.add_execution_schedule(graph_id, cron, input_data)
        }

    def update_schedule(self, schedule_id: str, input_data: dict) -> dict:
        execution_scheduler = self.execution_scheduler_client
        is_enabled = input_data.get("is_enabled", False)
        execution_scheduler.update_schedule(schedule_id, is_enabled)
        return {"id": schedule_id}

    def get_execution_schedules(self, graph_id: str) -> dict[str, str]:
        execution_scheduler = self.execution_scheduler_client
        return execution_scheduler.get_execution_schedules(graph_id)

    def update_configuration(
        self,
        updated_settings: Annotated[
            Dict[str, Any], Body(examples=[{"config": {"num_workers": 10}}])
        ],
    ):
        settings = Settings()
        try:
            updated_fields = {"config": [], "secrets": []}
            for key, value in updated_settings.get("config", {}).items():
                if hasattr(settings.config, key):
                    setattr(settings.config, key, value)
                    updated_fields["config"].append(key)
            for key, value in updated_settings.get("secrets", {}).items():
                if hasattr(settings.secrets, key):
                    setattr(settings.secrets, key, value)
                    updated_fields["secrets"].append(key)
            settings.save()
            return JSONResponse(
                content={
                    "message": "Settings updated successfully",
                    "updated_fields": updated_fields,
                },
                status_code=200,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
