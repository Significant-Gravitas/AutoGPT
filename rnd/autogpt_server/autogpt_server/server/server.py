import asyncio
import uuid
from typing import Annotated, Any, Dict

import uvicorn
from fastapi import WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from contextlib import asynccontextmanager
from fastapi import APIRouter, Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from autogpt_server.data import db, execution, block
from autogpt_server.data.graph import (
    create_graph,
    get_graph,
    get_graph_ids,
    Graph,
)
from autogpt_server.executor import ExecutionManager, ExecutionScheduler
from autogpt_server.server.conn_manager import ConnectionManager
from autogpt_server.server.ws_api import websocket_router as ws_impl
from autogpt_server.util.data import get_frontend_path
from autogpt_server.util.service import expose  # type: ignore
from autogpt_server.util.service import AppService, get_service_client
from autogpt_server.util.settings import Settings


class AgentServer(AppService):
    event_queue: asyncio.Queue[execution.ExecutionResult] = asyncio.Queue()
    manager = ConnectionManager()

    async def event_broadcaster(self):
        while True:
            event: execution.ExecutionResult = await self.event_queue.get()
            await self.manager.send_execution_result(event)

    @asynccontextmanager
    async def lifespan(self, _: FastAPI):
        await db.connect()
        self.run_and_wait(block.initialize_blocks())
        asyncio.create_task(self.event_broadcaster())
        yield
        await db.disconnect()

    def run_service(self):
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

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

        # Define the API routes
        router = APIRouter()
        router.add_api_route(
            path="/blocks",
            endpoint=self.get_graph_blocks,  # type: ignore
            methods=["GET"],
        )
        router.add_api_route(
            path="/blocks/{block_id}/execute",
            endpoint=self.execute_graph_block,
            methods=["POST"],
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
            endpoint=self.execute_graph,  # type: ignore
            methods=["POST"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}/executions",
            endpoint=self.list_graph_runs,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}/executions/{run_id}",
            endpoint=self.get_run_execution_results,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}/schedules",
            endpoint=self.create_schedule,  # type: ignore
            methods=["POST"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}/schedules",
            endpoint=self.get_execution_schedules,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs/schedules/{schedule_id}",
            endpoint=self.update_schedule,  # type: ignore
            methods=["PUT"],
        )

        router.add_api_route(
            path="/settings",
            endpoint=self.update_configuration,
            methods=["POST"],
        )

        app.add_exception_handler(500, self.handle_internal_error)

        app.mount(
            path="/frontend",
            app=StaticFiles(directory=get_frontend_path(), html=True),
            name="example_files",
        )

        app.include_router(router)

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):  # type: ignore
            await ws_impl(websocket, self.manager)

        uvicorn.run(app, host="0.0.0.0", port=8000)

    @property
    def execution_manager_client(self) -> ExecutionManager:
        return get_service_client(ExecutionManager)

    @property
    def execution_scheduler_client(self) -> ExecutionScheduler:
        return get_service_client(ExecutionScheduler)

    @classmethod
    def handle_internal_error(cls, request, exc):
        return JSONResponse(
            content={
                "message": f"{request.url.path} call failure",
                "error": str(exc),
            },
            status_code=500,
        )

    @classmethod
    def get_graph_blocks(cls) -> list[dict[Any, Any]]:
        return [v.to_dict() for v in block.get_blocks().values()]

    @classmethod
    def execute_graph_block(cls, block_id: str, data: dict[str, Any]) -> list:
        obj = block.get_block(block_id)
        if not obj:
            raise HTTPException(status_code=404, detail=f"Block #{block_id} not found.")
        return [{name: data} for name, data in obj.execute(data)]

    @classmethod
    async def get_graphs(cls) -> list[str]:
        return await get_graph_ids()

    @classmethod
    async def get_graph(cls, graph_id: str) -> Graph:
        graph = await get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
        return graph

    @classmethod
    async def create_new_graph(cls, graph: Graph) -> Graph:
        # TODO: replace uuid generation here to DB generated uuids.
        graph.id = str(uuid.uuid4())
        id_map = {node.id: str(uuid.uuid4()) for node in graph.nodes}

        for node in graph.nodes:
            node.id = id_map[node.id]

        for link in graph.links:
            link.source_id = id_map[link.source_id]
            link.sink_id = id_map[link.sink_id]

        return await create_graph(graph)

    async def execute_graph(
        self, graph_id: str, node_input: dict[Any, Any]
    ) -> dict[Any, Any]:
        try:
            return self.execution_manager_client.add_execution(graph_id, node_input)  # type: ignore
        except Exception as e:
            msg = e.__str__().encode().decode("unicode_escape")
            raise HTTPException(status_code=400, detail=msg)

    @classmethod
    async def list_graph_runs(cls, graph_id: str) -> list[str]:
        graph = await get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Agent #{graph_id} not found.")

        return await execution.list_executions(graph_id)

    @classmethod
    async def get_run_execution_results(
        cls, graph_id: str, run_id: str
    ) -> list[execution.ExecutionResult]:
        graph = await get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Agent #{graph_id} not found.")

        return await execution.get_execution_results(run_id)

    async def create_schedule(
        self, graph_id: str, cron: str, input_data: dict[Any, Any]
    ) -> dict[Any, Any]:
        graph = await get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
        execution_scheduler = self.execution_scheduler_client
        return {
            "id": execution_scheduler.add_execution_schedule(graph_id, cron, input_data)  # type: ignore
        }

    def update_schedule(
        self, schedule_id: str, input_data: dict[Any, Any]
    ) -> dict[Any, Any]:
        execution_scheduler = self.execution_scheduler_client
        is_enabled = input_data.get("is_enabled", False)
        execution_scheduler.update_schedule(schedule_id, is_enabled)  # type: ignore
        return {"id": schedule_id}

    def get_execution_schedules(self, graph_id: str) -> dict[str, str]:
        execution_scheduler = self.execution_scheduler_client
        return execution_scheduler.get_execution_schedules(graph_id)  # type: ignore

    @expose
    def send_execution_update(self, execution_result_dict: dict[Any, Any]):
        execution_result = execution.ExecutionResult(**execution_result_dict)
        self.run_and_wait(
            self.event_queue.put(execution_result)
        )

    @classmethod
    def update_configuration(
        cls,
        updated_settings: Annotated[
            Dict[str, Any], Body(examples=[{"config": {"num_workers": 10}}])
        ],
    ):
        settings = Settings()
        try:
            updated_fields: dict[Any, Any] = {"config": [], "secrets": []}
            for key, value in updated_settings.get("config", {}).items():
                if hasattr(settings.config, key):  # type: ignore
                    setattr(settings.config, key, value)  # type: ignore
                    updated_fields["config"].append(key)
            for key, value in updated_settings.get("secrets", {}).items():
                if hasattr(settings.secrets, key):  # type: ignore
                    setattr(settings.secrets, key, value)  # type: ignore
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
