import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict

import uvicorn
from fastapi import (
    APIRouter,
    Body,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import autogpt_server.data.graph
import autogpt_server.server.ws_api
from autogpt_server.data import block, db, execution
from autogpt_server.executor import ExecutionManager, ExecutionScheduler
from autogpt_server.server.conn_manager import ConnectionManager
from autogpt_server.server.model import CreateGraph, Methods, WsMessage
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
            endpoint=self.execute_graph_block,  # type: ignore
            methods=["POST"],
        )
        router.add_api_route(
            path="/templates",
            endpoint=self.get_templates,
            methods=["GET"],
        )
        router.add_api_route(
            path="/templates",
            endpoint=self.create_new_template,
            methods=["POST"],
        )
        router.add_api_route(
            path="/templates",
            endpoint=self.update_graph,
            methods=["PATCH"],
        )
        router.add_api_route(
            path="/templates/{graph_id}",
            endpoint=self.get_template,
            methods=["GET"],
        )
        router.add_api_route(
            path="/templates/{graph_id}/history",
            endpoint=self.get_graph_history,
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
            path="/graphs/{graph_id}/history",
            endpoint=self.get_graph_history,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs",
            endpoint=self.create_new_graph,
            methods=["POST"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}",
            endpoint=self.update_graph,
            methods=["PUT"],
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

        app.add_exception_handler(500, self.handle_internal_error)  # type: ignore

        app.mount(
            path="/frontend",
            app=StaticFiles(directory=get_frontend_path(), html=True),
            name="example_files",
        )

        app.include_router(router)

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):  # type: ignore
            await self.websocket_router(websocket)

        uvicorn.run(app, host="0.0.0.0", port=8000)

    @property
    def execution_manager_client(self) -> ExecutionManager:
        return get_service_client(ExecutionManager)

    @property
    def execution_scheduler_client(self) -> ExecutionScheduler:
        return get_service_client(ExecutionScheduler)

    @classmethod
    def handle_internal_error(cls, request, exc):  # type: ignore
        return JSONResponse(
            content={
                "message": f"{request.url.path} call failure",  # type: ignore
                "error": str(exc),  # type: ignore
            },
            status_code=500,
        )

    async def websocket_router(self, websocket: WebSocket):
        await self.manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                message = WsMessage.model_validate_json(data)
                if message.method == Methods.SUBSCRIBE:
                    await autogpt_server.server.ws_api.handle_subscribe(
                        websocket, self.manager, message
                    )

                elif message.method == Methods.UNSUBSCRIBE:
                    await autogpt_server.server.ws_api.handle_unsubscribe(
                        websocket, self.manager, message
                    )
                else:
                    print("Message type is not processed by the server")
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.ERROR,
                            success=False,
                            error="Message type is not processed by the server",
                        ).model_dump_json()
                    )

        except WebSocketDisconnect:
            self.manager.disconnect(websocket)
            print("Client Disconnected")

    @classmethod
    def get_graph_blocks(cls) -> list[dict[Any, Any]]:
        return [v.to_dict() for v in block.get_blocks().values()]  # type: ignore

    @classmethod
    def execute_graph_block(
        cls, block_id: str, data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        obj = block.get_block(block_id)  # type: ignore
        if not obj:
            raise HTTPException(status_code=404, detail=f"Block #{block_id} not found.")
        return [{name: data} for name, data in obj.execute(data)]

    @classmethod
    async def get_graphs(cls) -> list[str]:
        return await autogpt_server.data.graph.get_graph_ids()

    @classmethod
    async def get_templates(cls) -> list[autogpt_server.data.graph.GraphMeta]:
        return await autogpt_server.data.graph.get_graphs_meta(is_template=True)

    @classmethod
    async def get_graph(
        cls, graph_id: str, version: int | None = None
    ) -> autogpt_server.data.graph.Graph:
        graph = await autogpt_server.data.graph.get_graph(graph_id, version)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
        return graph

    @classmethod
    async def get_graph_history(
        cls, graph_id: str
    ) -> list[autogpt_server.data.graph.Graph]:
        graphs = await autogpt_server.data.graph.get_graph_history(graph_id)
        if not graphs:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
        return graphs

    @classmethod
    async def get_template(
        cls, graph_id: str, version: int | None = None
    ) -> autogpt_server.data.graph.Graph:
        graph = await autogpt_server.data.graph.get_graph(graph_id, version)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
        if not graph.is_template:
            raise HTTPException(
                status_code=400, detail=f"Graph #{graph_id} is not a template."
            )
        return graph

    @classmethod
    async def create_new_template(
        cls, create_graph: CreateGraph
    ) -> autogpt_server.data.graph.Graph:
        return await cls.create_graph(create_graph, is_template=True)

    @classmethod
    async def create_new_graph(
        cls, create_graph: CreateGraph
    ) -> autogpt_server.data.graph.Graph:
        return await cls.create_graph(create_graph, is_template=False)

    @classmethod
    async def update_graph(
        cls, graph_id: str, graph: autogpt_server.data.graph.Graph
    ) -> autogpt_server.data.graph.Graph:
        # Sanity check
        if graph.graph_id and graph.graph_id != graph_id:
            raise HTTPException(400, detail="Graph ID does not match ID in URI")

        # Determine new version
        existing_versions = await autogpt_server.data.graph.get_graph_history(graph_id)
        if not existing_versions:
            raise HTTPException(400, detail=f"Unknown graph ID '{graph_id}'")
        graph.version = max(g.version for g in existing_versions) + 1

        if not graph.is_template:
            graph.is_active = True
        else:
            graph.is_active = False

        # Assign new UUIDs to all nodes and links
        id_map = {node.id: str(uuid.uuid4()) for node in graph.nodes}
        for node in graph.nodes:
            node.id = id_map[node.id]
        for link in graph.links:
            link.source_id = id_map[link.source_id]
            link.sink_id = id_map[link.sink_id]

        new_graph_version = await autogpt_server.data.graph.create_graph(graph)

        if new_graph_version.is_active:
            # Ensure new version is the only active version
            await autogpt_server.data.graph.deactivate_other_graph_versions(
                graph_id=graph_id, except_version=new_graph_version.version
            )

        return new_graph_version

    @classmethod
    async def create_graph(
        cls, create_graph: CreateGraph, is_template: bool
    ) -> autogpt_server.data.graph.Graph:
        if create_graph.graph:
            graph = create_graph.graph
        elif create_graph.template_id:
            graph = await autogpt_server.data.graph.get_graph(
                create_graph.template_id, create_graph.version
            )
            graph.version = 1
        else:
            raise HTTPException(
                status_code=400, detail="Either graph or template_id must be provided."
            )

        # TODO: replace uuid generation here to DB generated uuids.
        graph.graph_id = str(uuid.uuid4())

        graph.is_template = is_template
        if not graph.is_template:
            graph.is_active = True
        else:
            graph.is_active = False

        id_map = {node.id: str(uuid.uuid4()) for node in graph.nodes}

        for node in graph.nodes:
            node.id = id_map[node.id]

        for link in graph.links:
            link.source_id = id_map[link.source_id]
            link.sink_id = id_map[link.sink_id]

        return await autogpt_server.data.graph.create_graph(graph)

    async def execute_graph(
        self, graph_id: str, node_input: dict[Any, Any]
    ) -> dict[Any, Any]:
        try:
            return self.execution_manager_client.add_execution(graph_id, node_input)  # type: ignore
        except Exception as e:
            msg = e.__str__().encode().decode("unicode_escape")
            raise HTTPException(status_code=400, detail=msg)

    @classmethod
    async def list_graph_runs(
        cls, graph_id: str, graph_version: int | None = None
    ) -> list[str]:
        graph = await autogpt_server.data.graph.get_graph(graph_id, graph_version)
        if not graph:
            rev = "" if graph_version is None else f" v{graph_version}"
            raise HTTPException(
                status_code=404, detail=f"Agent #{graph_id}{rev} not found."
            )

        return await execution.list_executions(graph_id, graph_version)

    @classmethod
    async def get_run_execution_results(
        cls, graph_id: str, run_id: str
    ) -> list[execution.ExecutionResult]:
        graph = await autogpt_server.data.graph.get_graph(graph_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Agent #{graph_id} not found.")

        return await execution.get_execution_results(run_id)

    async def create_schedule(
        self, graph_id: str, cron: str, input_data: dict[Any, Any]
    ) -> dict[Any, Any]:
        graph = await autogpt_server.data.graph.get_graph(graph_id)
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
        self.run_and_wait(self.event_queue.put(execution_result))

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
            return {
                "message": "Settings updated successfully",
                "updated_fields": updated_fields,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
