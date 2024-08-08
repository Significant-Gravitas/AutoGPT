import asyncio
import inspect
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import wraps
from typing import Annotated, Any, Dict

import uvicorn
from autogpt_libs.auth.jwt_utils import parse_jwt_token
from autogpt_libs.auth.middleware import auth_middleware
from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import autogpt_server.server.ws_api
from autogpt_server.data import block, db
from autogpt_server.data import graph as graph_db
from autogpt_server.data import user as user_db
from autogpt_server.data.block import BlockInput, CompletedBlockOutput
from autogpt_server.data.execution import (
    ExecutionResult,
    get_execution_results,
    list_executions,
)
from autogpt_server.data.user import get_or_create_user
from autogpt_server.executor import ExecutionManager, ExecutionScheduler
from autogpt_server.server.conn_manager import ConnectionManager
from autogpt_server.server.model import (
    CreateGraph,
    Methods,
    SetGraphActiveVersion,
    WsMessage,
)
from autogpt_server.util.lock import KeyedMutex
from autogpt_server.util.service import AppService, expose, get_service_client
from autogpt_server.util.settings import Settings

settings = Settings()


def get_user_id(payload: dict = Depends(auth_middleware)) -> str:
    if not payload:
        # This handles the case when authentication is disabled
        return "3e53486c-cf57-477e-ba2a-cb02dc828e1a"

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    return user_id


class AgentServer(AppService):
    event_queue: asyncio.Queue[ExecutionResult] = asyncio.Queue()
    manager = ConnectionManager()
    mutex = KeyedMutex()
    use_db = False
    _test_dependency_overrides = {}

    async def event_broadcaster(self):
        while True:
            event: ExecutionResult = await self.event_queue.get()
            await self.manager.send_execution_result(event)

    @asynccontextmanager
    async def lifespan(self, _: FastAPI):
        await db.connect()
        await block.initialize_blocks()
        await graph_db.import_packaged_templates()
        await user_db.create_default_user(settings.config.enable_auth)
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

        if self._test_dependency_overrides:
            app.dependency_overrides.update(self._test_dependency_overrides)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

        # Define the API routes
        router = APIRouter(prefix="/api")
        router.dependencies.append(Depends(auth_middleware))

        router.add_api_route(
            path="/auth/user",
            endpoint=self.get_or_create_user_route,
            methods=["POST"],
        )

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
            path="/graphs",
            endpoint=self.get_graphs,
            methods=["GET"],
        )
        router.add_api_route(
            path="/templates",
            endpoint=self.get_templates,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs",
            endpoint=self.create_new_graph,
            methods=["POST"],
        )
        router.add_api_route(
            path="/templates",
            endpoint=self.create_new_template,
            methods=["POST"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}",
            endpoint=self.get_graph,
            methods=["GET"],
        )
        router.add_api_route(
            path="/templates/{graph_id}",
            endpoint=self.get_template,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}",
            endpoint=self.update_graph,
            methods=["PUT"],
        )
        router.add_api_route(
            path="/templates/{graph_id}",
            endpoint=self.update_graph,
            methods=["PUT"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}/versions",
            endpoint=self.get_graph_all_versions,
            methods=["GET"],
        )
        router.add_api_route(
            path="/templates/{graph_id}/versions",
            endpoint=self.get_graph_all_versions,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}/versions/{version}",
            endpoint=self.get_graph,
            methods=["GET"],
        )
        router.add_api_route(
            path="/graphs/{graph_id}/versions/active",
            endpoint=self.set_graph_active_version,
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

        app.include_router(router)

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):  # type: ignore
            await self.websocket_router(websocket)

        uvicorn.run(app, host="0.0.0.0", port=8000)

    def set_test_dependency_overrides(self, overrides: dict):
        self._test_dependency_overrides = overrides

    def _apply_overrides_to_methods(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "__annotations__"):
                setattr(self, attr_name, self._override_method(attr))

    # TODO: fix this with some proper refactoring of the server
    def _override_method(self, method):
        @wraps(method)
        async def wrapper(*args, **kwargs):
            sig = inspect.signature(method)
            for param_name, param in sig.parameters.items():
                if param.annotation is inspect.Parameter.empty:
                    continue
                if isinstance(param.annotation, Depends) or (  # type: ignore
                    isinstance(param.annotation, type) and issubclass(param.annotation, Depends)  # type: ignore
                ):
                    dependency = param.annotation.dependency if isinstance(param.annotation, Depends) else param.annotation  # type: ignore
                    if dependency in self._test_dependency_overrides:
                        kwargs[param_name] = self._test_dependency_overrides[
                            dependency
                        ]()
            return await method(*args, **kwargs)

        return wrapper

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

    async def authenticate_websocket(self, websocket: WebSocket) -> str:
        if settings.config.enable_auth.lower() == "true":
            token = websocket.query_params.get("token")
            if not token:
                await websocket.close(code=4001, reason="Missing authentication token")
                return ""

            try:
                payload = parse_jwt_token(token)
                user_id = payload.get("sub")
                if not user_id:
                    await websocket.close(code=4002, reason="Invalid token")
                    return ""
                return user_id
            except ValueError:
                await websocket.close(code=4003, reason="Invalid token")
                return ""
        else:
            return "3e53486c-cf57-477e-ba2a-cb02dc828e1a"

    async def websocket_router(self, websocket: WebSocket):
        user_id = await self.authenticate_websocket(websocket)
        if not user_id:
            return
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
                elif message.method == Methods.EXECUTION_EVENT:
                    print("Execution event received")
                elif message.method == Methods.GET_BLOCKS:
                    data = self.get_graph_blocks()
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.GET_BLOCKS,
                            success=True,
                            data=data,
                        ).model_dump_json()
                    )
                elif message.method == Methods.EXECUTE_BLOCK:
                    assert isinstance(message.data, dict), "Data must be a dictionary"
                    data = self.execute_graph_block(
                        message.data["block_id"], message.data["data"]
                    )
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.EXECUTE_BLOCK,
                            success=True,
                            data=data,
                        ).model_dump_json()
                    )
                elif message.method == Methods.GET_GRAPHS:
                    data = await self.get_graphs(user_id=user_id)
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.GET_GRAPHS,
                            success=True,
                            data=data,
                        ).model_dump_json()
                    )
                    print("Get graphs request received")
                elif message.method == Methods.GET_GRAPH:
                    assert isinstance(message.data, dict), "Data must be a dictionary"
                    data = await self.get_graph(
                        message.data["graph_id"], user_id=user_id
                    )
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.GET_GRAPH,
                            success=True,
                            data=data.model_dump(),
                        ).model_dump_json()
                    )
                    print("Get graph request received")
                elif message.method == Methods.CREATE_GRAPH:
                    assert isinstance(message.data, dict), "Data must be a dictionary"
                    create_graph = CreateGraph.model_validate(message.data)
                    data = await self.create_new_graph(create_graph, user_id=user_id)
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.CREATE_GRAPH,
                            success=True,
                            data=data.model_dump(),
                        ).model_dump_json()
                    )

                    print("Create graph request received")
                elif message.method == Methods.RUN_GRAPH:
                    assert isinstance(message.data, dict), "Data must be a dictionary"
                    data = await self.execute_graph(
                        message.data["graph_id"], message.data["data"], user_id=user_id
                    )
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.RUN_GRAPH,
                            success=True,
                            data=data,
                        ).model_dump_json()
                    )

                    print("Run graph request received")
                elif message.method == Methods.GET_GRAPH_RUNS:
                    assert isinstance(message.data, dict), "Data must be a dictionary"
                    data = await self.list_graph_runs(
                        message.data["graph_id"], user_id=user_id
                    )
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.GET_GRAPH_RUNS,
                            success=True,
                            data=data,
                        ).model_dump_json()
                    )

                    print("Get graph runs request received")
                elif message.method == Methods.CREATE_SCHEDULED_RUN:
                    assert isinstance(message.data, dict), "Data must be a dictionary"
                    data = await self.create_schedule(
                        message.data["graph_id"],
                        message.data["cron"],
                        message.data["data"],
                        user_id=user_id,
                    )
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.CREATE_SCHEDULED_RUN,
                            success=True,
                            data=data,
                        ).model_dump_json()
                    )

                    print("Create scheduled run request received")
                elif message.method == Methods.GET_SCHEDULED_RUNS:
                    assert isinstance(message.data, dict), "Data must be a dictionary"
                    data = self.get_execution_schedules(
                        message.data["graph_id"], user_id=user_id
                    )
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.GET_SCHEDULED_RUNS,
                            success=True,
                            data=data,
                        ).model_dump_json()
                    )
                    print("Get scheduled runs request received")
                elif message.method == Methods.UPDATE_SCHEDULED_RUN:
                    assert isinstance(message.data, dict), "Data must be a dictionary"
                    data = self.update_schedule(
                        message.data["schedule_id"], message.data, user_id=user_id
                    )
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.UPDATE_SCHEDULED_RUN,
                            success=True,
                            data=data,
                        ).model_dump_json()
                    )

                    print("Update scheduled run request received")
                elif message.method == Methods.UPDATE_CONFIG:
                    assert isinstance(message.data, dict), "Data must be a dictionary"
                    data = self.update_configuration(message.data)
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.UPDATE_CONFIG,
                            success=True,
                            data=data,
                        ).model_dump_json()
                    )

                    print("Update config request received")
                elif message.method == Methods.ERROR:
                    print("Error message received")
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
    async def get_or_create_user_route(cls, user_data: dict = Depends(auth_middleware)):
        user = await get_or_create_user(user_data)
        return user.model_dump()

    @classmethod
    def get_graph_blocks(cls) -> list[dict[Any, Any]]:
        return [v.to_dict() for v in block.get_blocks().values()]  # type: ignore

    @classmethod
    def execute_graph_block(
        cls, block_id: str, data: BlockInput
    ) -> CompletedBlockOutput:
        obj = block.get_block(block_id)  # type: ignore
        if not obj:
            raise HTTPException(status_code=404, detail=f"Block #{block_id} not found.")

        output = defaultdict(list)
        for name, data in obj.execute(data):
            output[name].append(data)
        return output

    @classmethod
    async def get_graphs(
        cls, user_id: Annotated[str, Depends(get_user_id)]
    ) -> list[graph_db.GraphMeta]:
        return await graph_db.get_graphs_meta(filter_by="active", user_id=user_id)

    @classmethod
    async def get_templates(cls) -> list[graph_db.GraphMeta]:
        return await graph_db.get_graphs_meta(filter_by="template")

    @classmethod
    async def get_graph(
        cls,
        graph_id: str,
        user_id: Annotated[str, Depends(get_user_id)],
        version: int | None = None,
    ) -> graph_db.Graph:
        graph = await graph_db.get_graph(graph_id, version, user_id=user_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
        return graph

    @classmethod
    async def get_template(
        cls, graph_id: str, version: int | None = None
    ) -> graph_db.Graph:
        graph = await graph_db.get_graph(graph_id, version, template=True)
        if not graph:
            raise HTTPException(
                status_code=404, detail=f"Template #{graph_id} not found."
            )
        return graph

    @classmethod
    async def get_graph_all_versions(
        cls, graph_id: str, user_id: Annotated[str, Depends(get_user_id)]
    ) -> list[graph_db.Graph]:
        graphs = await graph_db.get_graph_all_versions(graph_id, user_id=user_id)
        if not graphs:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
        return graphs

    @classmethod
    async def create_new_graph(
        cls, create_graph: CreateGraph, user_id: Annotated[str, Depends(get_user_id)]
    ) -> graph_db.Graph:
        return await cls.create_graph(create_graph, is_template=False, user_id=user_id)

    @classmethod
    async def create_new_template(
        cls, create_graph: CreateGraph, user_id: Annotated[str, Depends(get_user_id)]
    ) -> graph_db.Graph:
        return await cls.create_graph(create_graph, is_template=True, user_id=user_id)

    @classmethod
    async def create_graph(
        cls, create_graph: CreateGraph, is_template: bool, user_id: str
    ) -> graph_db.Graph:
        if create_graph.graph:
            graph = create_graph.graph
        elif create_graph.template_id:
            # Create a new graph from a template
            graph = await graph_db.get_graph(
                create_graph.template_id,
                create_graph.template_version,
                template=True,
                user_id=user_id,
            )
            if not graph:
                raise HTTPException(
                    400, detail=f"Template #{create_graph.template_id} not found"
                )
            graph.version = 1
        else:
            raise HTTPException(
                status_code=400, detail="Either graph or template_id must be provided."
            )

        graph.is_template = is_template
        graph.is_active = not is_template
        graph.reassign_ids(reassign_graph_id=True)

        return await graph_db.create_graph(graph, user_id=user_id)

    @classmethod
    async def update_graph(
        cls,
        graph_id: str,
        graph: graph_db.Graph,
        user_id: Annotated[str, Depends(get_user_id)],
    ) -> graph_db.Graph:
        # Sanity check
        if graph.id and graph.id != graph_id:
            raise HTTPException(400, detail="Graph ID does not match ID in URI")

        # Determine new version
        existing_versions = await graph_db.get_graph_all_versions(
            graph_id, user_id=user_id
        )
        if not existing_versions:
            raise HTTPException(404, detail=f"Graph #{graph_id} not found")
        latest_version_number = max(g.version for g in existing_versions)
        graph.version = latest_version_number + 1

        latest_version_graph = next(
            v for v in existing_versions if v.version == latest_version_number
        )
        if latest_version_graph.is_template != graph.is_template:
            raise HTTPException(
                400, detail="Changing is_template on an existing graph is forbidden"
            )
        graph.is_active = not graph.is_template
        graph.reassign_ids()

        new_graph_version = await graph_db.create_graph(graph, user_id=user_id)

        if new_graph_version.is_active:
            # Ensure new version is the only active version
            await graph_db.set_graph_active_version(
                graph_id=graph_id, version=new_graph_version.version, user_id=user_id
            )

        return new_graph_version

    @classmethod
    async def set_graph_active_version(
        cls,
        graph_id: str,
        request_body: SetGraphActiveVersion,
        user_id: Annotated[str, Depends(get_user_id)],
    ):
        new_active_version = request_body.active_graph_version
        if not await graph_db.get_graph(graph_id, new_active_version, user_id=user_id):
            raise HTTPException(
                404, f"Graph #{graph_id} v{new_active_version} not found"
            )
        await graph_db.set_graph_active_version(
            graph_id=graph_id,
            version=request_body.active_graph_version,
            user_id=user_id,
        )

    async def execute_graph(
        self,
        graph_id: str,
        node_input: dict[Any, Any],
        user_id: Annotated[str, Depends(get_user_id)],
    ) -> dict[Any, Any]:
        try:
            return self.execution_manager_client.add_execution(
                graph_id, node_input, user_id=user_id
            )
        except Exception as e:
            msg = e.__str__().encode().decode("unicode_escape")
            raise HTTPException(status_code=400, detail=msg)

    @classmethod
    async def list_graph_runs(
        cls,
        graph_id: str,
        user_id: Annotated[str, Depends(get_user_id)],
        graph_version: int | None = None,
    ) -> list[str]:
        graph = await graph_db.get_graph(graph_id, graph_version, user_id=user_id)
        if not graph:
            rev = "" if graph_version is None else f" v{graph_version}"
            raise HTTPException(
                status_code=404, detail=f"Agent #{graph_id}{rev} not found."
            )

        return await list_executions(graph_id, graph_version)

    @classmethod
    async def get_run_execution_results(
        cls, graph_id: str, run_id: str, user_id: Annotated[str, Depends(get_user_id)]
    ) -> list[ExecutionResult]:
        graph = await graph_db.get_graph(graph_id, user_id=user_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")

        return await get_execution_results(run_id)

    async def create_schedule(
        self,
        graph_id: str,
        cron: str,
        input_data: dict[Any, Any],
        user_id: Annotated[str, Depends(get_user_id)],
    ) -> dict[Any, Any]:
        graph = await graph_db.get_graph(graph_id, user_id=user_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")
        execution_scheduler = self.execution_scheduler_client
        return {
            "id": execution_scheduler.add_execution_schedule(
                graph_id, graph.version, cron, input_data, user_id=user_id
            )
        }

    def update_schedule(
        self,
        schedule_id: str,
        input_data: dict[Any, Any],
        user_id: Annotated[str, Depends(get_user_id)],
    ) -> dict[Any, Any]:
        execution_scheduler = self.execution_scheduler_client
        is_enabled = input_data.get("is_enabled", False)
        execution_scheduler.update_schedule(schedule_id, is_enabled, user_id=user_id)  # type: ignore
        return {"id": schedule_id}

    def get_execution_schedules(
        self, graph_id: str, user_id: Annotated[str, Depends(get_user_id)]
    ) -> dict[str, str]:
        execution_scheduler = self.execution_scheduler_client
        return execution_scheduler.get_execution_schedules(graph_id, user_id)  # type: ignore

    @expose
    def send_execution_update(self, execution_result_dict: dict[Any, Any]):
        execution_result = ExecutionResult(**execution_result_dict)
        self.run_and_wait(self.event_queue.put(execution_result))

    @expose
    def acquire_lock(self, key: Any):
        self.mutex.lock(key)

    @expose
    def release_lock(self, key: Any):
        self.mutex.unlock(key)

    @classmethod
    def update_configuration(
        cls,
        updated_settings: Annotated[
            Dict[str, Any],
            Body(
                examples=[
                    {
                        "config": {
                            "num_graph_workers": 10,
                            "num_node_workers": 10,
                        }
                    }
                ]
            ),
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
