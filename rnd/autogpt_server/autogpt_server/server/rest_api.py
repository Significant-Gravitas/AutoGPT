import inspect
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import wraps
from typing import Annotated, Any, Dict

import uvicorn
from autogpt_libs.auth.middleware import auth_middleware
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from autogpt_server.data import block, db
from autogpt_server.data import execution as execution_db
from autogpt_server.data import graph as graph_db
from autogpt_server.data import user as user_db
from autogpt_server.data.block import BlockInput, CompletedBlockOutput
from autogpt_server.data.credit import get_block_costs, get_user_credit_model
from autogpt_server.data.queue import AsyncEventQueue, AsyncRedisEventQueue
from autogpt_server.data.user import get_or_create_user
from autogpt_server.executor import ExecutionManager, ExecutionScheduler
from autogpt_server.server.model import CreateGraph, SetGraphActiveVersion
from autogpt_server.util.lock import KeyedMutex
from autogpt_server.util.service import AppService, expose, get_service_client
from autogpt_server.util.settings import Config, Settings

from .utils import get_user_id

settings = Settings()


class AgentServer(AppService):
    mutex = KeyedMutex()
    use_redis = True
    _test_dependency_overrides = {}
    _user_credit_model = get_user_credit_model()

    def __init__(self, event_queue: AsyncEventQueue | None = None):
        super().__init__(port=Config().agent_server_port)
        self.event_queue = event_queue or AsyncRedisEventQueue()

    @asynccontextmanager
    async def lifespan(self, _: FastAPI):
        await db.connect()
        self.run_and_wait(self.event_queue.connect())
        await block.initialize_blocks()
        if await user_db.create_default_user(settings.config.enable_auth):
            await graph_db.import_packaged_templates()
        yield
        await self.event_queue.close()
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
        api_router = APIRouter(prefix="/api")
        api_router.dependencies.append(Depends(auth_middleware))

        # Import & Attach sub-routers
        from .integrations import integrations_api_router

        api_router.include_router(integrations_api_router, prefix="/integrations")

        api_router.add_api_route(
            path="/auth/user",
            endpoint=self.get_or_create_user_route,
            methods=["POST"],
        )

        api_router.add_api_route(
            path="/blocks",
            endpoint=self.get_graph_blocks,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/blocks/costs",
            endpoint=self.get_graph_block_costs,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/blocks/{block_id}/execute",
            endpoint=self.execute_graph_block,
            methods=["POST"],
        )
        api_router.add_api_route(
            path="/graphs",
            endpoint=self.get_graphs,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/templates",
            endpoint=self.get_templates,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/graphs",
            endpoint=self.create_new_graph,
            methods=["POST"],
        )
        api_router.add_api_route(
            path="/templates",
            endpoint=self.create_new_template,
            methods=["POST"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}",
            endpoint=self.get_graph,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/templates/{graph_id}",
            endpoint=self.get_template,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}",
            endpoint=self.update_graph,
            methods=["PUT"],
        )
        api_router.add_api_route(
            path="/templates/{graph_id}",
            endpoint=self.update_graph,
            methods=["PUT"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/versions",
            endpoint=self.get_graph_all_versions,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/templates/{graph_id}/versions",
            endpoint=self.get_graph_all_versions,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/versions/{version}",
            endpoint=self.get_graph,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/versions/active",
            endpoint=self.set_graph_active_version,
            methods=["PUT"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/input_schema",
            endpoint=self.get_graph_input_schema,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/execute",
            endpoint=self.execute_graph,
            methods=["POST"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/executions",
            endpoint=self.list_graph_runs,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/executions/{graph_exec_id}",
            endpoint=self.get_graph_run_node_execution_results,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/executions/{graph_exec_id}/stop",
            endpoint=self.stop_graph_run,
            methods=["POST"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/schedules",
            endpoint=self.create_schedule,
            methods=["POST"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/schedules",
            endpoint=self.get_execution_schedules,
            methods=["GET"],
        )
        api_router.add_api_route(
            path="/graphs/schedules/{schedule_id}",
            endpoint=self.update_schedule,
            methods=["PUT"],
        )
        api_router.add_api_route(
            path="/credits",
            endpoint=self.get_user_credits,
            methods=["GET"],
        )

        api_router.add_api_route(
            path="/settings",
            endpoint=self.update_configuration,
            methods=["POST"],
        )

        app.add_exception_handler(500, self.handle_internal_http_error)

        app.include_router(api_router)

        uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)

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
        return get_service_client(ExecutionManager, Config().execution_manager_port)

    @property
    def execution_scheduler_client(self) -> ExecutionScheduler:
        return get_service_client(ExecutionScheduler, Config().execution_scheduler_port)

    @classmethod
    def handle_internal_http_error(cls, request: Request, exc: Exception):
        return JSONResponse(
            content={
                "message": f"{request.method} {request.url.path} failed",
                "error": str(exc),
            },
            status_code=500,
        )

    @classmethod
    async def get_or_create_user_route(cls, user_data: dict = Depends(auth_middleware)):
        user = await get_or_create_user(user_data)
        return user.model_dump()

    @classmethod
    def get_graph_blocks(cls) -> list[dict[Any, Any]]:
        return [v.to_dict() for v in block.get_blocks().values()]

    @classmethod
    def get_graph_block_costs(cls) -> dict[Any, Any]:
        return get_block_costs()

    @classmethod
    def execute_graph_block(
        cls, block_id: str, data: BlockInput
    ) -> CompletedBlockOutput:
        obj = block.get_block(block_id)
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
        cls,
        create_graph: CreateGraph,
        is_template: bool,
        # user_id doesn't have to be annotated like on other endpoints,
        # because create_graph isn't used directly as an endpoint
        user_id: str,
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
    ) -> dict[str, Any]:  # FIXME: add proper return type
        try:
            graph_exec = self.execution_manager_client.add_execution(
                graph_id, node_input, user_id=user_id
            )
            return {"id": graph_exec["graph_exec_id"]}
        except Exception as e:
            msg = e.__str__().encode().decode("unicode_escape")
            raise HTTPException(status_code=400, detail=msg)

    async def stop_graph_run(
        self, graph_exec_id: str, user_id: Annotated[str, Depends(get_user_id)]
    ) -> list[execution_db.ExecutionResult]:
        if not await execution_db.get_graph_execution(graph_exec_id, user_id):
            raise HTTPException(
                404, detail=f"Agent execution #{graph_exec_id} not found"
            )

        self.execution_manager_client.cancel_execution(graph_exec_id)

        # Retrieve & return canceled graph execution in its final state
        return await execution_db.get_execution_results(graph_exec_id)

    @classmethod
    async def get_graph_input_schema(
        cls,
        graph_id: str,
        user_id: Annotated[str, Depends(get_user_id)],
    ) -> list[graph_db.InputSchemaItem]:
        try:
            graph = await graph_db.get_graph(graph_id, user_id=user_id)
            return graph.get_input_schema() if graph else []
        except Exception:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")

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

        return await execution_db.list_executions(graph_id, graph_version)

    @classmethod
    async def get_graph_run_status(
        cls,
        graph_id: str,
        graph_exec_id: str,
        user_id: Annotated[str, Depends(get_user_id)],
    ) -> execution_db.ExecutionStatus:
        graph = await graph_db.get_graph(graph_id, user_id=user_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")

        execution = await execution_db.get_graph_execution(graph_exec_id, user_id)
        if not execution:
            raise HTTPException(
                status_code=404, detail=f"Execution #{graph_exec_id} not found."
            )

        return execution.executionStatus

    @classmethod
    async def get_graph_run_node_execution_results(
        cls,
        graph_id: str,
        graph_exec_id: str,
        user_id: Annotated[str, Depends(get_user_id)],
    ) -> list[execution_db.ExecutionResult]:
        graph = await graph_db.get_graph(graph_id, user_id=user_id)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")

        return await execution_db.get_execution_results(graph_exec_id)

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
        execution_scheduler.update_schedule(schedule_id, is_enabled, user_id=user_id)
        return {"id": schedule_id}

    async def get_user_credits(
        self, user_id: Annotated[str, Depends(get_user_id)]
    ) -> dict[str, int]:
        return {"credits": await self._user_credit_model.get_or_refill_credit(user_id)}

    def get_execution_schedules(
        self, graph_id: str, user_id: Annotated[str, Depends(get_user_id)]
    ) -> dict[str, str]:
        execution_scheduler = self.execution_scheduler_client
        return execution_scheduler.get_execution_schedules(graph_id, user_id)

    @expose
    def send_execution_update(self, execution_result_dict: dict[Any, Any]):
        execution_result = execution_db.ExecutionResult(**execution_result_dict)
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
                if hasattr(settings.config, key):
                    setattr(settings.config, key, value)
                    updated_fields["config"].append(key)
            for key, value in updated_settings.get("secrets", {}).items():
                if hasattr(settings.secrets, key):
                    setattr(settings.secrets, key, value)
                    updated_fields["secrets"].append(key)
            settings.save()
            return {
                "message": "Settings updated successfully",
                "updated_fields": updated_fields,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
