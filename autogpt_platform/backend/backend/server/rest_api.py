import inspect
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import wraps
from typing import Annotated, Any, Dict

import uvicorn
from autogpt_libs.auth.middleware import auth_middleware
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing_extensions import TypedDict

from backend.data import block, db
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data.block import BlockInput, CompletedBlockOutput
from backend.data.credit import get_block_costs, get_user_credit_model
from backend.data.queue import RedisEventQueue
from backend.data.user import get_or_create_user
from backend.executor import ExecutionManager, ExecutionScheduler
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.server.model import CreateGraph, SetGraphActiveVersion
from backend.util.service import AppService, expose, get_service_client
from backend.util.settings import AppEnvironment, Config, Settings

from .utils import get_user_id

settings = Settings()
logger = logging.getLogger(__name__)


class AgentServer(AppService):
    use_queue = True
    _test_dependency_overrides = {}
    _user_credit_model = get_user_credit_model()

    def __init__(self):
        super().__init__(port=Config().agent_server_port)
        self.event_queue = RedisEventQueue()

    @asynccontextmanager
    async def lifespan(self, _: FastAPI):
        await db.connect()
        self.event_queue.connect()
        await block.initialize_blocks()
        yield
        self.event_queue.close()
        await db.disconnect()

    def run_service(self):
        docs_url = "/docs" if settings.config.app_env == AppEnvironment.LOCAL else None
        app = FastAPI(
            title="AutoGPT Agent Server",
            description=(
                "This server is used to execute agents that are created by the "
                "AutoGPT system."
            ),
            summary="AutoGPT Agent Server",
            version="0.1",
            lifespan=self.lifespan,
            docs_url=docs_url,
        )

        if self._test_dependency_overrides:
            app.dependency_overrides.update(self._test_dependency_overrides)

        logger.debug(
            f"FastAPI CORS allow origins: {Config().backend_cors_allow_origins}"
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=Config().backend_cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

        # Define the API routes
        api_router = APIRouter(prefix="/api")
        api_router.dependencies.append(Depends(auth_middleware))

        # Import & Attach sub-routers
        import backend.server.integrations.router
        import backend.server.routers.analytics

        api_router.include_router(
            backend.server.integrations.router.router,
            prefix="/integrations",
            tags=["integrations"],
            dependencies=[Depends(auth_middleware)],
        )
        self.integration_creds_manager = IntegrationCredentialsManager()

        api_router.include_router(
            backend.server.routers.analytics.router,
            prefix="/analytics",
            tags=["analytics"],
            dependencies=[Depends(auth_middleware)],
        )

        api_router.add_api_route(
            path="/auth/user",
            endpoint=self.get_or_create_user_route,
            methods=["POST"],
            tags=["auth"],
        )

        api_router.add_api_route(
            path="/blocks",
            endpoint=self.get_graph_blocks,
            methods=["GET"],
            tags=["blocks"],
        )
        api_router.add_api_route(
            path="/blocks/{block_id}/execute",
            endpoint=self.execute_graph_block,
            methods=["POST"],
            tags=["blocks"],
        )
        api_router.add_api_route(
            path="/graphs",
            endpoint=self.get_graphs,
            methods=["GET"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/templates",
            endpoint=self.get_templates,
            methods=["GET"],
            tags=["templates", "graphs"],
        )
        api_router.add_api_route(
            path="/graphs",
            endpoint=self.create_new_graph,
            methods=["POST"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/templates",
            endpoint=self.create_new_template,
            methods=["POST"],
            tags=["templates", "graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}",
            endpoint=self.get_graph,
            methods=["GET"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/templates/{graph_id}",
            endpoint=self.get_template,
            methods=["GET"],
            tags=["templates", "graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}",
            endpoint=self.update_graph,
            methods=["PUT"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/templates/{graph_id}",
            endpoint=self.update_graph,
            methods=["PUT"],
            tags=["templates", "graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}",
            endpoint=self.delete_graph,
            methods=["DELETE"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/versions",
            endpoint=self.get_graph_all_versions,
            methods=["GET"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/templates/{graph_id}/versions",
            endpoint=self.get_graph_all_versions,
            methods=["GET"],
            tags=["templates", "graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/versions/{version}",
            endpoint=self.get_graph,
            methods=["GET"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/versions/active",
            endpoint=self.set_graph_active_version,
            methods=["PUT"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/input_schema",
            endpoint=self.get_graph_input_schema,
            methods=["GET"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/execute",
            endpoint=self.execute_graph,
            methods=["POST"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/executions",
            endpoint=self.list_graph_runs,
            methods=["GET"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/executions/{graph_exec_id}",
            endpoint=self.get_graph_run_node_execution_results,
            methods=["GET"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/executions/{graph_exec_id}/stop",
            endpoint=self.stop_graph_run,
            methods=["POST"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/schedules",
            endpoint=self.create_schedule,
            methods=["POST"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/graphs/{graph_id}/schedules",
            endpoint=self.get_execution_schedules,
            methods=["GET"],
            tags=["graphs"],
        )
        api_router.add_api_route(
            path="/graphs/schedules/{schedule_id}",
            endpoint=self.update_schedule,
            methods=["PUT"],
            tags=["graphs"],
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
            tags=["settings"],
        )

        app.add_exception_handler(500, self.handle_internal_http_error)

        app.include_router(api_router)

        uvicorn.run(
            app,
            host=Config().agent_api_host,
            port=Config().agent_api_port,
            log_config=None,
        )

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
        blocks = block.get_blocks()
        costs = get_block_costs()
        return [{**b.to_dict(), "costs": costs.get(b.id, [])} for b in blocks.values()]

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
        cls,
        user_id: Annotated[str, Depends(get_user_id)],
        with_runs: bool = False,
    ) -> list[graph_db.GraphMeta]:
        return await graph_db.get_graphs_meta(
            include_executions=with_runs, filter_by="active", user_id=user_id
        )

    @classmethod
    async def get_templates(
        cls, user_id: Annotated[str, Depends(get_user_id)]
    ) -> list[graph_db.GraphMeta]:
        return await graph_db.get_graphs_meta(filter_by="template", user_id=user_id)

    @classmethod
    async def get_graph(
        cls,
        graph_id: str,
        user_id: Annotated[str, Depends(get_user_id)],
        version: int | None = None,
        hide_credentials: bool = False,
    ) -> graph_db.Graph:
        graph = await graph_db.get_graph(
            graph_id, version, user_id=user_id, hide_credentials=hide_credentials
        )
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

    class DeleteGraphResponse(TypedDict):
        version_counts: int

    @classmethod
    async def delete_graph(
        cls, graph_id: str, user_id: Annotated[str, Depends(get_user_id)]
    ) -> DeleteGraphResponse:
        return {
            "version_counts": await graph_db.delete_graph(graph_id, user_id=user_id)
        }

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
        self.event_queue.put(execution_result)

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
