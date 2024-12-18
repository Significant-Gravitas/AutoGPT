import contextlib
import logging
import typing

import fastapi
import fastapi.responses
import starlette.middleware.cors
import uvicorn
from autogpt_libs.feature_flag.client import (
    initialize_launchdarkly,
    shutdown_launchdarkly,
)

import backend.data.block
import backend.data.db
import backend.data.graph
import backend.data.user
import backend.server.routers.v1
import backend.server.v2.library.routes
import backend.server.v2.store.routes
import backend.util.service
import backend.util.settings

settings = backend.util.settings.Settings()
logger = logging.getLogger(__name__)

logging.getLogger("autogpt_libs").setLevel(logging.INFO)


@contextlib.contextmanager
def launch_darkly_context():
    if settings.config.app_env != backend.util.settings.AppEnvironment.LOCAL:
        initialize_launchdarkly()
        try:
            yield
        finally:
            shutdown_launchdarkly()
    else:
        yield


@contextlib.asynccontextmanager
async def lifespan_context(app: fastapi.FastAPI):
    await backend.data.db.connect()
    await backend.data.block.initialize_blocks()
    await backend.data.user.migrate_and_encrypt_user_integrations()
    await backend.data.graph.fix_llm_provider_credentials()
    with launch_darkly_context():
        yield
    await backend.data.db.disconnect()


docs_url = (
    "/docs"
    if settings.config.app_env == backend.util.settings.AppEnvironment.LOCAL
    else None
)

app = fastapi.FastAPI(
    title="AutoGPT Agent Server",
    description=(
        "This server is used to execute agents that are created by the "
        "AutoGPT system."
    ),
    summary="AutoGPT Agent Server",
    version="0.1",
    lifespan=lifespan_context,
    docs_url=docs_url,
)


def handle_internal_http_error(status_code: int = 500, log_error: bool = True):
    def handler(request: fastapi.Request, exc: Exception):
        if log_error:
            logger.exception(f"{request.method} {request.url.path} failed: {exc}")
        return fastapi.responses.JSONResponse(
            content={
                "message": f"{request.method} {request.url.path} failed",
                "detail": str(exc),
            },
            status_code=status_code,
        )

    return handler


app.add_exception_handler(ValueError, handle_internal_http_error(400))
app.add_exception_handler(Exception, handle_internal_http_error(500))
app.include_router(backend.server.routers.v1.v1_router, tags=["v1"], prefix="/api")
app.include_router(
    backend.server.v2.store.routes.router, tags=["v2"], prefix="/api/store"
)
app.include_router(
    backend.server.v2.library.routes.router, tags=["v2"], prefix="/api/library"
)


@app.get(path="/health", tags=["health"], dependencies=[])
async def health():
    return {"status": "healthy"}


class AgentServer(backend.util.service.AppProcess):
    def run(self):
        server_app = starlette.middleware.cors.CORSMiddleware(
            app=app,
            allow_origins=settings.config.backend_cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )
        uvicorn.run(
            server_app,
            host=backend.util.settings.Config().agent_api_host,
            port=backend.util.settings.Config().agent_api_port,
        )

    @staticmethod
    async def test_execute_graph(
        graph_id: str, node_input: dict[typing.Any, typing.Any], user_id: str
    ):
        return backend.server.routers.v1.execute_graph(graph_id, node_input, user_id)

    @staticmethod
    async def test_create_graph(
        create_graph: backend.server.routers.v1.CreateGraph,
        user_id: str,
    ):
        return await backend.server.routers.v1.create_new_graph(create_graph, user_id)

    @staticmethod
    async def test_get_graph_run_status(graph_exec_id: str, user_id: str):
        execution = await backend.data.graph.get_execution(
            user_id=user_id, execution_id=graph_exec_id
        )
        if not execution:
            raise ValueError(f"Execution {graph_exec_id} not found")
        return execution.status

    @staticmethod
    async def test_get_graph_run_node_execution_results(
        graph_id: str, graph_exec_id: str, user_id: str
    ):
        return await backend.server.routers.v1.get_graph_run_node_execution_results(
            graph_id, graph_exec_id, user_id
        )

    @staticmethod
    async def test_delete_graph(graph_id: str, user_id: str):
        return await backend.server.routers.v1.delete_graph(graph_id, user_id)

    def set_test_dependency_overrides(self, overrides: dict):
        app.dependency_overrides.update(overrides)
